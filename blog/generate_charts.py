"""Generate frozen evidence and charts for the options-RV blog post.

The script reruns the project pipeline against the committed offline inputs, then
copies only the blog's evidence tables and two publication figures into
``blog/``. It does not alter product outputs or the source data.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd
from options_rv.pipeline.offline_research import run_offline_research

BLOG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BLOG_ROOT.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = BLOG_ROOT / "data"
IMAGE_DIR = BLOG_ROOT / "images"
TARGET_COLUMN = "annualized_forward_variance_5d"
MODEL_COLORS = {
    "Realized": "#e8edf7",
    "Persistence": "#25c2c7",
    "ATM IV squared": "#f2a65a",
    "Ridge": "#7e8cff",
}


def generate_blog_evidence() -> None:
    """Run the offline study and write frozen tables plus publication charts.

    Returns
    -------
    None
        Files are written under ``blog/data`` and ``blog/images``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="options-rv-blog-") as temporary_directory:
        results = run_offline_research(
            raw_dir=RAW_DIR,
            output_dir=Path(temporary_directory),
            symbol="SPY",
            horizon_days=5,
            annualization_factor=252,
        )

    predictions = results["predictions"].copy()
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    test_predictions = predictions.loc[predictions["split"] == "test"].copy()
    test_metrics = results["metrics"].loc[results["metrics"]["split"] == "test"].copy()

    # Freeze the exact rows behind the article so a reader can audit each plot.
    test_predictions.to_csv(DATA_DIR / "test_predictions.csv", index=False)
    test_metrics.to_csv(DATA_DIR / "test_metrics.csv", index=False)

    _plot_test_forecasts(test_predictions=test_predictions)
    _plot_relative_mae(test_metrics=test_metrics)


def _plot_test_forecasts(test_predictions: pd.DataFrame) -> None:
    """Plot realized and forecast variance through the held-out test period.

    Parameters
    ----------
    test_predictions
        Test rows with dates, realized variance, and all three model forecasts.

    Returns
    -------
    None
        The figure is saved as ``blog/images/01_test_forecasts.png``.
    """
    series_columns = {
        "Realized": TARGET_COLUMN,
        "Persistence": "persistence_prediction",
        "ATM IV squared": "atm_iv_prediction",
        "Ridge": "ridge_prediction_variance",
    }

    figure, axis = plt.subplots(figsize=(13, 7), dpi=180, constrained_layout=True)
    figure.patch.set_facecolor("#071426")
    axis.set_facecolor("#0b1c31")

    for label, column in series_columns.items():
        axis.plot(
            test_predictions["trade_date"],
            test_predictions[column],
            color=MODEL_COLORS[label],
            linewidth=2.2 if label == "Realized" else 1.7,
            alpha=0.95,
            label=label,
        )

    # A log scale is necessary because the synthetic ATM-IV forecast sits three
    # orders of magnitude above the realized-variance target.
    axis.set_yscale("log")
    axis.set_title(
        "Held-out five-day variance forecasts",
        color="white",
        fontsize=17,
        pad=14,
    )
    axis.set_xlabel("Trade date", color="#c9d4e5")
    axis.set_ylabel("Annualized variance (log scale)", color="#c9d4e5")
    axis.tick_params(colors="#aebbd0")
    axis.grid(color="#406080", alpha=0.22, linewidth=0.8)
    for spine in axis.spines.values():
        spine.set_color("#35506d")
    legend = axis.legend(frameon=True, ncol=2)
    legend.get_frame().set_facecolor("#10253d")
    legend.get_frame().set_edgecolor("#35506d")
    for text in legend.get_texts():
        text.set_color("white")

    figure.savefig(
        IMAGE_DIR / "01_test_forecasts.png",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def _plot_relative_mae(test_metrics: pd.DataFrame) -> None:
    """Plot test mean absolute error relative to the persistence baseline.

    Parameters
    ----------
    test_metrics
        Test metric rows with one observation per model.

    Returns
    -------
    None
        The figure is saved as ``blog/images/02_relative_mae.png``.
    """
    metric_frame = test_metrics.loc[:, ["model", "mae"]].copy()
    persistence_mae = float(
        metric_frame.loc[metric_frame["model"] == "persistence", "mae"].iloc[0]
    )
    metric_frame["mae_multiple"] = metric_frame["mae"] / persistence_mae
    display_names = {
        "persistence": "Persistence",
        "atm_iv": "ATM IV squared",
        "ridge": "Ridge",
    }
    metric_frame["label"] = metric_frame["model"].map(display_names)

    figure, axis = plt.subplots(figsize=(10, 6.5), dpi=180, constrained_layout=True)
    figure.patch.set_facecolor("#071426")
    axis.set_facecolor("#0b1c31")
    colors = [MODEL_COLORS[label] for label in metric_frame["label"]]
    bars = axis.bar(
        metric_frame["label"],
        metric_frame["mae_multiple"],
        color=colors,
        width=0.58,
    )

    axis.set_yscale("log")
    axis.axhline(1.0, color="#e8edf7", linestyle="--", linewidth=1.1, alpha=0.7)
    axis.set_title(
        "Test MAE relative to persistence",
        color="white",
        fontsize=17,
        pad=14,
    )
    axis.set_ylabel("MAE multiple (log scale; lower is better)", color="#c9d4e5")
    axis.tick_params(colors="#aebbd0")
    axis.grid(axis="y", color="#406080", alpha=0.22, linewidth=0.8)
    for spine in axis.spines.values():
        spine.set_color("#35506d")

    for bar, value in zip(bars, metric_frame["mae_multiple"], strict=True):
        label = f"{value:,.0f}x" if value >= 100.0 else f"{value:.2f}x"
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.16,
            label,
            ha="center",
            va="bottom",
            color="white",
            fontsize=11,
        )

    figure.savefig(
        IMAGE_DIR / "02_relative_mae.png",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


if __name__ == "__main__":
    generate_blog_evidence()

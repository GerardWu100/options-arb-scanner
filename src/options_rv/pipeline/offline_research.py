"""Offline end-to-end research orchestration.

This module stitches together loader, target, feature, model, and evaluation
components into one reproducible workflow suitable for CLI and notebook usage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from options_rv.evaluation.metrics import (
    mean_absolute_error,
    qlike,
    root_mean_squared_error,
)
from options_rv.evaluation.splits import build_chronological_split_masks
from options_rv.features.option_surface import build_option_surface_features
from options_rv.io.local_loader import load_raw_data
from options_rv.models.baselines import (
    compute_atm_iv_baseline,
    compute_persistence_baseline,
)
from options_rv.models.train import train_ridge_regression
from options_rv.targets.realized_variance import build_forward_realized_variance_target


FEATURE_COLUMNS: list[str] = [
    "atm_iv_30d",
    "term_slope_60d_minus_30d",
    "downside_skew_30d",
    "avg_spread_ratio",
    "total_open_interest",
    "trailing_annualized_variance_20d",
]

# Model name paired with its variance-level prediction column in the output frame.
MODEL_PREDICTION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("persistence", "persistence_prediction"),
    ("atm_iv", "atm_iv_prediction"),
    ("ridge", "ridge_prediction_variance"),
)


def run_offline_research(
    raw_dir: Path,
    output_dir: Path,
    symbol: str = "SPY",
    horizon_days: int = 5,
    annualization_factor: int = 252,
) -> dict[str, pd.DataFrame]:
    """Run full offline research workflow and write output artifacts.

    Parameters
    ----------
    raw_dir
        Raw-data directory with contract files.
    output_dir
        Destination directory for output tables and figures.
    symbol
        Target symbol for the study.
    horizon_days
        Forecast horizon in trading days.
    annualization_factor
        Annualization factor for daily data.

    Returns
    -------
    dict[str, pd.DataFrame]
        In-memory artifact map keyed by artifact name.
    """
    raw_data_bundle = load_raw_data(
        raw_dir=raw_dir,
        required_symbol=symbol,
        minimum_trading_days=40,
    )

    target_frame = build_forward_realized_variance_target(
        underlying_daily=raw_data_bundle.underlying_daily,
        horizon_days=horizon_days,
        annualization_factor=annualization_factor,
        use_log_target=True,
        trailing_window_days=20,
    )

    feature_frame = build_option_surface_features(
        options_quotes=raw_data_bundle.options_quotes,
        underlying_daily=raw_data_bundle.underlying_daily,
        trailing_variance_frame=target_frame,
    )

    target_level_column = f"annualized_forward_variance_{horizon_days}d"
    target_log_column = f"target_log_annualized_forward_variance_{horizon_days}d"
    panel_frame = feature_frame.merge(
        target_frame.loc[
            :, ["symbol", "trade_date", target_level_column, target_log_column]
        ],
        on=["symbol", "trade_date"],
        how="inner",
        validate="one_to_one",
    )

    # Supervised rows require complete features and both level and log targets.
    panel_frame = panel_frame.dropna(
        subset=FEATURE_COLUMNS + [target_level_column, target_log_column]
    ).reset_index(drop=True)

    # A horizon-length purge prevents adjacent split labels from sharing future
    # returns. For h=5, the last five train and validation rows are excluded.
    split_labels = build_chronological_split_masks(
        frame=panel_frame,
        purge_gap_rows=horizon_days,
    )

    persistence_prediction = compute_persistence_baseline(panel_frame)
    atm_iv_prediction = compute_atm_iv_baseline(panel_frame)

    ridge_result = train_ridge_regression(
        feature_target_frame=panel_frame,
        split_labels=split_labels,
        feature_columns=FEATURE_COLUMNS,
        target_column=target_log_column,
        ridge_alpha=1.0,
    )

    prediction_frame = panel_frame.loc[
        :, ["symbol", "trade_date", target_level_column, target_log_column]
    ].copy()
    prediction_frame["split"] = split_labels
    prediction_frame["persistence_prediction"] = persistence_prediction
    prediction_frame["atm_iv_prediction"] = atm_iv_prediction
    prediction_frame["ridge_prediction_log"] = ridge_result.predictions[
        "prediction"
    ].to_numpy()
    # Ridge is fit in log-variance space; evaluation metrics use variance level.
    prediction_frame["ridge_prediction_variance"] = np.exp(
        prediction_frame["ridge_prediction_log"]
    )

    metric_frame = _build_metric_table(
        prediction_frame=prediction_frame,
        target_level_column=target_level_column,
        split_column="split",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel_frame.to_csv(tables_dir / "feature_target_panel.csv", index=False)
    metric_frame.to_csv(tables_dir / "evaluation_metrics.csv", index=False)
    ridge_result.coefficients.to_csv(tables_dir / "ridge_coefficients.csv", index=False)
    prediction_frame.to_csv(tables_dir / "predictions.csv", index=False)

    _write_prediction_scatter_figure(
        prediction_frame=prediction_frame,
        target_level_column=target_level_column,
        output_path=figures_dir / "prediction_scatter.png",
    )
    _write_markdown_summary(
        output_path=output_dir / "summary.md",
        metric_frame=metric_frame,
        target_level_column=target_level_column,
    )

    return {
        "feature_target_panel": panel_frame,
        "metrics": metric_frame,
        "coefficients": ridge_result.coefficients,
        "predictions": prediction_frame,
    }


def _build_metric_table(
    prediction_frame: pd.DataFrame,
    target_level_column: str,
    split_column: str,
) -> pd.DataFrame:
    """Build compact metric summary for each model and split."""
    rows: list[dict[str, object]] = []
    for split_name in ["train", "validation", "test"]:
        split_frame = prediction_frame.loc[prediction_frame[split_column] == split_name]
        if split_frame.empty:
            continue

        actual = split_frame[target_level_column]
        for model_name, prediction_column in MODEL_PREDICTION_COLUMNS:
            predicted = split_frame[prediction_column]
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "rmse": root_mean_squared_error(actual=actual, predicted=predicted),
                    "mae": mean_absolute_error(actual=actual, predicted=predicted),
                    "qlike": qlike(
                        actual_variance=actual, predicted_variance=predicted
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_prediction_scatter_figure(
    prediction_frame: pd.DataFrame,
    target_level_column: str,
    output_path: Path,
) -> None:
    """Write compact prediction-vs-actual scatter for ridge model."""
    figure, axis = plt.subplots(figsize=(10, 6), dpi=150, constrained_layout=True)
    axis.scatter(
        prediction_frame[target_level_column],
        prediction_frame["ridge_prediction_variance"],
        alpha=0.6,
        label="Ridge predictions",
    )
    axis.set_title("Ridge Forecast vs Realized Forward Variance")
    axis.set_xlabel("Actual annualized forward variance (level)")
    axis.set_ylabel("Predicted annualized forward variance (level)")
    axis.legend()
    figure.savefig(output_path)
    plt.close(figure)


def _write_markdown_summary(
    output_path: Path,
    metric_frame: pd.DataFrame,
    target_level_column: str,
) -> None:
    """Write short markdown summary for quick run inspection."""
    lines: list[str] = []
    lines.append("# Offline Options RV Run Summary")
    lines.append("")
    lines.append(f"- Target column: `{target_level_column}`")
    lines.append("- Models: persistence baseline, ATM IV baseline, ridge regression")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")

    if metric_frame.empty:
        lines.append("No metrics available.")
    else:
        header = "| split | model | rmse | mae | qlike |"
        separator = "| --- | --- | ---: | ---: | ---: |"
        lines.extend([header, separator])
        for _, row in metric_frame.iterrows():
            lines.append(
                "| "
                f"{row['split']} | {row['model']} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['qlike']:.6f} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Integration-style tests for offline orchestration path."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from options_rv.pipeline.offline_research import run_offline_research


def _write_bundle_for_pipeline(raw_dir: Path) -> None:
    """Create synthetic but coherent raw input files for pipeline test."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    trading_days = pd.date_range("2025-01-02", periods=70, freq="B")
    close_values = 600.0 + pd.Series(range(len(trading_days))).astype(float) * 0.5
    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY"] * len(trading_days),
            "trade_date": trading_days,
            "close": close_values,
        }
    )
    underlying_daily.to_parquet(raw_dir / "underlying_daily.parquet", index=False)

    option_rows: list[dict[str, object]] = []
    dte_set = [30, 60]
    strike_offsets = [-0.05, 0.0, 0.05]
    for trade_date, spot_close in zip(trading_days, close_values, strict=False):
        for dte in dte_set:
            expiry_date = trade_date + pd.Timedelta(days=int(dte))
            for option_type in ["c", "p"]:
                for strike_offset in strike_offsets:
                    strike_price = float(spot_close * (1.0 + strike_offset))
                    base_iv = (
                        0.18 + 0.0004 * dte + (0.01 if option_type == "p" else 0.0)
                    )
                    bid_iv = base_iv
                    ask_iv = base_iv + 0.01
                    bid_price = max(0.5, base_iv * 10.0)
                    ask_price = bid_price * 1.05
                    option_rows.append(
                        {
                            "symbol": "SPY",
                            "trade_date": trade_date,
                            "expiry_date": expiry_date,
                            "option_type": option_type,
                            "strike_price": strike_price,
                            "bid": bid_price,
                            "ask": ask_price,
                            "bid_iv": bid_iv,
                            "ask_iv": ask_iv,
                            "open_interest": 1000 + int(dte),
                            "volume": 400 + int(dte),
                        }
                    )

    options_quotes = pd.DataFrame(option_rows)
    options_quotes.to_parquet(raw_dir / "options_quotes.parquet", index=False)

    manifest = {
        "dataset_version": 1,
        "symbols": ["SPY"],
        "date_coverage": {
            "start": str(trading_days.min().date()),
            "end": str(trading_days.max().date()),
        },
        "files": {
            "options_quotes.parquet": {
                "rows": len(options_quotes),
                "columns": [
                    "symbol",
                    "trade_date",
                    "expiry_date",
                    "option_type",
                    "strike_price",
                    "bid",
                    "ask",
                    "bid_iv",
                    "ask_iv",
                    "open_interest",
                    "volume",
                ],
                "compression": "zstd",
            },
            "underlying_daily.parquet": {
                "rows": len(underlying_daily),
                "columns": ["symbol", "trade_date", "close"],
                "compression": "zstd",
            },
        },
    }
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def test_run_offline_research_writes_expected_artifacts(tmp_path: Path) -> None:
    """Orchestration should emit core tables and metric summary files."""
    raw_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "outputs"
    _write_bundle_for_pipeline(raw_dir)

    run_offline_research(
        raw_dir=raw_dir,
        output_dir=output_dir,
        symbol="SPY",
    )

    expected_artifacts = [
        output_dir / "tables" / "feature_target_panel.csv",
        output_dir / "tables" / "evaluation_metrics.csv",
        output_dir / "tables" / "ridge_coefficients.csv",
        output_dir / "tables" / "predictions.csv",
        output_dir / "figures" / "prediction_scatter.png",
        output_dir / "summary.md",
    ]

    for path in expected_artifacts:
        assert path.exists(), f"Missing expected artifact: {path}"

    predictions = pd.read_csv(output_dir / "tables" / "predictions.csv")
    assert predictions["split"].value_counts()["purged"] == 10

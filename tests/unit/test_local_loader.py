"""Tests for offline raw-data loading and validation.

The loader is a hard system boundary for this project. These tests verify that
the boundary is explicit, deterministic, and independent from ClickHouse.
"""

from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from options_rv.io.local_loader import RawDataValidationError
from options_rv.io.local_loader import load_raw_data


REQUIRED_OPTIONS_COLUMNS: list[str] = [
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
]

REQUIRED_UNDERLYING_COLUMNS: list[str] = ["symbol", "trade_date", "close"]


def _write_valid_raw_bundle(raw_dir: Path) -> None:
    """Write a small but valid raw-data bundle for loader tests.

    Parameters
    ----------
    raw_dir
        Directory where raw files are written.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    options_quotes = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "trade_date": ["2025-01-02", "2025-01-03"],
            "expiry_date": ["2025-02-01", "2025-02-01"],
            "option_type": ["c", "p"],
            "strike_price": [600.0, 600.0],
            "bid": [3.0, 4.0],
            "ask": [3.2, 4.2],
            "bid_iv": [0.20, 0.21],
            "ask_iv": [0.21, 0.22],
            "open_interest": [1000, 1100],
            "volume": [500, 550],
        }
    )
    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "trade_date": ["2025-01-02", "2025-01-03"],
            "close": [600.0, 602.0],
        }
    )

    options_path = raw_dir / "options_quotes.parquet"
    underlying_path = raw_dir / "underlying_daily.parquet"
    options_quotes.to_parquet(options_path, index=False)
    underlying_daily.to_parquet(underlying_path, index=False)

    manifest = {
        "dataset_version": 1,
        "symbols": ["SPY"],
        "date_coverage": {"start": "2025-01-02", "end": "2025-01-03"},
        "files": {
            "options_quotes.parquet": {
                "rows": int(len(options_quotes)),
                "columns": REQUIRED_OPTIONS_COLUMNS,
                "compression": "zstd",
            },
            "underlying_daily.parquet": {
                "rows": int(len(underlying_daily)),
                "columns": REQUIRED_UNDERLYING_COLUMNS,
                "compression": "zstd",
            },
        },
    }
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def test_load_raw_data_success(tmp_path: Path) -> None:
    """Loader returns parsed data when files and metadata are valid."""
    raw_dir = tmp_path / "data" / "raw"
    _write_valid_raw_bundle(raw_dir=raw_dir)

    raw_data_bundle = load_raw_data(
        raw_dir=raw_dir,
        required_symbol="SPY",
        minimum_trading_days=2,
    )

    assert not raw_data_bundle.options_quotes.empty
    assert not raw_data_bundle.underlying_daily.empty
    assert raw_data_bundle.options_quotes["symbol"].eq("SPY").all()
    assert raw_data_bundle.underlying_daily["symbol"].eq("SPY").all()


def test_load_raw_data_raises_for_missing_file(tmp_path: Path) -> None:
    """Loader names missing file when raw input is incomplete."""
    raw_dir = tmp_path / "data" / "raw"
    _write_valid_raw_bundle(raw_dir=raw_dir)
    (raw_dir / "underlying_daily.parquet").unlink()

    with pytest.raises(RawDataValidationError, match="underlying_daily.parquet"):
        load_raw_data(raw_dir=raw_dir, required_symbol="SPY", minimum_trading_days=2)


def test_load_raw_data_raises_for_missing_column(tmp_path: Path) -> None:
    """Loader names missing columns when schema contract is violated."""
    raw_dir = tmp_path / "data" / "raw"
    _write_valid_raw_bundle(raw_dir=raw_dir)

    broken_underlying = pd.read_parquet(raw_dir / "underlying_daily.parquet").drop(
        columns=["close"]
    )
    broken_underlying.to_parquet(raw_dir / "underlying_daily.parquet", index=False)

    with pytest.raises(RawDataValidationError, match="close"):
        load_raw_data(raw_dir=raw_dir, required_symbol="SPY", minimum_trading_days=2)


def test_load_raw_data_raises_for_manifest_date_mismatch(tmp_path: Path) -> None:
    """Loader catches inconsistent date coverage between files and manifest."""
    raw_dir = tmp_path / "data" / "raw"
    _write_valid_raw_bundle(raw_dir=raw_dir)

    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["date_coverage"]["start"] = "2024-01-01"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(RawDataValidationError, match="date_coverage"):
        load_raw_data(raw_dir=raw_dir, required_symbol="SPY", minimum_trading_days=2)


def test_local_loader_module_does_not_import_clickhouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local loader remains independent from ClickHouse client imports."""
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("clickhouse"):
            raise AssertionError("local_loader should not import ClickHouse modules")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import options_rv.io.local_loader as local_loader

    importlib.reload(local_loader)

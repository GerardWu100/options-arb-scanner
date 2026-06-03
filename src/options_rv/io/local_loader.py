"""Offline raw-data loader and contract validator.

This module is the runtime entry point for local data access. It enforces a
strict contract for committed raw files under ``data/raw`` so the rest of the
pipeline can assume clean, stable inputs.

Notes
-----
- Runtime path is intentionally local-file-only.
- This module does not import any ClickHouse client dependency.
- Validation errors are explicit and user-facing to improve debuggability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd


REQUIRED_OPTIONS_COLUMNS: tuple[str, ...] = (
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
)

REQUIRED_UNDERLYING_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "close",
)


class RawDataValidationError(ValueError):
    """Raised when offline raw inputs fail schema or coverage checks."""


@dataclass(frozen=True)
class RawDataBundle:
    """Container for validated offline input datasets.

    Parameters
    ----------
    options_quotes
        Option quote table with one row per quote observation.
    underlying_daily
        Underlying close table with one row per symbol-date close.
    manifest
        Parsed JSON metadata that documents file-level contract information.
    """

    options_quotes: pd.DataFrame
    underlying_daily: pd.DataFrame
    manifest: dict[str, object]


def load_raw_data(
    raw_dir: Path,
    required_symbol: str = "SPY",
    minimum_trading_days: int = 200,
) -> RawDataBundle:
    """Load local raw files and enforce the project data contract.

    Parameters
    ----------
    raw_dir
        Directory containing ``options_quotes.parquet``,
        ``underlying_daily.parquet``, and ``manifest.json``.
    required_symbol
        Required symbol that must be present in both options and underlying
        files. Default is ``"SPY"``.
    minimum_trading_days
        Minimum required count of unique trading dates for the required symbol.

    Returns
    -------
    RawDataBundle
        Validated datasets and manifest metadata.

    Raises
    ------
    RawDataValidationError
        If files are missing, schemas are invalid, coverage is insufficient, or
        manifest metadata is inconsistent with data files.
    """
    options_path = raw_dir / "options_quotes.parquet"
    underlying_path = raw_dir / "underlying_daily.parquet"
    manifest_path = raw_dir / "manifest.json"

    _require_file(path=options_path)
    _require_file(path=underlying_path)
    _require_file(path=manifest_path)

    options_quotes = pd.read_parquet(options_path)
    underlying_daily = pd.read_parquet(underlying_path)
    manifest = _read_manifest(path=manifest_path)

    _require_columns(
        frame=options_quotes,
        required_columns=REQUIRED_OPTIONS_COLUMNS,
        dataset_name="options_quotes.parquet",
    )
    _require_columns(
        frame=underlying_daily,
        required_columns=REQUIRED_UNDERLYING_COLUMNS,
        dataset_name="underlying_daily.parquet",
    )

    options_quotes = _normalize_date_columns(
        frame=options_quotes.copy(),
        date_columns=("trade_date", "expiry_date"),
    )
    underlying_daily = _normalize_date_columns(
        frame=underlying_daily.copy(),
        date_columns=("trade_date",),
    )

    options_quotes = options_quotes.loc[
        options_quotes["symbol"] == required_symbol
    ].copy()
    underlying_daily = underlying_daily.loc[
        underlying_daily["symbol"] == required_symbol
    ].copy()

    if options_quotes.empty:
        raise RawDataValidationError(
            f"options_quotes.parquet does not contain required symbol: {required_symbol}"
        )
    if underlying_daily.empty:
        raise RawDataValidationError(
            f"underlying_daily.parquet does not contain required symbol: {required_symbol}"
        )

    options_dates = pd.Index(sorted(options_quotes["trade_date"].dropna().unique()))
    underlying_dates = pd.Index(
        sorted(underlying_daily["trade_date"].dropna().unique())
    )

    if len(underlying_dates) < minimum_trading_days:
        raise RawDataValidationError(
            "Insufficient underlying date coverage for symbol "
            f"{required_symbol}: {len(underlying_dates)} < {minimum_trading_days}"
        )
    if len(options_dates) < minimum_trading_days:
        raise RawDataValidationError(
            "Insufficient options date coverage for symbol "
            f"{required_symbol}: {len(options_dates)} < {minimum_trading_days}"
        )

    missing_option_dates = underlying_dates.difference(options_dates)
    if len(missing_option_dates) > 0:
        raise RawDataValidationError(
            "Option quotes are missing trading dates present in underlying data, "
            f"first missing date: {missing_option_dates[0].date()}"
        )

    _validate_manifest(
        manifest=manifest,
        options_quotes=options_quotes,
        underlying_daily=underlying_daily,
        required_symbol=required_symbol,
    )

    options_quotes = options_quotes.sort_values(
        ["symbol", "trade_date", "expiry_date", "strike_price"]
    )
    underlying_daily = underlying_daily.sort_values(["symbol", "trade_date"])

    return RawDataBundle(
        options_quotes=options_quotes.reset_index(drop=True),
        underlying_daily=underlying_daily.reset_index(drop=True),
        manifest=manifest,
    )


def _normalize_date_columns(
    frame: pd.DataFrame, date_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Normalize calendar columns to midnight timestamps for stable joins."""
    normalized = frame.copy()
    for column in date_columns:
        normalized[column] = pd.to_datetime(normalized[column]).dt.normalize()
    return normalized


def _require_file(path: Path) -> None:
    """Raise explicit validation error when a required input file is absent."""
    if not path.exists():
        raise RawDataValidationError(f"Missing required raw file: {path.name}")


def _read_manifest(path: Path) -> dict[str, object]:
    """Read and parse raw-data manifest JSON with clear error reporting."""
    try:
        manifest_text = path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise RawDataValidationError(
            f"Invalid JSON in manifest file: {path.name}"
        ) from error

    if not isinstance(manifest, dict):
        raise RawDataValidationError("Manifest JSON root must be an object")
    return manifest


def _require_columns(
    frame: pd.DataFrame, required_columns: tuple[str, ...], dataset_name: str
) -> None:
    """Validate that a dataframe includes all required contract columns."""
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        raise RawDataValidationError(
            f"{dataset_name} is missing required columns: {missing_columns_text}"
        )


def _validate_manifest(
    manifest: dict[str, object],
    options_quotes: pd.DataFrame,
    underlying_daily: pd.DataFrame,
    required_symbol: str,
) -> None:
    """Verify manifest metadata matches file contents for reproducibility.

    The manifest is intentionally strict so a reviewer can audit raw data at a
    glance without opening the Parquet files.
    """
    required_keys = ["dataset_version", "symbols", "date_coverage", "files"]
    missing_manifest_keys = [key for key in required_keys if key not in manifest]
    if missing_manifest_keys:
        raise RawDataValidationError(
            f"manifest.json missing required keys: {', '.join(missing_manifest_keys)}"
        )

    manifest_symbols = manifest["symbols"]
    if not isinstance(manifest_symbols, list):
        raise RawDataValidationError("manifest.json field 'symbols' must be a list")
    if required_symbol not in manifest_symbols:
        raise RawDataValidationError(
            f"manifest.json symbols do not include required symbol: {required_symbol}"
        )

    files_object = manifest["files"]
    if not isinstance(files_object, dict):
        raise RawDataValidationError("manifest.json field 'files' must be an object")

    for file_name, frame, required_columns in (
        ("options_quotes.parquet", options_quotes, REQUIRED_OPTIONS_COLUMNS),
        ("underlying_daily.parquet", underlying_daily, REQUIRED_UNDERLYING_COLUMNS),
    ):
        _validate_manifest_file_entry(
            files_object=files_object,
            file_name=file_name,
            frame=frame,
            required_columns=required_columns,
        )

    manifest_coverage = manifest["date_coverage"]
    if not isinstance(manifest_coverage, dict):
        raise RawDataValidationError(
            "manifest.json field 'date_coverage' must be an object"
        )

    expected_start = str(underlying_daily["trade_date"].min().date())
    expected_end = str(underlying_daily["trade_date"].max().date())
    manifest_start = str(manifest_coverage.get("start", ""))
    manifest_end = str(manifest_coverage.get("end", ""))

    if manifest_start != expected_start or manifest_end != expected_end:
        raise RawDataValidationError(
            "manifest.json date_coverage does not match underlying_daily.parquet "
            f"(expected start={expected_start}, end={expected_end})"
        )


def _validate_manifest_file_entry(
    files_object: dict[str, object],
    file_name: str,
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    """Validate one manifest file entry against observed dataframe properties."""
    if file_name not in files_object:
        raise RawDataValidationError(
            f"manifest.json files object missing key: {file_name}"
        )

    file_entry = files_object[file_name]
    if not isinstance(file_entry, dict):
        raise RawDataValidationError(
            f"manifest.json files['{file_name}'] must be an object"
        )

    manifest_rows = file_entry.get("rows")
    if int(manifest_rows) != int(len(frame)):
        raise RawDataValidationError(
            f"manifest.json row count mismatch for {file_name}: "
            f"manifest={manifest_rows}, observed={len(frame)}"
        )

    manifest_columns = file_entry.get("columns")
    if not isinstance(manifest_columns, list):
        raise RawDataValidationError(
            f"manifest.json columns must be a list for {file_name}"
        )

    missing_required_columns = [
        column for column in required_columns if column not in manifest_columns
    ]
    if missing_required_columns:
        raise RawDataValidationError(
            f"manifest.json columns for {file_name} missing required columns: "
            f"{', '.join(missing_required_columns)}"
        )

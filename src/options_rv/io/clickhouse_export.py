"""One-time ClickHouse export helper for raw offline cache refresh.

This module is intentionally excluded from normal runtime execution paths.
Use it only when refreshing committed raw files under ``data/raw``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import clickhouse_connect
import pandas as pd


@dataclass(frozen=True)
class ClickHouseConnectionConfig:
    """Connection settings for one-time raw export command.

    Parameters
    ----------
    host
        ClickHouse host name.
    port
        ClickHouse port number.
    username
        ClickHouse username.
    password
        ClickHouse password.
    secure
        Whether to use TLS.
    verify
        Whether to verify TLS certificates.
    """

    host: str
    port: int
    username: str
    password: str
    secure: bool = False
    verify: bool = False


def export_raw_data_from_clickhouse(
    connection_config: ClickHouseConnectionConfig,
    output_raw_dir: Path,
    symbol: str = "SPY",
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
) -> dict[str, object]:
    """Export filtered options and underlying data to local Parquet files.

    Parameters
    ----------
    connection_config
        ClickHouse connection configuration.
    output_raw_dir
        Output directory that will receive Parquet files and manifest.
    symbol
        Target symbol for export.
    start_date
        Inclusive lower trade-date bound.
    end_date
        Inclusive upper trade-date bound.

    Returns
    -------
    dict[str, object]
        Manifest object written to ``manifest.json``.
    """
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    options_path = output_raw_dir / "options_quotes.parquet"
    underlying_path = output_raw_dir / "underlying_daily.parquet"
    manifest_path = output_raw_dir / "manifest.json"

    client = clickhouse_connect.get_client(
        host=connection_config.host,
        port=connection_config.port,
        username=connection_config.username,
        password=connection_config.password,
        secure=connection_config.secure,
        verify=connection_config.verify,
    )

    options_query = (
        "SELECT "
        "symbol, trade_date, expiry_date, option_type, strike_price, "
        "bid, ask, bid_iv, ask_iv, open_interest, volume "
        "FROM firstrate.options "
        "WHERE symbol = {symbol:String} "
        "AND trade_date BETWEEN {start_date:Date} AND {end_date:Date} "
        "AND expiry_date > trade_date "
        "AND dateDiff('day', trade_date, expiry_date) BETWEEN 7 AND 90"
    )
    underlying_query = (
        "SELECT "
        "symbol, trade_date, close "
        "FROM firstrate.underlying_daily "
        "WHERE symbol = {symbol:String} "
        "AND trade_date BETWEEN {start_date:Date} AND {end_date:Date}"
    )

    query_parameters = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
    }

    options_result = client.query_df(options_query, parameters=query_parameters)
    underlying_result = client.query_df(underlying_query, parameters=query_parameters)

    _write_parquet_bundle(
        output_raw_dir=output_raw_dir,
        options=options_result,
        underlying=underlying_result,
    )

    manifest = _build_manifest(
        source={
            "system": "clickhouse",
            "options_table": "firstrate.options",
            "underlying_table": "firstrate.underlying_daily",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "options_filter": "7-90 calendar days to expiry, end-of-day rows",
        },
        symbol=symbol,
        options=options_result,
        underlying=underlying_result,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_raw_data_from_csv(
    options_csv_path: Path,
    underlying_csv_path: Path,
    output_raw_dir: Path,
    symbol: str = "SPY",
) -> dict[str, object]:
    """Build raw Parquet files from local CSV snapshots as fallback helper.

    This helper keeps notebook and runtime paths offline while allowing quick
    local assembly of a synthetic or pre-extracted raw cache.
    """
    options = pd.read_csv(options_csv_path)
    underlying = pd.read_csv(underlying_csv_path)

    options = options.loc[options["symbol"] == symbol].copy()
    underlying = underlying.loc[underlying["symbol"] == symbol].copy()

    options["trade_date"] = pd.to_datetime(options["trade_date"])
    options["expiry_date"] = pd.to_datetime(options["expiry_date"])
    underlying["trade_date"] = pd.to_datetime(underlying["trade_date"])

    options = options.sort_values(
        ["symbol", "trade_date", "expiry_date", "strike_price"]
    )
    underlying = underlying.sort_values(["symbol", "trade_date"])

    output_raw_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_bundle(
        output_raw_dir=output_raw_dir,
        options=options,
        underlying=underlying,
    )

    manifest = _build_manifest(
        source={
            "system": "local_csv",
            "description": "offline fallback assembly from CSV snapshots",
        },
        symbol=symbol,
        options=options,
        underlying=underlying,
    )
    (output_raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _write_parquet_bundle(
    output_raw_dir: Path,
    options: pd.DataFrame,
    underlying: pd.DataFrame,
) -> None:
    """Write the standard raw Parquet pair used by the offline loader."""
    options.to_parquet(
        output_raw_dir / "options_quotes.parquet", index=False, compression="zstd"
    )
    underlying.to_parquet(
        output_raw_dir / "underlying_daily.parquet", index=False, compression="zstd"
    )


def _build_manifest(
    source: dict[str, object],
    symbol: str,
    options: pd.DataFrame,
    underlying: pd.DataFrame,
) -> dict[str, object]:
    """Build manifest metadata shared by ClickHouse and CSV export paths."""
    return {
        "dataset_version": 1,
        "source": source,
        "symbols": [symbol],
        "date_coverage": {
            "start": str(underlying["trade_date"].min().date()),
            "end": str(underlying["trade_date"].max().date()),
        },
        "files": {
            "options_quotes.parquet": {
                "rows": int(len(options)),
                "columns": list(options.columns),
                "compression": "zstd",
            },
            "underlying_daily.parquet": {
                "rows": int(len(underlying)),
                "columns": list(underlying.columns),
                "compression": "zstd",
            },
        },
    }

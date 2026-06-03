"""Thin CLI wrapper for one-time ClickHouse raw-data refresh.

Reusable export logic lives in ``options_rv.io.clickhouse_export``. This script
only parses arguments and calls that module. Normal offline runs do not need it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from options_rv.io.clickhouse_export import (
    ClickHouseConnectionConfig,
    export_raw_data_from_clickhouse,
)

# Project root: scripts/refresh_raw_from_clickhouse.py -> parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RAW_DIR = _PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """Export SPY options and underlying Parquet files from ClickHouse."""
    parser = argparse.ArgumentParser(
        description="One-time export of raw Parquet files from ClickHouse",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_RAW_DIR,
        help="Directory for options_quotes.parquet, underlying_daily.parquet, manifest.json",
    )
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--host", type=str, default=os.environ.get("CLICKHOUSE_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CLICKHOUSE_PORT", "8123")))
    parser.add_argument(
        "--username",
        type=str,
        default=os.environ.get("CLICKHOUSE_USER", "default"),
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.environ.get("CLICKHOUSE_PASSWORD", ""),
    )
    args = parser.parse_args()

    connection_config = ClickHouseConnectionConfig(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
    )
    manifest = export_raw_data_from_clickhouse(
        connection_config=connection_config,
        output_raw_dir=args.output_dir,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Wrote raw export to {args.output_dir.resolve()}")
    print(f"manifest symbols={manifest.get('symbols')}")
    print(f"manifest coverage={manifest.get('date_coverage')}")


if __name__ == "__main__":
    main()

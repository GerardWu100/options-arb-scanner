# Offline Raw Data Contract

## Scope

This contract defines the raw files required for normal offline runtime.

Required paths under `data/raw/`:

- `options_quotes.parquet`
- `underlying_daily.parquet`
- `manifest.json`

## Dataset: `options_quotes.parquet`

Required columns:

- `symbol` (string)
- `trade_date` (date)
- `expiry_date` (date)
- `option_type` (string, expected values `c` or `p`)
- `strike_price` (float)
- `bid` (float)
- `ask` (float)
- `bid_iv` (float)
- `ask_iv` (float)
- `open_interest` (integer)
- `volume` (integer)

Expected filtering assumptions:

- end-of-day quote snapshots only
- positive days-to-expiry
- bounded expiry horizon (recommended: 7 to 90 calendar days)
- single-symbol default (`SPY`) for compact payload

## Dataset: `underlying_daily.parquet`

Required columns:

- `symbol` (string)
- `trade_date` (date)
- `close` (float)

## Dataset: `manifest.json`

Required top-level keys:

- `dataset_version` (integer)
- `symbols` (list)
- `date_coverage` (object with `start`, `end`)
- `files` (object)

Required `files` entries:

- `options_quotes.parquet`
- `underlying_daily.parquet`

Each `files` entry must include:

- `rows`
- `columns`
- `compression`

## Validation Requirements

The local loader enforces:

- required files exist
- required columns exist
- required symbol is present in both files
- minimum date coverage is satisfied
- options date coverage includes underlying date coverage
- manifest row counts and date coverage match observed files

## Portability Guarantee

If this contract is satisfied, runtime code in `src/options_rv/` executes without
ClickHouse and without network access.

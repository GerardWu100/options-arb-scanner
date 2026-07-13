# Guide: `data/raw/`

## Purpose

`data/raw/` contains all committed raw inputs required for offline runtime.

This folder is the portability boundary: a fresh clone with these files present
can run tests, CLI, and notebook without any database connection.

## Required Files

- `options_quotes.parquet`: end-of-day options quotes.
- `underlying_daily.parquet`: underlying close series.
- `manifest.json`: self-describing metadata for schema, row counts, symbols, and date coverage.

## Contract Rules

- Keep raw files raw. Do not place engineered features or model outputs here.
- Keep schema stable unless docs and loader validation are updated together.
- Prefer compressed Parquet with `zstd`.
- Keep symbol scope narrow for payload control; default is only `SPY`.

The current manifest records a deterministic synthetic generator as the source.
These files test portability and execution; they are not historical SPY market
observations.

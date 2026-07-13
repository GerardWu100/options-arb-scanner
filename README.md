# Offline Options to Realized Variance Research

This repository is an offline-first quantitative research project that studies one narrow question:

- Can a compact and explainable set of end-of-day option-surface features improve forecasts of forward realized variance for SPY?

The project is intentionally small and interview-defensible. It focuses on one symbol, one horizon, a few interpretable features, two simple baselines, and one linear model.

## Research Framing

Target definition:

- Predict 5-trading-day forward annualized realized variance for SPY.

Symbols and conventions:

- `S_t`: SPY close on trade date `t`.
- `r_t = ln(S_t / S_{t-1})`: daily log return from date `t-1` to `t`.
- `RV_{t,t+5} = sum_{i=1}^{5} r_{t+i}^2`: forward 5-day realized variance in daily units.
- `annualized_RV_{t,t+5} = (252 / 5) * RV_{t,t+5}`: annualized forward variance.

The implementation first computes a trailing five-return sum and then shifts
the completed sum back by five rows. This aligns the label at `t` with returns
ending on `t+1` through `t+5`; shifting returns before applying a trailing
window would be an off-by-one error.

Model comparison:

- Persistence baseline from trailing realized variance.
- ATM implied-volatility baseline.
- Ridge regression model on compact options features.

The chronological 60%/20%/20% split purges five rows from the end of the train
and validation segments. Adjacent five-day labels overlap, so this gap prevents
training and evaluation labels from sharing future returns across a boundary.

## Scope: Forecasting, Not Arbitrage

The repository name is historical. The current package forecasts realized
variance; it does not scan for relative-value trades or prove arbitrage.
Quote-quality filters reject crossed prices, invalid implied-volatility bounds,
nonpositive strikes or mids, and negative activity fields. They do not test
cross-strike monotonicity, butterfly convexity, put-call parity, executable
transaction costs, or borrow constraints.

## Offline Data Contract

All committed raw data lives under `data/raw/`:

- `data/raw/options_quotes.parquet`
- `data/raw/underlying_daily.parquet`
- `data/raw/manifest.json`

Runtime code only reads those local files. If `data/raw/` is present, normal project usage does not need ClickHouse, credentials, or network access.

## One-Time Export vs Normal Usage

- Normal usage: local offline files only.
- One-time refresh: optional `scripts/refresh_raw_from_clickhouse.py` (package helper in `src/options_rv/io/clickhouse_export.py`) writes the same local Parquet and manifest files.

The notebook and CLI pipeline never query ClickHouse directly.

## Project Layout

```text
.
├── config.toml
├── data/
│   └── raw/
├── docs/
│   └── reference/
├── logs/
├── notebooks/
│   └── options_rv_pipeline_demo.ipynb
├── scripts/
│   └── refresh_raw_from_clickhouse.py
├── src/
│   └── options_rv/
├── tests/
│   └── unit/
├── GUIDE_ROOT.md
├── GUIDE_OVERVIEW.md
└── pyproject.toml
```

## Setup

Use `uv`:

```bash
uv sync
```

## Run Offline Pipeline

```bash
uv run python -m options_rv.cli \
  --config config.toml \
  --output-dir outputs/demo_run
```

The pipeline writes only non-frontend artifacts such as CSV or Parquet tables and PNG figures.

## Run Tests

```bash
uv run pytest -q
```

## Execute Notebook Non-Interactively

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/options_rv_pipeline_demo.ipynb \
  --output /tmp/options_rv_pipeline_demo.executed.ipynb
```

## Why This Scope

This repository is designed for clear communication and reproducibility:

- one symbol (`SPY`)
- one horizon (5 trading days)
- explainable features and linear model
- strict chronological evaluation
- portable offline execution from committed Parquet files

The committed inputs are deterministic synthetic fixtures. Pipeline output is
software evidence, not an empirical result about listed SPY options.

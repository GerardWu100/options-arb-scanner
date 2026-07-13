# Offline Options RV Project Overview

## Project Tree

```text
.
├── config.toml
├── data/
│   └── raw/
│       ├── GUIDE_data_raw.md
│       ├── manifest.json
│       ├── options_quotes.parquet
│       └── underlying_daily.parquet
├── docs/
│   └── reference/
│       ├── data_contract.md
│       └── methodology.md
├── logs/
├── notebooks/
│   └── options_rv_pipeline_demo.ipynb
├── scripts/
│   ├── GUIDE_scripts.md
│   └── refresh_raw_from_clickhouse.py
├── src/
│   ├── GUIDE_src.md
│   └── options_rv/
│       ├── GUIDE_options_rv.md
│       ├── cli.py
│       ├── io/
│       ├── targets/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       └── pipeline/
├── tests/
│   └── unit/
├── GUIDE_ROOT.md
├── GUIDE_OVERVIEW.md
├── README.md
└── pyproject.toml
```

## Purpose

The project studies whether option-implied information helps forecast short-horizon realized variance.

Research target:

- 5-trading-day forward annualized realized variance for SPY.

Model set:

- persistence baseline
- ATM implied-vol baseline
- ridge regression with compact option-surface features

## Core Data Flow

```text
data/raw/options_quotes.parquet + data/raw/underlying_daily.parquet
  -> io/local_loader.py validates schema, symbols, coverage, manifest
  -> targets/realized_variance.py builds forward 5-day annualized RV target
  -> features/option_surface.py builds one row per symbol-date feature panel
  -> evaluation/splits.py defines purged chronological windows
  -> models/baselines.py produces baseline forecasts
  -> models/train.py fits ridge model and prediction tables
  -> evaluation/metrics.py computes RMSE, MAE, and optional QLIKE
  -> pipeline/offline_research.py writes outputs tables and figures
```

## Runtime Guarantees

- Runtime path is offline and local-file-only.
- ClickHouse refresh uses `scripts/refresh_raw_from_clickhouse.py` only.
- CLI and notebook use stable package functions rather than ad hoc inline transformations.

## Outputs

Pipeline outputs live under `outputs/` and include:

- tabular artifacts (`.csv` and optionally `.parquet`)
- evaluation tables for baselines and ridge model
- coefficient table for interpretation
- PNG figures in `outputs/figures/`

No HTML, dashboard, or frontend assets are produced.

## Research Boundaries

The historical repository name mentions arbitrage, but the active package is a
variance-forecasting experiment. It filters individually invalid quotes without
testing a full option surface for static arbitrage. Five-day labels overlap, so
the pipeline removes five observations before the validation and test segments.
The committed synthetic inputs support reproducibility tests only; they cannot
support a claim about SPY forecastability or a tradeable variance risk premium.

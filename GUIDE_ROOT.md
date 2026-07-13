## Part 1: Conceptual Explanation

This repository is an offline-first quantitative research project focused on forecasting forward realized variance from end-of-day option data.

The root stays intentionally thin:

- implementation in `src/options_rv/`
- tests in `tests/unit/`
- runtime config in `config.toml`
- committed raw inputs in `data/raw/`
- thin one-time scripts in `scripts/`
- user and developer documentation in `README.md`, `GUIDE_OVERVIEW.md`, and `docs/reference/`

High-level flow:

```text
Committed Parquet raw data in data/raw/
  -> src/options_rv/io/local_loader.py
  -> src/options_rv/targets/realized_variance.py
  -> src/options_rv/features/option_surface.py
  -> src/options_rv/models/{baselines,train}.py
  -> src/options_rv/evaluation/{splits,metrics}.py
  -> src/options_rv/pipeline/offline_research.py
  -> outputs/ tables and figures
```

ClickHouse is isolated to a one-time export script and is not used during normal runtime.

## Part 2: Code Reference

- `README.md`: project purpose, research framing, commands, and offline portability statement.
- `config.toml`: runtime symbol, horizon, raw directory, and annualization settings.
- `pyproject.toml`: package metadata, dependencies, script entry points, and pytest settings.
- `.gitignore`: ignores generated artifacts and temporary raw-export folders while keeping `data/raw` tracked.
- `data/raw/`: committed raw options and underlying files plus `manifest.json`. See `data/raw/GUIDE_data_raw.md`.
- `src/options_rv/`: runtime package. See `src/GUIDE_src.md` and `src/options_rv/GUIDE_options_rv.md`.
- `scripts/`: thin CLI wrappers. See `scripts/GUIDE_scripts.md`.
- `tests/unit/`: unit and integration tests for loaders, targets, features, models, and pipeline.
- `docs/reference/data_contract.md`: raw-data schema and manifest contract.
- `docs/reference/methodology.md`: formulas, conventions, split design, and metric definitions.
- `notebooks/options_rv_pipeline_demo.ipynb`: teaching notebook that runs the same package functions used by CLI.
- `options-arb-scanner.md`: user-authored note file. Do not modify.

Where to start:

- Read `README.md` first.
- Then read `GUIDE_OVERVIEW.md`.
- Then read `src/GUIDE_src.md` before diving into modules.

## Part 3: Short Journal

- 2026-04-19: Repository pivoted from static arbitrage scanning to offline options-to-realized-variance research with local Parquet-first runtime.
- 2026-05-20: Standardized layout (`config.toml` at root, `tests/unit/`, `scripts/`).
- 2026-07-13: Corrected the forward-return alignment, purged overlapping labels
  at evaluation boundaries, and clarified that the active project forecasts
  variance rather than scanning arbitrage.

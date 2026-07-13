# Guide: `src/options_rv/`

## Part 1: Conceptual Explanation

`options_rv` implements offline research for forecasting 5-day forward
annualized realized variance from end-of-day option-surface features.

Package layout by responsibility:

- `io/`: local raw loader and ClickHouse export helpers.
- `targets/`: forward realized-variance target construction.
- `features/`: deterministic option-surface feature engineering.
- `models/`: persistence and ATM-IV baselines plus ridge training.
- `evaluation/`: chronological splits and forecast metrics.
- `pipeline/`: end-to-end orchestration for CLI and notebook use.

ClickHouse is isolated to `io/clickhouse_export.py` and is only invoked from
`scripts/refresh_raw_from_clickhouse.py` when refreshing `data/raw/`.

The target attached to date $t$ sums squared log returns ending on $t+1$
through $t+h$, where $h$ is the forecast horizon. The split builder then marks
the last $h$ rows before validation and test as `purged`, preventing overlapping
labels from sharing returns across an evaluation boundary.

Option feature construction filters invalid individual quotes and averages call
and put mid implied volatilities when both share the closest at-the-money strike.
It does not enforce cross-strike or cross-maturity no-arbitrage conditions.

## Part 2: Code Reference

- `cli.py`: reads repo-root `config.toml` and runs `run_offline_research`.
- `pipeline/offline_research.py`: writes tables and figures under `outputs/`.
- `io/local_loader.py`: validates and loads committed Parquet inputs.
- `targets/realized_variance.py`: builds forward RV target and trailing variance.
- `features/option_surface.py`: one row per symbol-date feature panel.
- `models/baselines.py`, `models/train.py`: baselines and ridge model.
- `evaluation/splits.py`, `evaluation/metrics.py`: split masks and RMSE/MAE/QLIKE.

## Part 3: Short Journal

- 2026-04-19: Initial offline options-to-RV research package.
- 2026-05-20: Runtime config moved to repo-root `config.toml`.
- 2026-07-13: Fixed forward-label alignment and added horizon-length split
  purging after a technical audit exposed an off-by-one target bug.

# options-arb-scanner

Offline research pipeline that forecasts SPY's forward realized variance from
end-of-day option-surface features. The repo name is historical: the code no
longer scans for arbitrage. It compares a linear model against two baselines
under a strict chronological evaluation.

## What it does

Given end-of-day SPY option quotes and underlying closes, the pipeline:

- builds the target: 5-trading-day forward annualized realized variance,
  `RV_{t,t+5} = sum_{i=1}^{5} r_{t+i}^2`, annualized by `252 / 5`
- builds six compact option-surface features (ATM 30-day implied vol, term
  slope, downside skew, average bid-ask spread ratio, total open interest,
  trailing 20-day realized variance)
- fits three models: a persistence baseline (trailing realized variance), an
  ATM implied-vol baseline (`atm_iv_30d^2`), and a ridge regression on the
  standardized feature set
- evaluates them with RMSE, MAE, and QLIKE on a purged chronological
  train/validation/test split (60/20/20, with the last 5 rows before each
  split boundary dropped to avoid overlapping-label leakage)

Quote-quality filters reject crossed prices, invalid implied-vol bounds, and
nonpositive strikes or mids. They do not test a full option surface for static
arbitrage (no butterfly, calendar, or put-call-parity checks). Full formulas
and leakage controls are in `docs/reference/methodology.md`.

The committed data under `data/raw/` is deterministic synthetic data, not a
real SPY options snapshot. Pipeline output is a software-correctness
demonstration, not an empirical claim about SPY.

## Requirements

- Python >=3.13
- No external service is needed for normal use: the pipeline reads only the
  Parquet files committed under `data/raw/`.
- ClickHouse is needed only for the optional one-time data refresh
  (`scripts/refresh_raw_from_clickhouse.py`), which reads these environment
  variables: `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`,
  `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_SECURE`, `CLICKHOUSE_VERIFY`.

## Setup

```bash
uv sync
```

## Usage

Run the offline pipeline:

```bash
uv run python -m options_rv.cli --config config.toml --output-dir outputs/demo_run
```

or, after `uv sync`, the installed entry point:

```bash
uv run options-rv --config config.toml --output-dir outputs/demo_run
```

Run tests:

```bash
uv run pytest -q
```

Execute the teaching notebook non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/options_rv_pipeline_demo.ipynb \
  --output /tmp/options_rv_pipeline_demo.executed.ipynb
```

Refresh raw data from ClickHouse (one-time, requires credentials above; not
used by normal runs):

```bash
set -a && source .env && set +a
uv run python scripts/refresh_raw_from_clickhouse.py --output-dir data/raw
```

## Configuration

Runtime settings live in `config.toml`:

- `raw_dir`: directory with the committed Parquet inputs (default `data/raw`)
- `symbol`: underlying symbol (only `SPY` is supported)
- `horizon_days`: forecast horizon in trading days (default 5)
- `annualization_factor`: annualization factor for daily variance (default 252)

## Layout

```text
config.toml           runtime settings read by the CLI
data/raw/              committed synthetic Parquet inputs and manifest
docs/reference/         data contract and full methodology notes
notebooks/              teaching notebook that calls the same package functions as the CLI
scripts/                one-time ClickHouse refresh wrapper
src/options_rv/         package: io, targets, features, models, evaluation, pipeline
tests/unit/              unit and integration tests
```

See `GUIDE_ROOT.md` and `GUIDE_OVERVIEW.md` for the full module map.

## Output

The pipeline writes to the `--output-dir` you pass (for example
`outputs/demo_run/`): CSV/Parquet evaluation tables for the baselines and
ridge model, a ridge coefficient table, and PNG figures. No HTML, dashboard,
or frontend artifacts are produced.

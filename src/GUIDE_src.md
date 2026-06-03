# Guide: `src/`

## Part 1: Conceptual Explanation

The `src/` directory holds the installable Python package for this project. All
offline research logic lives under `options_rv/`, organized by responsibility
(loaders, targets, features, models, evaluation, pipeline orchestration).

Runtime configuration is **not** stored here. Project-wide settings live in the
repo-root `config.toml`. One-time ClickHouse refresh is exposed through a thin
script in `scripts/`, while reusable export code remains in the package.

Normal execution is offline-first: CLI and notebook read committed Parquet files
under `data/raw/` and write artifacts under `outputs/`.

## Part 2: Code Reference

- `options_rv/`: main package. See `options_rv/GUIDE_options_rv.md`.
- `options_rv/cli.py`: CLI entry (`options-rv` script) and TOML config loader.
- `options_rv/pipeline/offline_research.py`: end-to-end orchestration.
- `options_rv/io/clickhouse_export.py`: reusable ClickHouse export helpers (not
  used in normal offline runs).

Where to start:

- Read `options_rv/GUIDE_options_rv.md`, then `options_rv/pipeline/offline_research.py`.

## Part 3: Short Journal

- 2026-05-20: Aligned layout with standard `src/` package tree; moved `config.toml` to repo root.

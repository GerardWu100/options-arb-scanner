# Guide: `scripts/`

## Part 1: Conceptual Explanation

`scripts/` holds thin command-line wrappers. Reusable logic stays in
`src/options_rv/`. Scripts parse arguments, resolve project-relative paths, and
call package functions.

## Part 2: Code Reference

- `refresh_raw_from_clickhouse.py`: one-time ClickHouse export to `data/raw/`.
  Requires network credentials; not used by tests, CLI, or notebook.

Run example:

```bash
set -a && source .env && set +a
uv run python scripts/refresh_raw_from_clickhouse.py --output-dir data/raw
```

## Part 3: Short Journal

- 2026-05-20: Added thin ClickHouse refresh wrapper during layout refactor.

"""Command-line entry point for offline options RV research runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import tomllib

from options_rv.pipeline.offline_research import run_offline_research

# Project root: src/options_rv/cli.py -> parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class OfflineResearchConfig:
    """Runtime configuration for the offline research pipeline.

    Parameters
    ----------
    raw_dir
        Directory containing committed raw input files.
    symbol
        Underlying symbol to analyze.
    horizon_days
        Forecast horizon measured in trading days.
    annualization_factor
        Annualization factor for daily realized variance.
    """

    raw_dir: Path
    symbol: str
    horizon_days: int
    annualization_factor: int


def main() -> None:
    """Parse CLI arguments, load config, and run offline pipeline."""
    parser = argparse.ArgumentParser(
        prog="options-rv",
        description="Run offline options-to-realized-variance research pipeline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to TOML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where output artifacts are written.",
    )
    args = parser.parse_args()

    runtime_config = load_config(config_path=args.config)
    run_offline_research(
        raw_dir=runtime_config.raw_dir,
        output_dir=args.output_dir,
        symbol=runtime_config.symbol,
        horizon_days=runtime_config.horizon_days,
        annualization_factor=runtime_config.annualization_factor,
    )


def load_config(config_path: Path) -> OfflineResearchConfig:
    """Load runtime configuration from TOML file.

    Parameters
    ----------
    config_path
        Path to TOML configuration file.

    Returns
    -------
    OfflineResearchConfig
        Parsed configuration dataclass.

    Notes
    -----
    A relative ``raw_dir`` is interpreted relative to the directory holding the
    configuration file, not the current working directory. The shipped
    ``config.toml`` sits at the project root and stores ``raw_dir = "data/raw"``,
    so this keeps the installed ``options-rv`` command working from any folder.
    """
    resolved_config_path = config_path.resolve()
    config_object = tomllib.loads(resolved_config_path.read_text(encoding="utf-8"))

    runtime_object = config_object.get("runtime", {})
    if not isinstance(runtime_object, dict):
        raise ValueError("[runtime] section must be a TOML table")

    raw_dir_value = runtime_object.get("raw_dir", "data/raw")
    symbol_value = runtime_object.get("symbol", "SPY")
    horizon_value = runtime_object.get("horizon_days", 5)
    annualization_value = runtime_object.get("annualization_factor", 252)

    raw_dir_path = Path(str(raw_dir_value))
    if not raw_dir_path.is_absolute():
        raw_dir_path = resolved_config_path.parent / raw_dir_path

    return OfflineResearchConfig(
        raw_dir=raw_dir_path,
        symbol=str(symbol_value),
        horizon_days=int(horizon_value),
        annualization_factor=int(annualization_value),
    )


if __name__ == "__main__":
    main()

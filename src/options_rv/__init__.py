"""Top-level package for offline options-to-realized-variance research.

This package exposes a compact and explainable quantitative research pipeline.
The runtime path is intentionally offline-first: normal execution reads only local
Parquet files committed under ``data/raw``.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"

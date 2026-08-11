"""Chronological split construction for time-ordered evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_chronological_split_masks(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    purge_gap_rows: int = 0,
) -> pd.Series:
    """Create deterministic chronological train/validation/test labels.

    Parameters
    ----------
    frame
        Dataframe containing at least a ``trade_date`` column.
    train_fraction
        Fraction of earliest rows assigned to train split.
    validation_fraction
        Fraction of middle rows assigned to validation split.
    purge_gap_rows
        Number of rows removed from the end of train and validation so labels
        with overlapping forward horizons cannot cross a split boundary.

    Returns
    -------
    pd.Series
        Split label (`train`, `validation`, `test`, or `purged`) for each row
        in original row order.
    """
    if "trade_date" not in frame.columns:
        raise ValueError("frame must contain trade_date column")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    if purge_gap_rows < 0:
        raise ValueError("purge_gap_rows must be non-negative")

    # Sort by calendar date but return labels aligned to the caller's row order.
    sorted_indices = frame.sort_values("trade_date").index.to_numpy()
    total_rows = len(sorted_indices)
    if total_rows < 3:
        raise ValueError("Need at least 3 rows to create train/validation/test splits")

    train_end = max(1, int(np.floor(total_rows * train_fraction)))
    validation_end = max(
        train_end + 1,
        int(np.floor(total_rows * (train_fraction + validation_fraction))),
    )
    # Reserve at least one row for the held-out test segment.
    validation_end = min(validation_end, total_rows - 1)

    if purge_gap_rows >= train_end:
        raise ValueError("purge_gap_rows must leave at least one train row")
    validation_rows = validation_end - train_end
    if purge_gap_rows >= validation_rows:
        raise ValueError("purge_gap_rows must leave at least one validation row")

    labels = pd.Series(index=frame.index, dtype="object")
    labels.loc[sorted_indices[:train_end]] = "train"
    labels.loc[sorted_indices[train_end:validation_end]] = "validation"
    labels.loc[sorted_indices[validation_end:]] = "test"

    if purge_gap_rows > 0:
        train_purge_start = max(0, train_end - purge_gap_rows)
        validation_purge_start = max(train_end, validation_end - purge_gap_rows)
        labels.loc[sorted_indices[train_purge_start:train_end]] = "purged"
        labels.loc[sorted_indices[validation_purge_start:validation_end]] = "purged"
    return labels

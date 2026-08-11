"""Tests for chronological split construction and overlap purging."""

from __future__ import annotations

import pandas as pd
from options_rv.evaluation.splits import build_chronological_split_masks


def test_purge_removes_rows_before_each_forward_split_boundary() -> None:
    """Purging should exclude the requested rows before validation and test."""
    frame = pd.DataFrame(
        {"trade_date": pd.date_range("2025-01-01", periods=20, freq="B")}
    )

    labels = build_chronological_split_masks(
        frame=frame,
        train_fraction=0.5,
        validation_fraction=0.25,
        purge_gap_rows=2,
    )

    assert labels.tolist() == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "purged",
        "purged",
        "validation",
        "validation",
        "validation",
        "purged",
        "purged",
        "test",
        "test",
        "test",
        "test",
        "test",
    ]

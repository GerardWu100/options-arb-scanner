"""Tests for ridge-regression model training utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from options_rv.models.train import train_ridge_regression


def test_train_ridge_returns_predictions_and_coefficients() -> None:
    """Training helper returns fitted model outputs for evaluation and interpretation."""
    feature_frame = pd.DataFrame(
        {
            "symbol": ["SPY"] * 6,
            "trade_date": pd.date_range("2025-01-01", periods=6, freq="B"),
            "atm_iv_30d": [0.20, 0.21, 0.22, 0.23, 0.24, 0.25],
            "term_slope_60d_minus_30d": [0.01, 0.011, 0.012, 0.013, 0.014, 0.015],
            "downside_skew_30d": [0.02, 0.021, 0.023, 0.024, 0.025, 0.026],
            "avg_spread_ratio": [0.03, 0.031, 0.031, 0.032, 0.033, 0.034],
            "total_open_interest": [1000, 1100, 1200, 1300, 1400, 1500],
            "trailing_annualized_variance_20d": [
                0.04,
                0.041,
                0.042,
                0.043,
                0.044,
                0.045,
            ],
            "target_log_annualized_forward_variance_5d": [
                -3.2,
                -3.1,
                -3.0,
                -2.9,
                -2.8,
                -2.7,
            ],
        }
    )

    split_labels = pd.Series(["train", "train", "train", "validation", "test", "test"])

    result = train_ridge_regression(
        feature_target_frame=feature_frame,
        split_labels=split_labels,
        feature_columns=[
            "atm_iv_30d",
            "term_slope_60d_minus_30d",
            "downside_skew_30d",
            "avg_spread_ratio",
            "total_open_interest",
            "trailing_annualized_variance_20d",
        ],
        target_column="target_log_annualized_forward_variance_5d",
        ridge_alpha=1.0,
    )

    assert len(result.predictions) == len(feature_frame)
    assert set(result.coefficients["feature"]) == {
        "atm_iv_30d",
        "term_slope_60d_minus_30d",
        "downside_skew_30d",
        "avg_spread_ratio",
        "total_open_interest",
        "trailing_annualized_variance_20d",
    }
    assert np.isfinite(result.predictions["prediction"]).all()

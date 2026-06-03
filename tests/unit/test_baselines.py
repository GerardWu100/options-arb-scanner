"""Tests for simple forecast baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from options_rv.models.baselines import compute_atm_iv_baseline
from options_rv.models.baselines import compute_persistence_baseline


def test_persistence_baseline_uses_trailing_variance_level() -> None:
    """Persistence baseline should copy trailing variance feature."""
    frame = pd.DataFrame(
        {
            "trailing_annualized_variance_20d": [0.04, 0.05, 0.06],
        }
    )

    prediction = compute_persistence_baseline(frame)

    np.testing.assert_allclose(prediction.to_numpy(), np.array([0.04, 0.05, 0.06]))


def test_atm_iv_baseline_squares_iv_to_variance() -> None:
    """ATM implied-vol baseline should map volatility to variance by squaring."""
    frame = pd.DataFrame(
        {
            "atm_iv_30d": [0.2, 0.25],
        }
    )

    prediction = compute_atm_iv_baseline(frame)

    np.testing.assert_allclose(prediction.to_numpy(), np.array([0.04, 0.0625]))

"""Tests for forward realized-variance target construction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from options_rv.targets.realized_variance import build_forward_realized_variance_target


def test_forward_realized_variance_annualization_and_log_target() -> None:
    """Target uses forward 5-day squared log returns and annualization."""
    constant_daily_log_return = 0.01
    close_values = [100.0]
    for _ in range(11):
        close_values.append(close_values[-1] * math.exp(constant_daily_log_return))

    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY"] * len(close_values),
            "trade_date": pd.date_range(
                "2025-01-01", periods=len(close_values), freq="B"
            ),
            "close": close_values,
        }
    )

    target_frame = build_forward_realized_variance_target(
        underlying_daily=underlying_daily,
        horizon_days=5,
        annualization_factor=252,
        use_log_target=True,
    )

    expected_annualized_variance = (252.0 / 5.0) * 5.0 * (constant_daily_log_return**2)
    expected_log_target = math.log(expected_annualized_variance)

    np.testing.assert_allclose(
        target_frame.loc[target_frame.index[0], "annualized_forward_variance_5d"],
        expected_annualized_variance,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        target_frame.loc[
            target_frame.index[0], "target_log_annualized_forward_variance_5d"
        ],
        expected_log_target,
        rtol=1e-10,
    )


def test_forward_window_alignment_uses_t_plus_1_return() -> None:
    """Target at date t starts with return ending at t+1, not return ending at t."""
    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "trade_date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]
            ),
            "close": [100.0, 101.0, 103.0, 104.0],
        }
    )

    target_frame = build_forward_realized_variance_target(
        underlying_daily=underlying_daily,
        horizon_days=1,
        annualization_factor=252,
        use_log_target=False,
    )

    first_forward_return = math.log(101.0 / 100.0)
    expected_first_target = 252.0 * (first_forward_return**2)

    np.testing.assert_allclose(
        target_frame.loc[target_frame.index[0], "annualized_forward_variance_1d"],
        expected_first_target,
        rtol=1e-10,
    )


def test_forward_window_alignment_excludes_contemporaneous_return() -> None:
    """A non-constant example distinguishes the future window from a trailing one."""
    daily_log_returns = [0.01, 0.02, 0.03, 0.04]
    close_values = [100.0]
    for daily_log_return in daily_log_returns:
        close_values.append(close_values[-1] * math.exp(daily_log_return))

    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY"] * len(close_values),
            "trade_date": pd.date_range(
                "2025-01-01", periods=len(close_values), freq="B"
            ),
            "close": close_values,
        }
    )

    target_frame = build_forward_realized_variance_target(
        underlying_daily=underlying_daily,
        horizon_days=2,
        annualization_factor=252,
        use_log_target=False,
    )

    expected_at_first_date = (252.0 / 2.0) * (0.01**2 + 0.02**2)
    expected_at_second_date = (252.0 / 2.0) * (0.02**2 + 0.03**2)
    np.testing.assert_allclose(
        target_frame["annualized_forward_variance_2d"].iloc[:2],
        [expected_at_first_date, expected_at_second_date],
        rtol=1e-10,
    )


def test_rows_without_full_forward_window_are_removed() -> None:
    """Rows that cannot see the full horizon are excluded for leakage safety."""
    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY", "SPY", "SPY"],
            "trade_date": pd.date_range("2025-01-01", periods=6, freq="B"),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )

    target_frame = build_forward_realized_variance_target(
        underlying_daily=underlying_daily,
        horizon_days=5,
        annualization_factor=252,
        use_log_target=False,
    )

    assert len(target_frame) == 1

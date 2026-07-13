"""Tests for compact option-surface feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from options_rv.features.option_surface import build_option_surface_features


def test_feature_builder_selects_atm_and_term_points_correctly() -> None:
    """Builder computes ATM level and term slope from closest DTE slices."""
    options_quotes = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY", "SPY", "SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"] * 6),
            "expiry_date": pd.to_datetime(
                [
                    "2025-02-01",
                    "2025-03-03",
                    "2025-02-01",
                    "2025-03-03",
                    "2025-02-01",
                    "2025-03-03",
                ]
            ),
            "option_type": ["c", "c", "p", "p", "p", "p"],
            "strike_price": [600.0, 600.0, 590.0, 590.0, 600.0, 600.0],
            "bid": [4.0, 6.0, 3.0, 5.0, 3.5, 5.5],
            "ask": [4.4, 6.6, 3.5, 5.6, 3.9, 6.1],
            "bid_iv": [0.19, 0.24, 0.22, 0.27, 0.20, 0.25],
            "ask_iv": [0.21, 0.26, 0.24, 0.29, 0.22, 0.27],
            "open_interest": [1000, 1200, 800, 850, 900, 950],
            "volume": [400, 500, 300, 320, 330, 360],
        }
    )

    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "close": [600.0],
        }
    )

    trailing_realized_variance = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "trailing_annualized_variance_20d": [0.04],
        }
    )

    feature_frame = build_option_surface_features(
        options_quotes=options_quotes,
        underlying_daily=underlying_daily,
        trailing_variance_frame=trailing_realized_variance,
    )

    atm_30_call_mid_iv = (0.19 + 0.21) / 2.0
    atm_30_put_mid_iv = (0.20 + 0.22) / 2.0
    atm_60_call_mid_iv = (0.24 + 0.26) / 2.0
    atm_60_put_mid_iv = (0.25 + 0.27) / 2.0
    atm_30_mid_iv = (atm_30_call_mid_iv + atm_30_put_mid_iv) / 2.0
    atm_60_mid_iv = (atm_60_call_mid_iv + atm_60_put_mid_iv) / 2.0

    np.testing.assert_allclose(
        feature_frame.loc[0, "atm_iv_30d"], atm_30_mid_iv, rtol=1e-12
    )
    np.testing.assert_allclose(
        feature_frame.loc[0, "term_slope_60d_minus_30d"],
        atm_60_mid_iv - atm_30_mid_iv,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        feature_frame.loc[0, "trailing_annualized_variance_20d"], 0.04, rtol=1e-12
    )


def test_feature_builder_handles_missing_required_slice_with_nan() -> None:
    """Builder keeps deterministic row but leaves unavailable slices as NaN."""
    options_quotes = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "expiry_date": pd.to_datetime(["2025-02-01", "2025-02-01"]),
            "option_type": ["c", "p"],
            "strike_price": [600.0, 590.0],
            "bid": [4.0, 3.0],
            "ask": [4.4, 3.5],
            "bid_iv": [0.19, 0.22],
            "ask_iv": [0.21, 0.24],
            "open_interest": [1000, 800],
            "volume": [400, 300],
        }
    )

    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "close": [600.0],
        }
    )

    trailing_realized_variance = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "trailing_annualized_variance_20d": [0.04],
        }
    )

    feature_frame = build_option_surface_features(
        options_quotes=options_quotes,
        underlying_daily=underlying_daily,
        trailing_variance_frame=trailing_realized_variance,
    )

    assert np.isnan(feature_frame.loc[0, "term_slope_60d_minus_30d"])
    assert np.isnan(feature_frame.loc[0, "downside_skew_30d"])


def test_feature_builder_filters_crossed_quotes() -> None:
    """A crossed ATM quote must not influence the selected surface point."""
    options_quotes = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "expiry_date": pd.to_datetime(["2025-02-01", "2025-02-01"]),
            "option_type": ["c", "c"],
            "strike_price": [600.0, 610.0],
            "bid": [5.0, 4.0],
            "ask": [4.0, 4.5],
            "bid_iv": [0.20, 0.21],
            "ask_iv": [0.19, 0.23],
            "open_interest": [1000, 900],
            "volume": [400, 300],
        }
    )
    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "close": [600.0],
        }
    )
    trailing_variance = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "trailing_annualized_variance_20d": [0.04],
        }
    )

    features = build_option_surface_features(
        options_quotes=options_quotes,
        underlying_daily=underlying_daily,
        trailing_variance_frame=trailing_variance,
    )

    np.testing.assert_allclose(features.loc[0, "atm_iv_30d"], 0.22)


def test_feature_builder_rejects_nonpositive_underlying_close() -> None:
    """Log moneyness is undefined when the underlying close is nonpositive."""
    options_quotes = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "expiry_date": pd.to_datetime(["2025-02-01"]),
            "option_type": ["c"],
            "strike_price": [600.0],
            "bid": [4.0],
            "ask": [4.5],
            "bid_iv": [0.20],
            "ask_iv": [0.22],
            "open_interest": [1000],
            "volume": [400],
        }
    )
    underlying_daily = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "close": [0.0],
        }
    )
    trailing_variance = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "trailing_annualized_variance_20d": [0.04],
        }
    )

    with pytest.raises(ValueError, match="finite positive"):
        build_option_surface_features(
            options_quotes=options_quotes,
            underlying_daily=underlying_daily,
            trailing_variance_frame=trailing_variance,
        )

"""Simple forecast baselines for forward variance prediction."""

from __future__ import annotations

import pandas as pd


def compute_persistence_baseline(feature_target_frame: pd.DataFrame) -> pd.Series:
    """Return persistence forecast from trailing realized variance feature.

    Parameters
    ----------
    feature_target_frame
        Panel containing ``trailing_annualized_variance_20d``.

    Returns
    -------
    pd.Series
        Baseline variance-level prediction indexed like input frame.
    """
    # Persistence uses the same trailing variance feature the ridge model sees.
    return feature_target_frame["trailing_annualized_variance_20d"].astype(float)


def compute_atm_iv_baseline(feature_target_frame: pd.DataFrame) -> pd.Series:
    """Return ATM-implied baseline by converting volatility to variance.

    Parameters
    ----------
    feature_target_frame
        Panel containing ``atm_iv_30d``.

    Returns
    -------
    pd.Series
        Baseline variance-level prediction indexed like input frame.
    """
    # Variance is volatility squared under the standard Black-Scholes convention.
    atm_iv = feature_target_frame["atm_iv_30d"].astype(float)
    return atm_iv * atm_iv

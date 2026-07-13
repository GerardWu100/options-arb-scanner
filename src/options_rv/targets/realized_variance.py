"""Realized-variance target construction for short-horizon forecasting.

This module builds forward-looking annualized realized variance targets from an
underlying close series. It also optionally builds a trailing realized-variance
feature used by baselines and the linear model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_forward_realized_variance_target(
    underlying_daily: pd.DataFrame,
    horizon_days: int = 5,
    annualization_factor: int = 252,
    use_log_target: bool = True,
    trailing_window_days: int = 20,
) -> pd.DataFrame:
    """Build forward realized-variance target frame from daily closes.

    Parameters
    ----------
    underlying_daily
        Dataframe with columns ``symbol``, ``trade_date``, and ``close``.
    horizon_days
        Forecast horizon in trading days for forward realized variance.
    annualization_factor
        Annualization factor for daily data. Standard is ``252``.
    use_log_target
        Whether to include log-transformed target column.
    trailing_window_days
        Window size for trailing realized variance feature.

    Returns
    -------
    pd.DataFrame
        One row per symbol-date where forward window is fully available.

    Notes
    -----
    Definitions used in the implementation:

    - ``r_t = ln(S_t / S_{t-1})``
    - ``RV_{t,t+h} = sum_{i=1}^{h} r_{t+i}^2``
    - ``annualized_RV_{t,t+h} = (annualization_factor / h) * RV_{t,t+h}``

    The forward construction uses returns from ``t+1`` through ``t+h`` to avoid
    leakage from contemporaneous return information.
    """
    _validate_underlying_contract(underlying_daily=underlying_daily)
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if trailing_window_days <= 0:
        raise ValueError("trailing_window_days must be positive")

    forward_variance_column, log_forward_variance_column, trailing_variance_column = (
        _target_column_names(
            horizon_days=horizon_days,
            trailing_window_days=trailing_window_days,
        )
    )

    frame = underlying_daily.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # Daily log return and its square are the building blocks for realized variance.
    frame["log_return"] = np.log(
        frame["close"] / frame.groupby("symbol")["close"].shift(1)
    )
    frame["squared_log_return"] = frame["log_return"] ** 2

    # First form a trailing h-return sum ending at each row, then move that sum
    # back by h rows. The value aligned to date t therefore contains returns
    # ending on t+1 through t+h, never the return ending on t.
    forward_sum_squared_returns = frame.groupby("symbol")[
        "squared_log_return"
    ].transform(
        lambda series: series.rolling(
            window=horizon_days, min_periods=horizon_days
        )
        .sum()
        .shift(-horizon_days)
    )
    trailing_sum_squared_returns = frame.groupby("symbol")[
        "squared_log_return"
    ].transform(
        lambda series: series.rolling(
            window=trailing_window_days, min_periods=trailing_window_days
        ).sum()
    )

    frame[forward_variance_column] = (
        annualization_factor / float(horizon_days)
    ) * forward_sum_squared_returns
    frame[trailing_variance_column] = (
        annualization_factor / float(trailing_window_days)
    ) * trailing_sum_squared_returns

    if use_log_target:
        frame[log_forward_variance_column] = np.log(frame[forward_variance_column])

    output_columns = [
        "symbol",
        "trade_date",
        forward_variance_column,
        trailing_variance_column,
    ]
    if use_log_target:
        output_columns.append(log_forward_variance_column)

    output_frame = frame.loc[:, output_columns]
    # Rows without a full forward window cannot define a supervised label.
    output_frame = output_frame.dropna(subset=[forward_variance_column]).reset_index(
        drop=True
    )
    return output_frame


def _validate_underlying_contract(underlying_daily: pd.DataFrame) -> None:
    """Verify mandatory underlying columns are present."""
    required_columns = ["symbol", "trade_date", "close"]
    missing_columns = [
        column for column in required_columns if column not in underlying_daily.columns
    ]
    if missing_columns:
        raise ValueError(
            f"underlying_daily is missing required columns: {', '.join(missing_columns)}"
        )


def _target_column_names(
    horizon_days: int, trailing_window_days: int
) -> tuple[str, str, str]:
    """Return deterministic column names for forward, log-forward, and trailing variance."""
    forward_variance = f"annualized_forward_variance_{horizon_days}d"
    log_forward_variance = f"target_log_annualized_forward_variance_{horizon_days}d"
    trailing_variance = f"trailing_annualized_variance_{trailing_window_days}d"
    return forward_variance, log_forward_variance, trailing_variance

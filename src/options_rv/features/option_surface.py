"""Explainable option-surface feature engineering.

This module builds a compact daily feature panel that can be discussed clearly
in an interview setting. The design intentionally favors transparency over model
complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_option_surface_features(
    options_quotes: pd.DataFrame,
    underlying_daily: pd.DataFrame,
    trailing_variance_frame: pd.DataFrame,
    atm_dte_target_days: int = 30,
    long_dte_target_days: int = 60,
) -> pd.DataFrame:
    """Build one-row-per-symbol-date options feature panel.

    Parameters
    ----------
    options_quotes
        Options quotes with required contract columns including implied
        volatilities and bid/ask prices.
    underlying_daily
        Underlying daily close series with ``symbol``, ``trade_date``, ``close``.
    trailing_variance_frame
        Frame containing trailing realized variance feature with columns
        ``symbol``, ``trade_date``, and
        ``trailing_annualized_variance_20d``.
    atm_dte_target_days
        Target days-to-expiry used for ATM implied-vol level.
    long_dte_target_days
        Target days-to-expiry used for term slope against ATM target.

    Returns
    -------
    pd.DataFrame
        Deterministic daily feature panel keyed by ``symbol`` and ``trade_date``.
    """
    _validate_option_contract(options_quotes=options_quotes)
    _validate_underlying_contract(underlying_daily=underlying_daily)

    option_frame = options_quotes.copy()
    option_frame["trade_date"] = pd.to_datetime(
        option_frame["trade_date"]
    ).dt.normalize()
    option_frame["expiry_date"] = pd.to_datetime(
        option_frame["expiry_date"]
    ).dt.normalize()
    option_frame["mid_iv"] = 0.5 * (option_frame["bid_iv"] + option_frame["ask_iv"])
    option_frame["mid_price"] = 0.5 * (option_frame["bid"] + option_frame["ask"])
    option_frame = _filter_valid_quotes(option_frame=option_frame)
    option_frame["spread_ratio"] = (
        option_frame["ask"] - option_frame["bid"]
    ) / option_frame["mid_price"]

    spot_frame = underlying_daily.loc[:, ["symbol", "trade_date", "close"]].copy()
    spot_frame["trade_date"] = pd.to_datetime(spot_frame["trade_date"]).dt.normalize()

    option_frame = option_frame.merge(
        spot_frame,
        on=["symbol", "trade_date"],
        how="inner",
        validate="many_to_one",
    )
    option_frame["days_to_expiry"] = (
        option_frame["expiry_date"] - option_frame["trade_date"]
    ).dt.days
    option_frame = option_frame.loc[option_frame["days_to_expiry"] > 0].copy()

    # Moneyness is represented in log space so symmetric strike offsets around
    # spot become symmetric around zero.
    option_frame["log_moneyness"] = np.log(
        option_frame["strike_price"] / option_frame["close"]
    )

    feature_rows: list[dict[str, object]] = []
    grouped = option_frame.groupby(["symbol", "trade_date"], sort=True)
    for (symbol, trade_date), day_slice in grouped:
        feature_row = _build_feature_row_for_day(
            symbol=symbol,
            trade_date=trade_date,
            day_slice=day_slice,
            atm_dte_target_days=atm_dte_target_days,
            long_dte_target_days=long_dte_target_days,
        )
        feature_rows.append(feature_row)

    feature_frame = pd.DataFrame(feature_rows)
    feature_frame = feature_frame.sort_values(["symbol", "trade_date"]).reset_index(
        drop=True
    )

    trailing_variance_columns = [
        "symbol",
        "trade_date",
        "trailing_annualized_variance_20d",
    ]
    feature_frame = feature_frame.merge(
        trailing_variance_frame.loc[:, trailing_variance_columns],
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    return feature_frame


def _build_feature_row_for_day(
    symbol: str,
    trade_date: pd.Timestamp,
    day_slice: pd.DataFrame,
    atm_dte_target_days: int,
    long_dte_target_days: int,
) -> dict[str, object]:
    """Build one deterministic daily row from one symbol-date option slice."""
    atm_30_slice, atm_30_selected_dte = _slice_nearest_dte(
        day_slice=day_slice, target_dte_days=atm_dte_target_days
    )
    atm_60_slice, atm_60_selected_dte = _slice_nearest_dte(
        day_slice=day_slice, target_dte_days=long_dte_target_days
    )

    atm_iv_30d = _mid_iv_at_closest_strike(slice_frame=atm_30_slice)
    atm_iv_60d = _mid_iv_at_closest_strike(slice_frame=atm_60_slice)

    term_slope = np.nan
    if (
        np.isfinite(atm_iv_30d)
        and np.isfinite(atm_iv_60d)
        and atm_30_selected_dte is not None
        and atm_60_selected_dte is not None
        and atm_30_selected_dte != atm_60_selected_dte
    ):
        term_slope = atm_iv_60d - atm_iv_30d

    # Downside skew is defined as below-spot put IV minus ATM put IV at roughly
    # 30 DTE. This keeps the sign and interpretation stable.
    downside_put_iv_30d = _below_spot_put_mid_iv(slice_frame=atm_30_slice)
    atm_put_iv_30d = _mid_iv_at_closest_strike(
        slice_frame=atm_30_slice, option_type="p"
    )
    downside_skew_30d = np.nan
    if (
        np.isfinite(downside_put_iv_30d)
        and np.isfinite(atm_put_iv_30d)
        and _is_distinct_downside_and_atm_put(slice_frame=atm_30_slice)
    ):
        downside_skew_30d = downside_put_iv_30d - atm_put_iv_30d

    avg_spread_ratio = float(
        day_slice["spread_ratio"].replace([np.inf, -np.inf], np.nan).mean()
    )
    total_open_interest = float(day_slice["open_interest"].sum())

    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "atm_iv_30d": float(atm_iv_30d),
        "term_slope_60d_minus_30d": float(term_slope),
        "downside_skew_30d": float(downside_skew_30d),
        "avg_spread_ratio": avg_spread_ratio,
        "total_open_interest": total_open_interest,
    }


def _slice_nearest_dte(
    day_slice: pd.DataFrame,
    target_dte_days: int,
) -> tuple[pd.DataFrame, int | None]:
    """Select rows for nearest available DTE and return selected DTE value."""
    unique_dte = np.sort(day_slice["days_to_expiry"].dropna().unique())
    if len(unique_dte) == 0:
        return day_slice.iloc[0:0].copy(), None
    nearest_dte = unique_dte[np.argmin(np.abs(unique_dte - target_dte_days))]
    return day_slice.loc[day_slice["days_to_expiry"] == nearest_dte].copy(), int(
        nearest_dte
    )


def _mid_iv_at_closest_strike(
    slice_frame: pd.DataFrame,
    option_type: str | None = None,
) -> float:
    """Return mid IV at the strike closest to spot, optionally filtered by type."""
    if slice_frame.empty:
        return float("nan")

    candidates = slice_frame
    if option_type is not None:
        candidates = slice_frame.loc[slice_frame["option_type"] == option_type]
        if candidates.empty:
            return float("nan")

    minimum_distance = float(candidates["log_moneyness"].abs().min())
    closest_rows = candidates.loc[
        np.isclose(candidates["log_moneyness"].abs(), minimum_distance)
    ]
    # Call and put quotes can share the closest strike. Averaging their mids is
    # deterministic and avoids making the feature depend on input row order.
    return float(closest_rows["mid_iv"].mean())


def _below_spot_put_mid_iv(slice_frame: pd.DataFrame) -> float:
    """Return below-spot put IV from strike nearest to spot from below."""
    puts_below_spot = slice_frame.loc[
        (slice_frame["option_type"] == "p") & (slice_frame["log_moneyness"] < 0.0)
    ]
    if puts_below_spot.empty:
        return float("nan")
    selected_row = puts_below_spot.loc[puts_below_spot["log_moneyness"].abs().idxmin()]
    return float(selected_row["mid_iv"])


def _is_distinct_downside_and_atm_put(slice_frame: pd.DataFrame) -> bool:
    """Return whether downside and ATM put references are distinct quotes."""
    puts = slice_frame.loc[slice_frame["option_type"] == "p"].copy()
    if puts.empty:
        return False

    atm_put = puts.loc[puts["log_moneyness"].abs().idxmin()]
    downside_puts = puts.loc[puts["log_moneyness"] < 0.0]
    if downside_puts.empty:
        return False
    downside_put = downside_puts.loc[downside_puts["log_moneyness"].abs().idxmin()]

    same_strike = float(atm_put["strike_price"]) == float(downside_put["strike_price"])
    same_expiry = pd.Timestamp(atm_put["expiry_date"]) == pd.Timestamp(
        downside_put["expiry_date"]
    )
    return not (same_strike and same_expiry)


def _validate_option_contract(options_quotes: pd.DataFrame) -> None:
    """Validate required options columns for feature engineering."""
    required_columns = [
        "symbol",
        "trade_date",
        "expiry_date",
        "option_type",
        "strike_price",
        "bid",
        "ask",
        "bid_iv",
        "ask_iv",
        "open_interest",
        "volume",
    ]
    missing_columns = [
        column for column in required_columns if column not in options_quotes.columns
    ]
    if missing_columns:
        raise ValueError(
            f"options_quotes is missing required columns: {', '.join(missing_columns)}"
        )


def _filter_valid_quotes(option_frame: pd.DataFrame) -> pd.DataFrame:
    """Remove quotes that cannot support price or implied-volatility features.

    Parameters
    ----------
    option_frame
        Option records after mid-price and mid-implied-volatility construction.

    Returns
    -------
    pd.DataFrame
        Quotes with non-crossed prices, ordered positive implied volatilities,
        positive strikes, recognized option types, and non-negative activity.

    Raises
    ------
    ValueError
        If every supplied quote fails the quality checks.
    """
    numeric_columns = [
        "strike_price",
        "bid",
        "ask",
        "bid_iv",
        "ask_iv",
        "open_interest",
        "volume",
        "mid_price",
        "mid_iv",
    ]
    finite_numeric = np.isfinite(option_frame[numeric_columns]).all(axis=1)
    valid_mask = (
        finite_numeric
        & option_frame["option_type"].isin(["c", "p"])
        & option_frame["strike_price"].gt(0.0)
        & option_frame["bid"].ge(0.0)
        & option_frame["ask"].ge(option_frame["bid"])
        & option_frame["mid_price"].gt(0.0)
        & option_frame["bid_iv"].gt(0.0)
        & option_frame["ask_iv"].ge(option_frame["bid_iv"])
        & option_frame["open_interest"].ge(0.0)
        & option_frame["volume"].ge(0.0)
    )
    filtered_frame = option_frame.loc[valid_mask].copy()
    if filtered_frame.empty:
        raise ValueError("No valid option quotes remain after quote-quality filtering")
    return filtered_frame


def _validate_underlying_contract(underlying_daily: pd.DataFrame) -> None:
    """Validate required underlying columns for moneyness calculations."""
    required_columns = ["symbol", "trade_date", "close"]
    missing_columns = [
        column for column in required_columns if column not in underlying_daily.columns
    ]
    if missing_columns:
        raise ValueError(
            f"underlying_daily is missing required columns: {', '.join(missing_columns)}"
        )
    close_values = underlying_daily["close"].to_numpy(dtype=float)
    if not np.isfinite(close_values).all() or (close_values <= 0.0).any():
        raise ValueError("underlying_daily close must contain finite positive values")

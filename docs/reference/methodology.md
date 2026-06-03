# Methodology Reference

## Research Objective

Forecast 5-trading-day forward annualized realized variance for SPY using a
compact set of option-surface features.

## Symbols and Definitions

- `S_t`: underlying close at date `t`.
- `r_t = ln(S_t / S_{t-1})`: daily log return.
- `h = 5`: forecast horizon in trading days.
- `A = 252`: annualization factor for daily data.

Forward realized variance:

- `RV_{t,t+h} = sum_{i=1}^{h} r_{t+i}^2`
- `annualized_RV_{t,t+h} = (A / h) * RV_{t,t+h}`

Model target:

- primary modeling target: `log(annualized_RV_{t,t+h})`
- interpretation and baseline comparison: variance-level metrics

Trailing realized variance feature:

- `trailing_RV_20d_t = (A / 20) * sum_{i=0}^{19} r_{t-i}^2`

## Feature Set

Compact explainable features per symbol-date:

- `atm_iv_30d`: ATM mid implied volatility near 30 DTE.
- `term_slope_60d_minus_30d`: ATM mid IV near 60 DTE minus ATM mid IV near 30 DTE.
- `downside_skew_30d`: below-spot put IV minus ATM put IV near 30 DTE.
- `avg_spread_ratio`: average `(ask - bid) / mid_price` across daily chain slice.
- `total_open_interest`: daily sum of open interest.
- `trailing_annualized_variance_20d`: trailing realized variance baseline feature.

Mid implied volatility convention:

- `mid_iv = (bid_iv + ask_iv) / 2`

## Models

Baselines:

1. Persistence baseline from `trailing_annualized_variance_20d`.
2. ATM implied-volatility baseline from `atm_iv_30d^2`.

Main model:

- Ridge regression on standardized feature set.

## Split Design

Chronological split only:

- earliest rows: train
- middle rows: validation
- latest rows: test

No shuffling is used.

## Metrics

Reported per split and model:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- QLIKE in variance space:
  - `QLIKE = mean(log(f_t) + y_t / f_t)`
  - `y_t`: realized variance target
  - `f_t`: forecast variance (strictly positive)

## Leakage Controls

- Forward target uses returns from `t+1` to `t+h` only.
- Features are built from date `t` information only.
- Split logic is strictly chronological.

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

Implementation alignment:

- compute the trailing sum of `h` squared returns ending at each row
- shift that completed sum backward by `h` rows
- the label at `t` then contains only `r_{t+1}, ..., r_{t+h}`

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
- if call and put quotes share the closest ATM strike, average their mid IVs

Quote-quality filters remove records with crossed bid/ask prices, invalid or
nonpositive IV bounds, nonpositive strike or mid price, unknown option type, or
negative volume/open interest. These are validity checks rather than a complete
static-arbitrage screen. The pipeline does not enforce strike monotonicity,
butterfly convexity, calendar-spread conditions, or put-call parity.

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
- last `h` rows before validation and test: purged

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
- A horizon-length purge prevents labels on opposite sides of a split boundary
  from sharing forward returns.

## Interpretation Limits

- The raw cache is deterministic synthetic data, not an exchange snapshot.
- The 30-day ATM IV-squared baseline is annualized variance. Comparing it with
  annualized five-day realized variance assumes the annualized expected variance
  is flat enough across those horizons to be a useful benchmark.
- End-of-day timestamps do not encode whether the option snapshot precedes or
  follows the underlying close. Live research must define one synchronized
  snapshot convention.
- Exponentiating a log-variance regression produces a conditional median in
  level space unless a retransformation correction is applied. This can suit
  absolute-error loss but is not automatically optimal for squared-error loss.

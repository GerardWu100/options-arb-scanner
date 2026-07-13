---
title: "Can an Option Surface Forecast Realized Variance?"
description: "A leakage-controlled offline experiment with SPY option-surface features, simple benchmarks, and one result that refuses to flatter the model."
date: 2026-07-13
image: images/cover-options-rv.png
categories: ["Quantitative Research", "Risk Management"]
---

# Can an Option Surface Forecast Realized Variance?

The repository is called `options-arb-scanner`, but its current code no longer scans for arbitrage. It asks a narrower question: can a small set of end-of-day option-surface features forecast SPY's realized variance over the next five trading days?

That distinction matters. An arbitrage claim needs tradeable prices, execution rules, and a proof that the apparent profit survives costs. This project does none of those things. It builds a supervised forecasting experiment with an offline data contract, chronological evaluation, two simple benchmarks, and ridge regression. The honest subject of the post is the experiment the code runs today.

There is another constraint worth stating up front. The committed dataset is deterministic and synthetic. It covers 522 business-day observations from 2 January 2024 through 31 December 2025, with 20,880 option quotes. Each date has four expiries, five moneyness levels, and both calls and puts. This makes the pipeline portable and testable. It does not make the output evidence about the live SPY options market.

![A stylized option surface flowing into an uncertain realized-variance forecast](images/cover-options-rv.png)

The image captures the pipeline's basic compression: a two-dimensional option surface observed today becomes one forecast for a quantity that will only be known after five more sessions.

## Start with a target that cannot see the present

Let $S_t$ denote the SPY close on trading date $t$. The daily log return $r_t$, measured from $t-1$ to $t$, is

$$
r_t = \ln\left(\frac{S_t}{S_{t-1}}\right).
$$

The forecast horizon is $h=5$ trading days and the annualization factor is $A=252$ trading days per year. The annualized forward realized variance attached to date $t$ is

$$
RV^{(A)}_{t,t+h} = \frac{A}{h}\sum_{i=1}^{h}r_{t+i}^{2}.
$$

Every return in that sum occurs after date $t$. The first is $r_{t+1}$, not $r_t$. This is the small indexing choice on which the whole study rests: a feature observed at the close of $t$ must not be evaluated against a label that already includes the return ending at that close.

The implementation makes the shift explicit before applying the rolling window:

```python
forward_sum_squared_returns = frame.groupby("symbol")[
    "squared_log_return"
].transform(
    lambda series: (
        series.shift(-1)
        .rolling(window=horizon_days, min_periods=horizon_days)
        .sum()
    )
)
```

The model is fitted to $\ln(RV^{(A)}_{t,t+h})$ rather than the variance level. Logs keep fitted forecasts positive after exponentiation and reduce the influence of large variance observations. Metrics are still calculated in variance space, where the result is easier to interpret.

## Reduce the chain to six numbers

An option chain is not naturally a rectangular machine-learning table. Strike grids and expiration dates vary, while several quotes may be equally close to a desired point on the surface. The feature builder resolves this by applying the same deterministic selection rules on every symbol-date.

For a quote with bid implied volatility $\sigma^{bid}$ and ask implied volatility $\sigma^{ask}$, the mid implied volatility is

$$
\sigma^{mid} = \frac{\sigma^{bid}+\sigma^{ask}}{2}.
$$

The code selects the available expiration nearest 30 calendar days and the one nearest 60 calendar days. Within each slice, the at-the-money (ATM) quote is the strike with the smallest absolute log moneyness, where log moneyness is $\ln(K/S_t)$ and $K$ is strike price. If $\sigma_{30}$ and $\sigma_{60}$ denote those ATM mids, the term slope is

$$
\text{term slope}_t = \sigma_{60}-\sigma_{30}.
$$

For downside skew, the code uses puts near 30 days. Let $\sigma_{30}^{put,down}$ be the mid implied volatility of the nearest strike strictly below spot and let $\sigma_{30}^{put,ATM}$ be the ATM put mid. Then

$$
\text{downside skew}_t = \sigma_{30}^{put,down}-\sigma_{30}^{put,ATM}.
$$

The sign is intuitive: a positive number means the below-spot put carries more implied volatility than the ATM put.

Four option-derived features enter the model: ATM 30-day implied volatility, the 60-minus-30-day term slope, 30-day downside skew, and the average bid-ask spread divided by mid price. Total open interest supplies a fifth feature. Trailing 20-day annualized realized variance supplies the sixth and gives the regression access to the same recent-variance information used by the persistence benchmark.

This nearest-expiry rule is simple, but it is not interpolation. A production study would normally interpolate total variance to fixed maturities and define delta-based, rather than nearest-strike, skew points. Here, determinism and inspectability take priority over a smoother surface estimate.

## Give the model benchmarks it should struggle to beat

The first benchmark is persistence: forecast the next five-day variance with trailing 20-day annualized realized variance. The second squares 30-day ATM implied volatility, converting an annualized volatility quote into variance.

The main model is ridge regression. Let $x_t$ be the six standardized features on date $t$, let $y_t=\ln(RV^{(A)}_{t,t+5})$, let $\beta$ be the coefficient vector, and let $\alpha=1$ be the fixed penalty weight. With $n$ training observations, the fitted coefficients minimize

$$
\sum_{t=1}^{n}\left(y_t-x_t^{\mathsf{T}}\beta\right)^2
+\alpha\sum_{j=1}^{6}\beta_j^2.
$$

The penalty shrinks unstable coefficients toward zero. Standardization is fitted inside the training pipeline, so each coefficient refers to a one-standard-deviation feature move. The earliest 60% of complete rows train the model, the next 20% form validation, and the latest 20% remain held out. No rows are shuffled.

The final panel has 501 complete observations:

| Split | Rows | First date | Last date |
| --- | ---: | --- | --- |
| Train | 300 | 2024-01-30 | 2025-03-24 |
| Validation | 100 | 2025-03-25 | 2025-08-11 |
| Test | 101 | 2025-08-12 | 2025-12-30 |

The study reports root mean squared error (RMSE), mean absolute error (MAE), and QLIKE. RMSE squares errors before averaging, so large misses receive more weight. MAE averages absolute errors. For realized variance $v_t$ and a strictly positive forecast $f_t$, QLIKE is

$$
QLIKE = \frac{1}{n}\sum_{t=1}^{n}\left[\ln(f_t)+\frac{v_t}{f_t}\right].
$$

Lower is better for all three losses. QLIKE can be negative because variance is measured in decimal units. Its level is less informative than a comparison made on the same sample.

## The synthetic result does not flatter ridge

The held-out result is unambiguous. Persistence has the lowest RMSE, MAE, and QLIKE. Ridge misses persistence by about 17% on MAE. The squared-ATM-implied-volatility benchmark is roughly 3,033 times worse than persistence on MAE.

| Model | Test RMSE | Test MAE | Test QLIKE | MAE / persistence |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 1.23e-05 | 9.96e-06 | -10.007959 | 1.00x |
| ATM IV squared | 3.03e-02 | 3.02e-02 | -3.502271 | 3,033.11x |
| Ridge | 1.52e-05 | 1.17e-05 | -8.764908 | 1.17x |

![Realized and forecast variance through the held-out test period](images/01_test_forecasts.png)

The logarithmic vertical axis is not cosmetic. Without it, the ATM-implied series would flatten the realized, persistence, and ridge lines against zero. In the synthetic fixture, daily underlying returns have a standard deviation of only 0.024%, while ATM implied volatility sits around ordinary market-looking levels. Squaring that implied volatility produces forecasts near $0.03$. Realized variance is closer to $10^{-5}$. The two simulated processes were not calibrated to each other.

![Test mean absolute error relative to the persistence benchmark](images/02_relative_mae.png)

The comparison catches exactly the kind of problem a benchmark is supposed to catch. Squared implied volatility is dimensionally a variance forecast, but dimensional consistency alone does not guarantee calibration. Ridge can fit combinations of the synthetic trends in the training segment, yet that does not create stable predictive information in the held-out segment. The late test forecasts drift below the target while persistence continues to track its scale.

These numbers are useful as software evidence. They show that the loader, feature construction, split, training, inverse log transform, metrics, and figures run end to end. They are not estimates of how much information a real option surface contains.

## What I would change before treating this as research

The first replacement is the data. Real SPY closes need an adjusted-price convention, an exchange-session calendar, and a documented snapshot time. Real option quotes need stale-quote filters, crossed-market checks, minimum bid and open-interest rules, and a consistent treatment of dividends and rates. The synthetic calendar uses business days, which can include market holidays.

The second change is the surface representation. I would interpolate total implied variance to fixed maturities, compute skew at fixed deltas, and record coverage diagnostics for every date. That avoids allowing a changing strike or expiry grid to masquerade as a signal.

The third change is evaluation. Five-day forward targets overlap, so neighboring labels share four returns. A standard chronological split prevents direct lookahead, but it does not remove dependence at the train-validation or validation-test boundary. Purging at least $h=5$ observations around each boundary would give a cleaner out-of-sample comparison. Rolling or expanding-window refits would also show whether coefficients survive different regimes.

Finally, model selection should happen on validation data only, with the test period opened once. Useful additions include a HAR-RV model (heterogeneous autoregressive realized variance), an implied-minus-realized variance feature, Diebold-Mariano forecast-comparison tests with overlap-aware standard errors, and economic evaluation tied to an actual variance trade. Until then, the right conclusion is modest: this is a sound offline research skeleton, and its current synthetic result is a test of the skeleton rather than a finding about markets.

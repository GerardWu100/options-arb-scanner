---
title: "Can an Option Surface Forecast Realized Variance?"
description: "An audited SPY variance-forecasting experiment: corrected target alignment, purged evaluation, option-surface logic, and a synthetic result that persistence wins."
date: 2026-07-13
image: images/cover-options-rv.png
categories: ["Quantitative Research", "Risk Management"]
---

# Can an Option Surface Forecast Realized Variance?

The repository is called `options-arb-scanner`, but the current code does not scan for arbitrage. It compresses an end-of-day SPY option chain into six features and asks whether they forecast annualized realized variance over the next five trading days.

Those are different research problems. A relative-value scanner compares executable option prices against no-arbitrage relationships or a pricing model, then accounts for transaction costs and hedge execution. This project fits a statistical forecast. Its output is a variance estimate, not a trade or a guaranteed profit.

The distinction became more than editorial during a second audit. The original target implementation had an off-by-one error: its rolling operation could include the return ending on the feature date. A constant-return unit test passed because every possible window had the same value. A non-constant hand example exposed the defect. The corrected pipeline also purges overlapping labels at split boundaries and rejects individually invalid quotes.

![A stylized option surface flowing into an uncertain realized-variance forecast](images/cover-options-rv.png)

The committed sample contains 522 business-day rows from 2 January 2024 through 31 December 2025 and 20,880 synthetic option quotes. It is deterministic test data. The results below test the research pipeline, not the live SPY options market.

## Define the quantity before forecasting it

Let $S_t$ be the SPY closing price on trading date $t$. The close-to-close log return ending on $t$ is $r_t$:

$$
r_t = \ln\left(\frac{S_t}{S_{t-1}}\right).
$$

Let $h=5$ be the forecast horizon in trading days and let $A=252$ be the annualization factor in trading days per year. The unannualized forward realized variance known after date $t+h$ is

$$
RV_{t,t+h} = \sum_{i=1}^{h}r_{t+i}^{2}.
$$

Multiplying the average daily squared return by $A$ gives the annualized target:

$$
RV^{(A)}_{t,t+h}
= A\left(\frac{1}{h}\sum_{i=1}^{h}r_{t+i}^{2}\right)
= \frac{A}{h}RV_{t,t+h}.
$$

Every return in this label ends after $t$. The implementation now forms a completed trailing sum and moves the sum back by $h$ rows:

```python
forward_sum_squared_returns = frame.groupby("symbol")["squared_log_return"].transform(
    lambda series: (
        series.rolling(window=horizon_days, min_periods=horizon_days)
        .sum()
        .shift(-horizon_days)
    )
)
```

For a two-day hand check, suppose the returns ending on dates 1 through 4 are $0.01$, $0.02$, $0.03$, and $0.04$. The target on date 0 must use $0.01^2+0.02^2$. The target on date 1 must use $0.02^2+0.03^2$. The new unit test checks both values. This catches an indexing error that a constant sequence cannot reveal.

This realized-variance definition follows the high-frequency variance literature, although the present project uses daily returns rather than intraday returns. [Andersen, Bollerslev, Diebold, and Labys (2003)](https://doi.org/10.1111/1468-0262.00418) give the empirical foundation for modeling and forecasting realized volatility.

## Turn an irregular chain into six daily features

For each option quote, let $b$ be the bid price and $a$ be the ask price. Let $\sigma^{bid}$ and $\sigma^{ask}$ be the implied volatilities backed out from those two prices. The code defines the price and implied-volatility mids as

$$
m = \frac{a+b}{2}, \qquad
\sigma^{mid} = \frac{\sigma^{ask}+\sigma^{bid}}{2}.
$$

Before building a surface point, the pipeline removes crossed prices ($a<b$), nonpositive mids or strikes, nonpositive or reversed implied-volatility bounds, unknown option types, and negative volume or open interest. If a call and put share the closest at-the-money (ATM) strike, their mid implied volatilities are averaged. That makes the feature independent of input row order.

The expiration nearest 30 calendar days supplies the short point. The expiration nearest 60 days supplies the long point. Let $K$ be strike price. Log moneyness is $\ln(K/S_t)$, so the ATM strike minimizes $|\ln(K/S_t)|$. If $\sigma_{30}$ and $\sigma_{60}$ are the selected ATM mids, then

$$
\text{term slope}_t = \sigma_{60}-\sigma_{30}.
$$

For downside skew, let $\sigma_{30}^{put,down}$ be the mid implied volatility at the closest put strike strictly below spot, and let $\sigma_{30}^{put,ATM}$ be the ATM put mid. Then

$$
\text{downside skew}_t
= \sigma_{30}^{put,down}-\sigma_{30}^{put,ATM}.
$$

A positive value says the below-spot put has higher implied volatility than the ATM put. The complete feature vector contains ATM 30-day implied volatility, the 60-minus-30-day term slope, downside skew, average relative bid-ask spread $(a-b)/m$, total open interest, and trailing 20-day annualized realized variance.

Nearest-strike and nearest-expiry selection is inspectable, but crude. A market study would interpolate total implied variance $\sigma^2\tau$, where $\tau$ is time to expiry in years, to fixed maturities and calculate skew at fixed option deltas. The [Cboe VIX methodology](https://cdn.cboe.com/api/global/us_indices/governance/VIX_Methodology.pdf) shows a model-free, multi-strike approach to extracting a constant-maturity variance measure from SPX options. This project uses SPY options, and squaring one ATM quote is not the VIX calculation.

## Quote validity is not an arbitrage test

The new filters establish that a single record is usable. They do not establish that the surface is free of static arbitrage.

For European calls with the same expiry, let $C(K)$ be the call price as a function of strike $K$. If $K_1<K_2$, absence of vertical-spread arbitrage requires

$$
C(K_1) \geq C(K_2).
$$

For equally spaced strikes $K_1<K_2<K_3$, absence of butterfly arbitrage requires convexity:

$$
C(K_1)-2C(K_2)+C(K_3) \geq 0.
$$

Let $P(K)$ be the corresponding put price, $r$ the continuously compounded risk-free rate, $q$ the continuous dividend yield, and $\tau$ time to expiry in years. European put-call parity requires

$$
C(K)-P(K)=S_t e^{-q\tau}-K e^{-r\tau}.
$$

The pipeline checks none of these cross-quote conditions. The equations above are European conditions, while listed SPY options permit American-style early exercise, so a live scanner would need the appropriate exercise-aware bounds. This project does not record exercise style. It also lacks executable size, fees, hedge slippage, borrow constraints, and synchronized timestamps. [Davis and Hobson (2007)](https://doi.org/10.1111/j.1467-9965.2007.00291.x) study option-price bounds and the arbitrage logic behind them. Calling this code an arbitrage scanner would overstate its scope.

## Benchmarks, ridge, and a purged clock

The persistence benchmark forecasts five-day annualized variance with trailing 20-day annualized variance. The option benchmark squares 30-day ATM implied volatility:

$$
f_t^{IV}=\sigma_{30,t}^{2}.
$$

Both $f_t^{IV}$ and $RV^{(A)}_{t,t+5}$ are annualized variance, so no extra factor of $5/252$ belongs in the comparison. Their matching units hide a strong economic assumption: a 30-day risk-neutral implied variance must proxy a five-day physical expected variance. Horizon mismatch and the variance risk premium can break that link even with perfect data.

The main model is ridge regression. Let $x_t$ be the six standardized features, $y_t=\ln(RV^{(A)}_{t,t+5})$, $b$ be an intercept, $\beta$ be the six coefficients, $n$ be the number of training rows, and $\alpha=1$ be the penalty weight. The fitted parameters minimize

$$
\sum_{t=1}^{n}\left(y_t-b-x_t^{\mathsf{T}}\beta\right)^2
+\alpha\sum_{j=1}^{6}\beta_j^2.
$$

Ridge regression was introduced by [Hoerl and Kennard (1970)](https://doi.org/10.1080/00401706.1970.10488634). Standardization is learned on the training rows only. Exponentiating a log forecast produces a positive level forecast, but without a retransformation correction it estimates a conditional median under common assumptions, not the conditional mean. That is compatible with mean absolute error (MAE) more naturally than with root mean squared error (RMSE). The experiment reports both, so this objective mismatch remains a limitation.

The split is chronological: 60% train, 20% validation, and 20% test before purging. Adjacent five-day labels share four future returns. The last five rows before validation and test are therefore marked `purged` and excluded from every metric and from model fitting.

| Segment | Rows | First date | Last date |
| --- | ---: | --- | --- |
| Train | 293 | 2024-01-30 | 2025-03-13 |
| Purged before validation | 5 | 2025-03-14 | 2025-03-20 |
| Validation | 94 | 2025-03-21 | 2025-07-30 |
| Purged before test | 5 | 2025-07-31 | 2025-08-06 |
| Test | 100 | 2025-08-07 | 2025-12-24 |

The losses are RMSE, MAE, and QLIKE. Let $v_t>0$ be realized variance, let $f_t>0$ be forecast variance, and let $N$ be the number of evaluated rows. QLIKE is

$$
QLIKE = \frac{1}{N}\sum_{t=1}^{N}
\left[\ln(f_t)+\frac{v_t}{f_t}\right].
$$

Lower is better for all three. QLIKE can be negative when variance is expressed in decimals. [Patton (2011)](https://doi.org/10.1016/j.jeconom.2010.03.034) explains why loss choice matters when volatility itself is measured with error.

## Persistence still wins the corrected test

After correcting the label, purging ten boundary rows, and rebuilding the ATM feature deterministically, persistence has the lowest held-out RMSE, MAE, and QLIKE. Ridge has 5.8% higher MAE. ATM implied volatility squared has about 2,135 times the persistence MAE.

| Model | Test RMSE | Test MAE | Test QLIKE | MAE / persistence |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 1.76e-05 | 1.43e-05 | -9.582540 | 1.00x |
| ATM IV squared | 3.06e-02 | 3.05e-02 | -3.492651 | 2,135.39x |
| Ridge | 1.99e-05 | 1.51e-05 | -7.447701 | 1.06x |

![Realized and forecast variance through the held-out test period](images/01_test_forecasts.png)

The logarithmic vertical axis is necessary. The synthetic underlying has daily return volatility of 0.024%, while ATM implied volatility stays near ordinary market-looking levels. Squaring that implied volatility produces annualized variance near $0.03$. Realized variance in the test set ranges from $1.31\times10^{-7}$ to $5.96\times10^{-5}$. The generator did not calibrate its option and underlying processes to each other.

![Test mean absolute error relative to the persistence benchmark](images/02_relative_mae.png)

The second chart makes two points. The implied-volatility benchmark is dimensionally correct but economically uncalibrated. Ridge stays close to persistence on MAE, yet fails to beat it and performs much worse on QLIKE. Nothing here supports a claim that option features improve SPY variance forecasts.

## What the experiment can and cannot establish

The fresh run, 18 unit tests, and non-interactive notebook execution establish that the offline loader, corrected target, quote filters, purged split, model, metrics, and charts work together. The frozen CSV files under `blog/data/` reproduce every plotted test value.

They do not establish market predictability. A live study still needs exchange-session dates, adjusted underlying prices, a precise option snapshot time at or before the feature timestamp, stale-quote rules, fixed-maturity interpolation, delta-based skew, rates and dividends, and complete static-arbitrage diagnostics. Model and penalty choices should be made on validation data, with the test period opened once. Rolling refits would reveal coefficient instability.

Forecast comparisons also need uncertainty estimates that respect overlapping horizons. A Diebold-Mariano test, introduced by [Diebold and Mariano (1995)](https://doi.org/10.1080/07350015.1995.10524599), would require an overlap-aware long-run variance estimate here. Economic value would require a specified variance trade and all execution costs.

The corrected conclusion is narrow and useful: this repository is now a cleaner variance-forecasting scaffold. Its synthetic benchmark result says persistence wins this fixture. It says nothing yet about arbitrage or a tradeable edge.

## References

- Andersen, T. G., Bollerslev, T., Diebold, F. X., and Labys, P. (2003), [“Modeling and Forecasting Realized Volatility”](https://doi.org/10.1111/1468-0262.00418), *Econometrica* 71(2), 579–625.
- Cboe Global Indices, [*VIX Index Methodology*](https://cdn.cboe.com/api/global/us_indices/governance/VIX_Methodology.pdf).
- Davis, M. H. A., and Hobson, D. G. (2007), [“The Range of Traded Option Prices”](https://doi.org/10.1111/j.1467-9965.2007.00291.x), *Mathematical Finance* 17(1), 1–14.
- Diebold, F. X., and Mariano, R. S. (1995), [“Comparing Predictive Accuracy”](https://doi.org/10.1080/07350015.1995.10524599), *Journal of Business & Economic Statistics* 13(3), 253–263.
- Hoerl, A. E., and Kennard, R. W. (1970), [“Ridge Regression: Biased Estimation for Nonorthogonal Problems”](https://doi.org/10.1080/00401706.1970.10488634), *Technometrics* 12(1), 55–67.
- Patton, A. J. (2011), [“Volatility Forecast Comparison Using Imperfect Volatility Proxies”](https://doi.org/10.1016/j.jeconom.2010.03.034), *Journal of Econometrics* 160(1), 246–256.

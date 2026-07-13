# Outline proposal

## Project scan summary

- Project archetype candidate: mixed `risk-model` and `data-pipeline`, with the risk-model blueprint dominant.
- Supporting evidence from files: `realized_variance.py` now aligns a five-trading-day target strictly after the feature date; `option_surface.py` validates quotes and reduces each chain to six explainable features; `offline_research.py` compares persistence, at-the-money implied variance, and ridge regression using purged chronological splits. The audit found and corrected an off-by-one label defect. The committed manifest identifies deterministic synthetic inputs, so this is a pipeline demonstration rather than market evidence.

## Blueprint selection

- Selected blueprint: adapted risk-model blueprint with data-pipeline elements.
- Why this blueprint fits this project: the analytical question is forecast comparison in variance space, while the code's strongest contribution is a reproducible offline path from validated Parquet files to leakage-controlled evaluation.
- Planned section order:
  1. The question hidden behind the repository name
  2. Define the five-day forward target
  3. Compress an option chain into six daily features
  4. Distinguish quote validity from static no-arbitrage
  5. Compare benchmarks and ridge on a purged clock
  6. What the corrected synthetic run says
  7. What would need to change for a real study

## Planned equations

1. Daily log return:
   - Purpose: define the return input to realized variance.
   - Symbols: $S_t$ is the SPY close on date $t$ and $r_t$ is the close-to-close log return ending on date $t$.
   - Delimiter: display.
2. Five-day forward annualized realized variance:
   - Purpose: show exactly which future returns form the label and how the daily sum is annualized.
   - Symbols: $h=5$ trading days, $A=252$ trading days per year, and $RV^{(A)}_{t,t+h}$ is annualized forward variance known only after date $t+h$.
   - Delimiter: display.
3. Implied-volatility mid and two option-surface spreads:
   - Purpose: define the at-the-money level, term slope, and downside skew.
   - Symbols: $sigma^{bid}$ and $sigma^{ask}$ are quoted implied volatilities; $sigma_{30}$ and $sigma_{60}$ are nearest available at-the-money mids; $sigma_{30}^{put,down}$ is the nearest below-spot put mid.
   - Delimiter: display.
4. Ridge objective and QLIKE loss:
   - Purpose: distinguish the log-variance training objective from variance-level evaluation.
   - Symbols: $y_t$ is log forward variance, $x_t$ is the standardized feature vector, $\beta$ is the coefficient vector, $\alpha=1$ is the penalty, $v_t$ is realized variance, and $f_t>0$ is forecast variance.
   - Delimiter: display.
5. Static no-arbitrage conditions:
   - Purpose: show why filtering crossed individual quotes does not make this an arbitrage scanner.
   - Symbols: $C(K)$ and $P(K)$ are European call and put prices at strike $K$; $r$ is the risk-free rate, $q$ is dividend yield, and $\tau$ is time to expiry.
   - Delimiter: display.

## Planned code excerpts

1. File: `src/options_rv/targets/realized_variance.py`
   - Function/block: the shifted rolling window that begins at $t+1$.
   - Why include this excerpt: it is the most important leakage-control detail in the implementation.
2. File: `src/options_rv/features/option_surface.py`
   - Function/block: nearest-maturity selection and downside-skew construction.
   - Why include this excerpt: it shows how an irregular chain becomes deterministic daily features.

## Planned technical graphs

1. Graph type: test-period time series on a logarithmic variance axis.
   - Source: generate from a fresh offline pipeline run; freeze the selected test predictions under `blog/data/`.
   - Expected takeaway: the persistence forecast stays close to the synthetic target, the ridge forecast drifts low late in the sample, and squared at-the-money implied volatility is orders of magnitude too high.
2. Graph type: test mean absolute error relative to persistence.
   - Source: generate from the pipeline metric table; freeze the test metrics under `blog/data/`.
   - Expected takeaway: persistence is the correct benchmark winner in this fixture; the chart also makes the scale mismatch of the implied-volatility baseline impossible to miss.

## Risks, gaps, and assumptions

- Data gaps: the committed sample is synthetic, contains one symbol, and is not suitable for claims about live option markets or forecast profitability.
- Assumptions: implied volatilities are decimal annualized volatilities, variance predictions must stay positive for QLIKE, split proportions are 60% train, 20% validation, and 20% test before purging, and a 30-day risk-neutral variance proxy is compared with a five-day physical realized-variance target.
- Validation checks to run before final draft: execute all tests; rerun the command-line pipeline; confirm row counts and split dates; regenerate both charts; verify every Markdown image path; run the blog validator on English and French files.
- Workspace and deployment: canonical materials remain in `options-arb-scanner/blog/`. Per the user's instruction, there is no publish bundle, Hugo build, website commit, or deployment in this task. Only the current project repository will be committed and pushed.

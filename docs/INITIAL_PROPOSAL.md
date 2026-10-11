# Preliminary Project Proposal / Technical Execution Plan

**Course:** Data Driven Methods in Finance. **Catalog #67:** Wood, Roberts & Zohren (2022), *Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection*, arXiv:2105.13727. **Group:** Enze Hu (eh3203), Jinrui Hu (jh5260). This is a preliminary research proposal, **not a results report**. Full formatted 3-page DOCX is provided separately in the project conversation.

## 1. Research Objective and Hypothesis
Momentum strategies can be slow to react when a market trend changes. We will test whether adding **online changepoint detection (CPD)** to an **LSTM Deep Momentum Network** improves its **out-of-sample performance after transaction costs**. CPD estimates when a recent change may have occurred and its severity. Compare **TSMOM, LSTM, and LSTM+CPD** on exactly the same sample, chronology, risk scaling and fee assumptions. Test whether the CPD variant improves OOS **net Sharpe**, especially near reversals; no improvement is assumed. Adapt **Exhibit 3** and the transaction-cost analysis in **Appendix C**; additionally evaluate returns, maximum drawdown and turnover.

## 2. Data and Variable Construction
Use **daily LSEG Datastream via WRDS, approximately 2010–2025**. Initial E-mini S&P 500 continuous excess-return index **ISMCS04** contains **4,033 observations**; separately checked expiry/identifier records cover **64 quarterly contracts** (per team). Verify return-index conventions/contract rolls and add rates, FX, commodities and other equity-index futures after coverage auditing. The original paper used **50 Pinnacle CLC futures from 1990–2020**, so WRDS results are an explicitly **adapted partial replication**. Calculate index return `r(i,t)=I(i,t)/I(i,t-1)-1`, 1/21/63/126/252-session trailing normalized returns, MACD and EWMA60 volatility. CPD inputs are past standardized returns. Sort/deduplicate, audit missingness, warm-up, expiry rolls, settlement issues and negative prices; no future-data imputation.

## 3. Proposed Methodology
TSMOM takes sign of trailing 252-day returns. Baseline **63-time-step LSTM** with tanh output selects positions `X(i,t)∈[-1,1]`. CPD-LSTM augments exactly the same network features with normalized **severity and location** of changepoints estimated by comparing a stationary Matérn 3/2 Gaussian Process and a GP changepoint kernel over trailing **21/63-day** windows (10/126/252 optional). LSTM trains on negative Sharpe of next-period captured P&L with Adam and validation-only hyperparameter/early stopping. Daily risk-scaled weight `w(i,t)=X(i,t)*0.15/(N*sigma_annual(i,t))`, trailing EWMA60 `sigma_annual`, same risk/leverage limits; apply timestamp-causal execution lag. `R_net(t+1)=Σ_i w(i,t)r(i,t+1)-10^-4*c*traded_notional(t+1)`; charge initial trades, position changes, flips and roll legs; **c = 0,1,2,3,5 bps**, one-way scenario costs. Provisional WRDS chronology: **2010–2016 training**, **2017–2018 validation**, **2019–2020 locked out-of-sample**. 2021–2025 optional separate holdout if comparable coverage; paper original used five-year expanding OOS 1995–2020. **Only after core runs:** cost-aware CPD-LSTM optimizing net-Sharpe/turnover objective with an otherwise identical baseline.

## 4. Technical Implementation / Research Pipeline
WRDS index/expiry/contract audit → clean causal daily returns → normalized momentum/MACD/volatility features → GP-CPD severity/location → chronological folds → matched TSMOM/LSTM/CPD-LSTM estimation → lagged weights → fee-aware gross/net backtests → diagnostics and reports. Already implemented: WRDS course-cache adapters, futures-contract roll/return audit, causal generic feature construction, purged splits, TSMOM/cost engine and synthetic tests. **No trained LSTM or GP-CPD model is yet complete.** Official GP-CPD code is vendored from `kieranjwood/slow-momentum-fast-reversion`; full author LSTM module is supported by `kieranjwood/trading-momentum-transformer` (the Transformer itself is **not** the chosen paper). TensorFlow/GPflow legacy environment requires isolation.

## 5. Evaluation and Validation
Primary metric: matched **OOS net Sharpe LSTM+CPD minus LSTM**, with TSMOM benchmark. Report annualized gross/net return and volatility, Sharpe, max drawdown, turnover, cost drag and cumulative wealth on common test dates; present 0/1/2/3/5 bps sensitivity and CPD 21/63-day windows. Test seed dispersion and paired return uncertainty when sample permits; never select on test-period results.

## 6. Robustness and Potential Issues
(1) Look-ahead leakage: use past-only CPD/normalization/universe, purge boundary-crossing next-day labels; execution-delay sensitivity. (2) Contract roll / index mismatch: an excess-return index embeds vendor methodology; expiry metadata does not guarantee executable roll prices; audit 64 contracts, missed sessions, nonpositive settlements and possible double costs. (3) Small universe/overfit: the initial single E-mini is an engineering pilot, not diversified replication; restrict tuning, compare identical dates, CPD windows and seeds. (4) Frictions: hypothetical bps are not measured fees, report break-even cost and stress scenarios.

## 7. Expected Results and Deliverables
Auditable WRDS data dictionary and roll log; dated CPD/features; TSMOM/LSTM/CPD-LSTM configurations, weights and OOS return series; partial Exhibit 3 results table and Appendix C cost/turnover charts; drawdown/wealth panels; reproducible code, final report and presentation. Original **gross** Sharpe **1.62 (LSTM)** and **2.16 (CPD-LSTM)** are literature context, not claims of our own empirical results.

## 8. Work Plan
- **Data (Oct 11–18):** Enze verifies ISMCS04, the 64 expiries, missingness and diversified future candidates; Jinrui cross-checks.
- **Research (Oct 19–25):** jointly freeze feature definitions, split/cost protocol and one-index pilot.
- **Implementation (Oct 26–Nov 15):** Jinrui owns GP-CPD/LSTM training; Enze owns WRDS preprocessing, futures backtest and fees.
- **Validation (Nov 16–29):** joint matched OOS comparisons, CPD-window/seed/timing and turnover-cost sensitivity.
- **Analysis (Nov 30–Dec 6):** interpret reversal behavior and limitations; if ready, cost-aware loss and separate 2021–2025 holdout.
- **Finalization (Dec 7–11):** joint final report, evidence tables, repository README and slides. **Internal dates only; official course deadlines prevail.**

**References:** [Wood et al. 2022, arXiv:2105.13727](https://arxiv.org/abs/2105.13727); [official CPD](https://github.com/kieranjwood/slow-momentum-fast-reversion); [author LSTM/CPD](https://github.com/kieranjwood/trading-momentum-transformer); [optional DeePM](https://github.com/kieranjwood/deepm).

# Data selection — paper #67 CPD-LSTM futures replication

The correct selected paper is Wood, Roberts & Zohren (2022), *Slow Momentum with Fast Reversion*, arXiv:2105.13727. The paper uses 50 **Pinnacle CLC** continuous futures (1990–2020), not provided in the official GitHub repository.

## Current WRDS evidence and decisions

Team's initial plan reports daily **LSEG Datastream via WRDS**, aiming for 2010–2025: E-mini S&P 500 continuous **excess-return index ISMCS04** (4,033 observations), plus identifiers and last-trading dates for **64 quarterly futures expirations**. Verify these counts, continuous-index conventions and actual timestamp availability before fitting. Existing local course cache has ten futures series but is not the published 50-futures panel. Expand to stock-index, rates, FX and commodity futures only with verified comparable daily histories.

**Distinguish:** continuous excess-return index ratio changes vs contract-level future settlement P&L. Expiration metadata is helpful for auditing rolls but not sufficient to reconstruct fillable futures returns. Avoid double counting an index's embedded roll mechanics and standalone contract-roll costs. For nonpositive futures prices (e.g., WTI in 2020), percentage returns may be invalid: use audited dollar P&L with contract multipliers or disclose exclusions. Verify currency, point value, dates, series revisions, stale prices, missing sessions, ex-ante universe selection, no backfilling and any causal roll dates.

## Data access and storage

The personal interactive WRDS username/password workflow remains in the local Enze worktree; **shared main has only `scripts/01_get_data.py` preview**, and no login secrets/vendor data. The local `scripts/05_course_pilot.py` imports preexisting vetted contract data. Do not redistribute WRDS history under academic license. Other potential data sources (original Pinnacle, old Nasdaq Data Link CHRIS example, IBKR API) are alternatives **only after entitlement/coverage verification**.

Freeze jointly: exact tickers and contract codes, dates/quality, historical eligibility, index vs contract meaning, lag/execution convention, 60-day EWMA risk scaling, CPD 21/63 windows, 2010–2016 train / 2017–2018 validation / 2019–2020 primary OOS and separate potential 2021–2025 holdout, and one-way 0/1/2/3/5 bps cost stress scenarios. Any substitute dataset is labeled **partial/methodological replication**.

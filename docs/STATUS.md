# Status — paper #67, 2026-10-11

**Selected study:** Wood, Roberts & Zohren (2022), *Slow Momentum with Fast Reversion*, arXiv:2105.13727. History is preserved in backup branch `backup-before-paper67-correction-20261011`.

**Previously built and reusable:** pinned author GP-CPD + supporting LSTM code snapshots, futures contract expiry and local WRDS-cache input adapters, past-only momentum/MACD features, purged time splits, TSMOM risk/cost backtests, roll legs, and synthetic tests. Interactive authenticated WRDS downloader remains only in Enze's local branch; raw licensed data/credentials are NOT in main.

**Team-reported actual data:** LSEG Datastream / WRDS E-mini futures excess-return series ISMCS04, 4,033 observations, and codes/last-trading dates of 64 quarterly contracts; target 2010–2025; local course ten-series cache may support separate pilot. Verify the underlying fields, date counts, roll semantics and entitlement before making empirical claims. Original 50-futures Pinnacle 1990–2020 sample is unavailable from author GitHub.

**NOT done:** verified LSTM/GPflow training environment; actual GP changepoint features, trained LSTM / LSTM+CPD, original Exhibit 3 replication, net trading results, 2021–2025 holdout, cost-aware neural training. Published Sharpe 1.62 and 2.16 are literature numbers, not team results.

**Internal timeline (not official due dates):**
- Oct 11–18: Enze confirms WRDS data/expiry data, multi-asset coverage; Jinrui audits assumptions.
- Oct 19–25: freeze causal features, split and cost grid; single-asset smoke pilot.
- Oct 26–Nov 15: Jinrui integrates GP-CPD and LSTM; Enze integrates futures data and cost backtest.
- Nov 16–29: joint OOS comparisons and CPD/cost/timing/seed robustness.
- Nov 30–Dec 6: interpret market changes; optional cost-aware neural experiment and additional holdout.
- Dec 7–11: joint report/presentation, paper Exhibit 3 / Appendix C comparison. CourseWorks dates supersede these internal targets.

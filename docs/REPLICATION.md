# Replication protocol and audited differences

## Paper identity and commitments

Primary: Wood, Giegerich, Roberts & Zohren (2021), arXiv:2112.08534. Core comparison: TSMOM, LSTM (63-day sequence), LSTM+CPD-21, decoder-only TFT (252-day sequence), TFT+CPD-21/126. The official primary repository already supplies these models; the supporting CPD repository is a second reference, not our main research question.

The paper uses 50 liquid Pinnacle CLC continuous futures, backward ratio adjusted, across commodities/equities/fixed income/FX, 1990–2020. Its main folds have train/validation [1990,1995), test [1995,2000), then expand training by five years through test [2015,2020). Report the 2020 crisis separately, Jan 1–Oct 15. The official Quandl example instead selects 100 CHRIS tickers with different coverage and defaults to annual testing beginning 2016; neither that example nor the ten-series local WRDS cache is exact Pinnacle replication.

Required project work: audit data and preprocessing; get TFT/LSTM/CPD into a reproducible environment; fit identical train/validation/test intervals; report net Sharpe, annual return/volatility, drawdown, turnover, cost sensitivity and regime diagnostics. Present attention/variable-selection patterns descriptively, without treating them as causal explanations.

## Critical code issues discovered before running

1. `examples/run_dmn_experiment.py` constructs intervals with `range(test_start, test_end - 1)` and a one-year stride even if `test_window_size=5`. Running it unchanged can create overlapping five-year tests and omit the final annual start. A dedicated adapter must call `mom_trans.backtest.run_all_windows` with an explicit interval list.
2. `mom_trans/classical_strategies.py` backfills warm-up volatility and MACD standard deviations. Our utilities leave warm-up values missing and test future-data invariance. Retain the original for provenance; document any corrected replication results separately.
3. `mom_trans/data_prep.py` uses an OR condition for filtering positive/nonmissing prices. Our input contract rejects undefined/infinite returns and flags nonpositive price observations for a contract-dollar treatment.
4. Upstream cost code initializes the first trade cost as zero. Our cost model charges initial entries and both roll legs, as well as volatility-driven position changes.
5. The paper's 90/10 training/validation split and the code's per-ticker row split must be reconciled. Our fold utility defines a documented calendar variant and purges labels crossing boundaries. Any scaler/embedding vocabulary/hyperparameter selection must be fit only on the training portion.

## Models, features and loss

Author features: normalized returns at 1/21/63/126/252 sessions; MACD pairs 8/24, 16/48, 32/96; 60-day EWM daily volatility; CPD normalized location/severity at 21 and 126 sessions. Inputs are trailing; labels are next-session returns. The authors winsorize price levels at EWM mean ±5 EWM standard deviations with 252-day half-life. Our startup feature export uses a compounded return index with no winsorization and no backward filling; this is deliberately recorded as a data/preprocessing variant, not an exact paper feature match. `label_date` is for purging only and must not enter X.

The author DMN loss is a negative annualized Sharpe objective over captured risk-scaled returns. Model selection uses validation only. A cost-aware portfolio loss is a later extension: include changes in actual volatility-scaled exposures with preceding-window positions carried across batches; do not charge unrelated shuffled sequence edges as trades. Resetting position history at every batch gives a different objective.

Paper-scale search is expensive (e.g. 50 trials for LSTM/TFT, 300 epochs, patience 25). First run a disclosed smoke profile (1 epoch/2 trials/one fold); then a predeclared reduced profile (10 trials/3 seeds) if compute is limited. Do not call these equal-compute paper reproduction. Freeze test windows before research and record all model variants. Paired moving-block bootstrap of daily model-return differences is planned after trained results exist; no significance claim from the startup pilot.

## Environment

Upstream pins TensorFlow 2.3.1, TensorFlow Probability 0.11.1, GPflow 2.2.1 and NumPy 1.18.5. Do not install these into the modern course or research environment. The full requirements include unrelated/conflicting-era pins, so even Python 3.8/Linux is not a verified working lock. Use an isolated Linux x86_64 environment for a compatibility audit, pin a resolved minimal training environment, test TFT/LSTM construction and one mini-batch, then run CPD. A modern PyTorch port is an optional engineering decision, not a completed faithful replication.

## Optional extensions (after core works)

1. Train with costs, compare to identical no-cost training, evaluate the same one-way 0/0.5/1/2/3/5-bps grid; 5 bps is stress, not a calibrated broker estimate.
2. Lock 2021–2025 as a post-2020 chronological holdout only after data selection and basic experiments; never tune on its results. The existing class cache ends 2025, so no 2026 coverage is claimed.
3. Use causal filtered HMM regimes to stratify failures or gate risk; all HMM fits and state matching must be done using past data only. Full-sample smoothing and Viterbi labels leak future information.
4. Read-only IBKR historical data and paper execution design. Commercial use of academic vendor data requires separate rights; a course subscription does not establish those rights.

# Completed WRDS replication: results and comparison

This report covers the completed 50-instrument PyTorch / SciPy-Numba experiment based on Wood, Roberts and Zohren, *Slow Momentum with Fast Reversion*. The original return/MACD feature functions are retained. The LSTM and GP fitting backends are ports; this experiment did not execute the original TensorFlow/GPflow pipeline. WRDS individual-contract histories replace the proprietary Pinnacle panel.

The execution manifest records **36 completed searches, 1,800 candidate fits and 126,763 training epochs**. Each search used 50 candidates, a 300-epoch cap and 25-epoch patience. Hyperparameters and CPD horizons were selected from pre-test validation observations. One search repetition was completed.

## Results

Mean five-window gross Sharpe was **1.815 for LSTM** and **1.602 for validation-selected CPD**. Exhibit 3 of the paper reports **1.62** and **2.16**, respectively. The gaps are +0.195 for LSTM and -0.558 for selected CPD. In this run, adding selected CPD reduced Sharpe by 0.212, approximately 11.7%, rather than recovering the paper's roughly one-third improvement. [Published study](https://arxiv.org/html/2105.13727#S4.SS2).

A higher LSTM Sharpe does not mean a higher return: its mean annual return was 2.71%, versus the paper's 3.53%, while volatility was lower at 1.55%, versus 2.52%. Selected CPD returned 3.02% at 1.88% volatility. Its higher return relative to LSTM was accompanied by enough additional risk to reduce Sharpe.

Not every CPD specification underperformed. Fixed 252-session CPD achieved 1.927 Sharpe and fixed 21-session CPD achieved 1.850. The optimized result nevertheless follows validation selection, not the best observed test Sharpe. Its main-fold lookbacks were 10, 21, 63, 63 and 252 sessions. Their weaker subsequent test performance indicates that validation rankings did not reliably carry forward in this realization. Selecting the 252-session result after observing the test outcome would be hindsight selection.

See [the complete comparison](tables/replication/published_comparison.csv), [per-fold results](tables/replication/exhibits_by_fold.csv) and the [executed notebook](Paper67_Original_Study_Replication.ipynb).

## Costs and interpretation

At 2.5 one-way basis points of traded notional, original-style turnover-adjusted Sharpe was **0.302 for LSTM**, **-0.377 for selected CPD** and approximately **0.940 for annual momentum**. Using un-winsorized realized returns and charging both roll legs gave **0.257, -0.418 and 0.860**, respectively. Frequent reversals can therefore consume much of the gross performance in these scenarios.

These are cost sensitivities, not estimates of actual spreads, market impact or capacity. Raw-return and roll-accounting extensions are separate from the paper-style gross comparison. They cannot, by themselves, explain a difference between gross Sharpe figures. [Cost results](tables/replication/transaction_costs.csv); [roll/execution results](tables/extension/roll_costs.csv).

## Sources of differences

**Data construction and coverage.** WRDS histories use an explicit calendar-roll rule and same-contract returns. Sessions, composite series, start dates and discontinued histories differ from Pinnacle. Avoiding cross-contract price gaps does not reconstruct the unavailable vendor series. Return paths, features and the eligible asset pool can all change. [Coverage](tables/replication/data_coverage.csv) and [data decisions](../docs/DATA_DECISIONS.md) record these choices.

**Implementation.** PyTorch training is not bitwise equivalent to Keras. The state-space GP likelihood is algebraically equivalent to dense covariance evaluation, but the port uses numerical gradients, parameter bounds and stabilization that can affect optimization. Likelihood-equivalence tests do not prove identical fitted changepoints, weights or traded positions.

**Random search.** Matching 50 candidates does not recover the authors' unpublished draws, weights or random states. One search repetition cannot distinguish ordinary training variation from systematic implementation or data effects. Bootstrap intervals conditional on fitted positions do not measure retraining uncertainty.

**Dates and aggregation.** This completed version averages 1995-1999, 2000-2004, 2005-2009, 2010-2014 and 2015-2019. A separately refitted 2020 test is excluded from those averages. The paper describes 1995-2020, so the endpoint is not automatically identical. Exhibit 4 uses disclosed ex-post mean-window volatility rescaling, not an investable risk forecast.

These factors change together. The gaps cannot be assigned to the data vendor, backend or training randomness in isolation. The completed run **does not recover the paper's central improvement from validation-selected CPD**. It is evidence about this WRDS implementation, not a refutation of the original experiment.

## Computational scope

The 346,218-row feature panel produces **1,731,090 asset-date/window CPD outputs** across five horizons. Each requires stationary and changepoint GP fitting with iterative likelihood/gradient evaluations and a 200-iteration limit. Dense GP evaluation repeatedly factorizes covariance matrices, with rapidly increasing cost as the lookback grows. The paper also notes greater CPD cost at 126 and 252 sessions. [Study discussion](https://arxiv.org/html/2105.13727#S4.SS3).

The neural stage then performs the 1,800 candidate fits recorded above. GPUs accelerate LSTM training but do not eliminate CPU optimization and model-selection work. Repeated-seed comparisons multiply this workload. Per-fit elapsed fields can include pauses and scheduling, so they are not a clean hardware timing benchmark.

The full configured search budget was completed for this port. A controlled comparison using the original framework, matched data and repeated seeds requires additional computation. Exact numerical recovery also requires original data and execution details; more compute alone cannot supply them. Runtime cost is a practical limitation, not evidence that a discrepancy is acceptable or that its cause is known.

## Evidence and layout

[execution_manifest.json](execution_manifest.json) records configuration, completion counts and input hashes. [experiment_inventory.csv](tables/replication/experiment_inventory.csv) records individual search completions. [result_artifacts_manifest.json](result_artifacts_manifest.json) hashes the committed aggregate artifacts. Published reference values were not inserted into generated returns.

Reusable logic is under src/momentum_lab/, ordered entry points under scripts/, and parameters under configs/. Final notebooks, reports, tables and figures are under report/. Licensed observations, fitted weights, logs, credentials and local environments remain outside Git. See [the repository hierarchy](../resources/repo_hierarchy.txt).

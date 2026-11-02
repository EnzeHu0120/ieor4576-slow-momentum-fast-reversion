# Five-asset results and comparison with the paper

This experiment uses SP (S&P 500 day-session futures), TY (US 10-year Treasury notes), BN (British-pound futures), ZG (gold) and ZU (crude oil). These instruments were selected for asset-class coverage and long histories, before reviewing strategy performance. The run completed all **30 model/window searches and 1,500 candidate fits**, with 50 candidates per search. All **25 CPD instrument/window jobs** completed, producing **192,765 daily outputs**.

## Gross results

Mean test-window Sharpe was **0.589 for LSTM** and **0.518 for validation-selected CPD-LSTM**. The paper reports **1.62 and 2.16**, respectively, for its 50-instrument Pinnacle portfolio. The absolute gaps are -1.031 and -1.642. Within this five-asset sample, adding selected CPD reduced Sharpe by approximately **12%**, whereas the published improvement is about one-third. [Paper, Exhibit 3](https://arxiv.org/html/2105.13727#S4.SS2).

LSTM's mean annual return was 3.24% at 4.56% volatility. Selected CPD returned 2.13% at 4.61% volatility. The long-only baseline achieved 0.768 Sharpe and MACD achieved 0.649, both above the two neural strategies in this sample. These are volatility-scaled futures strategies, not an unlevered S&P 500 buy-and-hold comparison.

Fixed 126-day CPD achieved the highest neural Sharpe, **0.629**, modestly above LSTM. It is not substituted for the validation-selected result: doing so after observing test performance would introduce hindsight. The pre-test selection chose 63, 63, 126, 126 and 21 days in successive folds. The selected strategy's weaker test result indicates that the validation ordering did not reliably carry into the next test period in this realization.

The [complete comparison](paper_reference_comparison.csv) reports every model and classical benchmark. [Per-fold metrics](tables/replication/exhibits_by_fold.csv) and [selected windows](tables/replication/selected_cpd_windows.csv) allow the aggregate result to be inspected by period.

## Costs

At 2.5 one-way basis points, the author cost convention gives mean Sharpe **0.309 for LSTM** and **0.091 for selected CPD**. Evaluating raw same-contract returns and charging both roll legs gives **0.262 and 0.045**. These sensitivities reduce the already limited gross advantage; they do not estimate real spreads, market impact or capacity. Gross comparisons and cost extensions are separate outputs, and cost deductions do not explain a gap between two gross Sharpe figures.

## Differences affecting the outputs

**Universe and portfolio construction.** Five instruments replace the paper's 50-instrument, multi-asset portfolio. This changes the amount and diversity of pooled training data, cross-asset dependence and portfolio diversification. It is the largest deliberate design difference. The published metrics are a reference, not a matched-universe control.

**Price histories and rolls.** The source is WRDS individual-contract history rather than Pinnacle CLC continuous futures. Returns are computed within the held contract and compounded into a price index, using an explicit five-business-day calendar roll. Session definitions, vendor revisions, missing observations and continuous-series adjustments may differ. The near/far settlement gap is not counted as earned return. The exact original Pinnacle roll history is unavailable.

**Coverage and eligible observations.** Historical coverage and warmup periods determine the rows available for each instrument and fold. Contiguous next-observation labels and the paper's first-test eligibility are retained. Missing histories are not fabricated. Coverage and vendor mappings are listed in [data_panel_audit.csv](tables/replication/data_panel_audit.csv).

**Environment and runtime adaptations.** The core Keras LSTM, ModelFeatures formatter, Sharpe callbacks, GPflow kernels and fitting functions come from the pinned author repository. Modern dependency APIs require explicit compatibility edits. Execution used WSL2, Python 3.11, legacy Keras/TensorFlow 2.21 and GPflow 2.10, including a Blackwell-compatible community TensorFlow build. It is not the paper's historical binary environment. GPflow model/optimizer graphs are reused, with author parameters reset at every date; no warm starts, reduced iteration cap or approximate likelihood are introduced. Fifteen real windows across all five horizons matched the unaccelerated fitting functions within 2.81e-10 absolute error. The test covers those cases, not proof of identical results for every possible input.

**Search realization.** TensorFlow and tuner seeds are explicitly set to 67 plus the test-start year. The published hyperparameter grid, 50-candidate budget, 300-epoch limit and 25-epoch patience are retained. The paper's exact candidate draws and weights are unavailable. One completed search realization does not establish robustness to retraining; no repeated-seed significance claim is made.

**Dates and aggregation.** Main results average five folds: 1995-1999, 2000-2004, 2005-2009, 2010-2014 and **2015-2020 inclusive**. The final fold spans six calendar years to include 2020 without an additional model refit. The paper's prose does not uniquely disambiguate that endpoint. A [strict 1995-2019 sensitivity](tables/replication/strict_1995_2019_sensitivity.csv) is also saved. Exhibit 4 applies the disclosed ex-post 15% mean-window volatility scaling and recomputes metrics; it is not an implementable volatility forecast.

These factors change the inputs and experimental realization. Their separate contributions cannot be identified from this run alone. The result does not recover the paper's central CPD improvement on these five instruments; it neither refutes the 50-instrument study nor establishes a particular implementation error.

## Computational scope and evidence

Daily CPD entails repeated stationary and changepoint GP optimization for five lookbacks. Longer windows require more expensive covariance calculations. The 1,500 LSTM fits then repeat training and validation for all five periods and six model families. CPU optimization, GPU training and repeated-seed checks all consume time; a GPU does not eliminate the full pipeline cost. Runtime logs include interruptions, so no clean active-compute benchmark is claimed. Additional computation cannot replace missing original data or execution details.

Completion is recorded in [execution_manifest.json](execution_manifest.json). [Search inventory](tables/replication/experiment_inventory.csv), [candidate records](tables/replication/candidate_search_inventory.csv), [training dates](training_context.json) and [artifact hashes](result_artifacts_manifest.json) provide supporting evidence. Only aggregate outputs are distributed; licensed observations, checkpoints, credentials and local machine state are excluded.

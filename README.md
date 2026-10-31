# Phase 1 — Slow Momentum with Fast Reversion

Enze Hu · Jinrui Hu · IEOR 4576

Wood, Roberts & Zohren (2022), [*Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection*](https://arxiv.org/abs/2105.13727). This branch implements the original LSTM / GP-CPD study. No Transformer model is substituted.

## Run on a new computer

Detailed NVIDIA/Windows instructions: [docs/GPU_RUN.md](docs/GPU_RUN.md).

```bash
git clone --branch enze --single-branch https://github.com/EnzeHu0120/ieor4576-slow-momentum-fast-reversion.git
cd ieor4576-slow-momentum-fast-reversion
python3.12 -m venv .venv-replication
source .venv-replication/bin/activate
python -m pip install -e '.[dev]'
python -m ipykernel install --user --name paper67-replication --display-name 'Paper67 Replication (Python 3.12)'
python scripts/check_environment.py
```

For NVIDIA hardware, install the appropriate CUDA-enabled PyTorch build using the [official installer](https://pytorch.org/get-started/locally/). Confirm `cuda_available: true` before a GPU run. Device selection is automatic: CUDA, Metal, then CPU.

Configure **your own authorized WRDS account** using `WRDS_USERNAME` and libpq's secure password authentication, or a process-only `WRDS_PASSWORD`. The account needs access to WRDS LSEG/Datastream `trdstrm` tables. Credentials and licensed data are never included in Git. Cloning code cannot grant data-service access.

Open **[report/Paper67_Original_Study_Replication.ipynb](report/Paper67_Original_Study_Replication.ipynb)**, select the environment above, and **Run All**. Its training cell displays live progress and per-epoch training/validation curves. Computation is implemented in reusable Python modules; no assistant session or model API is needed. Save the notebook after execution.

The default is `RUN_REPRODUCTION=True`, `FRESH_MODELS=True`: old model checkpoints are archived and every candidate starts from random initialization. No prior progress notebook or fitted model is required. Prepared raw data and causal GP feature caches may be reused. Missing data/features are acquired/computed automatically. For an interrupted run, change `FRESH_MODELS=False` to resume that run. To only display completed results, set `RUN_REPRODUCTION=False`.

One-command fresh run, including automatic execution/saving of the final notebook:

```bash
make fresh
```

Without Make, use `python scripts/run_all.py --fresh-models`. `make reproduce` rebuilds from prepared raw data and reuses compatible model checkpoints; `make resume` also reuses prepared features. Do not run notebook and terminal training simultaneously. Logs: `results/replication/logs/local_notebook_run.log`. A new machine without GP caches must also complete expensive CPU-based changepoint feature construction.

## Reproduction scope and interpretation

- Appendix A's 50 mapped futures, targeting 1990–2020 through WRDS individual-contract histories. Historical coverage, substitutions and rolls are disclosed in `data_coverage.csv` and [DATA_DECISIONS.md](docs/DATA_DECISIONS.md).
- Original author return/MACD features; stationary versus changepoint Matérn GP features at 10/21/63/126/252 sessions; stateless 63-step LSTM with Sharpe loss. Preserved MIT source and provenance: `vendor/` and `UPSTREAM.json`.
- Fifty distinct candidates from the full published grid for each of six model families and six test boundaries: 36 searches, 1,800 candidate fits, at most 300 epochs and patience 25. All tuning uses pre-test validation data.
- Five main OOS windows, 1995–1999 through 2015–2019; a separately fitted 2020 test. Exhibit 4 uses the disclosed author-style ex-post volatility rescaling convention.
- Original-style transaction-cost sensitivity, including 2.5 one-way bps. Subsequent raw-return / both-roll-leg / execution-delay / exposure-cap diagnostics address graded-proposal feedback. They do not alter the original gross experiment or introduce a new model.

This is a WRDS adaptation of the proprietary Pinnacle study. Source histories, sessions, rolls, random-search realization and numerical backends differ. Published LSTM/optimized-CPD Sharpe values **1.62 / 2.16 are reference values, not generated results**. Repeated-seed inference, measured market impact, integer-lot capacity and a USD conversion overlay are not claimed.

## Structure and outputs

The repository follows the supplied `repo_hierarchy.txt`: `src/momentum_lab/` owns research logic; `scripts/01_get_data.py` through `05_report.py` are entry points; `configs/` holds the universe, parameters and published comparison values. `notebooks/` is reserved for exploration; the final research notebook is in `report/`. Runtime folders are created automatically and excluded from Git.

| Output after completion | Location |
|---|---|
| Executed notebook and HTML | `report/Paper67_Original_Study_Replication.*` |
| Exhibits 3/4 and comparison | `results/replication/tables/` |
| Replication figures | `results/replication/figures/` |
| Costs and execution diagnostics | `results/replication/tables/transaction_costs.csv`, `results/extension/` |
| Actual experiment inventory and hashes | `report/execution_manifest.json` |
| Written report and presentation | `report/Paper67_Replication_Report.pdf`, `report/Research_Presentation.html` |

`make test` runs essential GP-equivalence, chronology, loss, futures-roll and cost checks. [REPLICATION.md](docs/REPLICATION.md) specifies the numerical protocol. [STATUS.md](docs/STATUS.md) distinguishes implemented code from completed experiments. The fresh full study is not represented as completed before it has run.

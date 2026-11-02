# Slow Momentum with Fast Reversion: a WRDS 24-series method replication

Group 67: Enze Hu (eh3203), Jinrui Hu (jh5260).

Paper: Kieran Wood, Stephen Roberts and Stefan Zohren, *Slow Momentum with Fast
Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection*,
arXiv:2105.13727v3 (2022), https://arxiv.org/abs/2105.13727.

This project adapts the author's LSTM/GP changepoint approach to a fixed WRDS
Datastream panel of 24 provisional distinct underlying markets. It is a partial
method replication, not a numerical reproduction of the original 50 Pinnacle
futures, 1990-2020 expanding-window experiment.

## Main result

On 474 retained common signal/return intervals in 2019-2020, the validation-selected
252-observation CPD model has gross Sharpe **1.8803**, versus ordinary LSTM
**-0.3484** and long only **1.6823**. At **1 basis point one-way** turnover cost,
the CPD model falls to **1.4995**, below long only **1.6638**. These are conditional
comparison statistics, not full-calendar executable portfolio results.
The result supports a gross improvement in this run, with substantial cost
sensitivity. It does not establish universal or statistically significant CPD
superiority. The 21-observation CPD model has negative test Sharpe.

An exploratory development comparison also motivates a universe-size hypothesis:
the earlier single-sequence pilot had CPD/ordinary Sharpe 1.0469/1.1205, whereas
Enze's five-asset expanding-window study reports 0.5182/0.5889, and the 24-market
panel has 1.8803/-0.3484. These heterogeneous outcomes strengthen the motivation
to study market composition, diversification and pooled learning. They are
**not a controlled market-count experiment**: data construction, dates, aggregation,
candidate draws and optimization exposure differ. The five-asset means cover
1995-2020, while the 24-market result uses 2019-2020. Gold and crude oil in the
five-asset study are absent from the 24-market pool. Higher Sharpe or a greater
chance of ranking first as markets are added remains a hypothesis. Follow-up
subset/seed/retraining experiments are proposed, not performed.

The companion five-asset implementation is pinned at team repository commit
`f88874663ba3641f06c867c45446b1c4e5faf93e`. Its committed aggregate evidence
is included under `report/evidence/enze_five_assets/`. Its full private-data
estimation is not embedded or independently rerun by this 24-market package.

## Rebuild the verified report without retraining

For the private course ZIP, start with `START_HERE.md`. The three archives and
the fixed input snapshot are included; no old author checkout is required.
The external import instructions below apply only to the smaller code-only package.

Keep these three exact experiment archives together in a directory:

- `paper67_panel24_lstm_results.zip`
- `paper67_panel24_cpd_results.zip`
- `paper67_panel24_supplement_results.zip`

Use the existing WSL Python environment, or install base reporting dependencies
with `python -m pip install -e .`. Python 3.10 was used for the empirical GPU
experiment. Reporting was additionally verified under Python 3.12.

```bash
source ~/paper67-author/.venv/bin/activate
cd ~/group67-wrds24
python run_all.py --bundle-dir /mnt/c/Users/sunji/Downloads
```

The default command imports the three private archives into ignored `data/`
directories, checks their SHA-256 values, verifies 300 trial records, reconstructs
sampled GP marginal likelihoods, recomputes gross/scaled/cost tables, and rebuilds
figures. It performs **no model training and no GP fitting**. Expected marker:
`REPORT_REBUILD_OK`.

Once imported, `python run_all.py` repeats the same report rebuild. A checkout
without the private run archives requires the authorized data/artifact access
steps in [DATA_ACCESS.md](DATA_ACCESS.md).

The already-rendered English PDF is `report/phase1_results.pdf`; its editable
source is `report/phase1_results.md`. To rebuild only the PDF, install the optional
typesetting dependencies with `python -m pip install -e ".[pdf]"`, provide the
DejaVu Sans fonts, and run `python scripts/05_build_report.py`. This does not
train models. The PDF and cross-study comparison were checked in this revision;
the prior full empirical reference audit is preserved in VALIDATION.json.

Run the small sanity suite without installing a test framework:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Optional full re-estimation

This is expensive and is **not required to inspect or rebuild the completed
reference results**. It runs the preserved scientific scripts in a separate
generated runtime, leaving the old `~/paper67-author` directory untouched.

Use Python 3.10 and the empirical dependency pins in the `reproduce` extra
(`python -m pip install -e ".[reproduce,dev]"`). GPU setup must also provide a
compatible Windows NVIDIA driver, WSL2 GPU support and the CUDA libraries.

```bash
python run_all.py --stage full --workers 8
```

This imports the immutable selected raw input snapshot, builds segmented causal
features, trains 50 ordinary LSTMs, computes all five GP lookbacks, trains 250
CPD-LSTMs, and runs saved-checkpoint analysis. The isolated runtime is
`data/interim/author_runtime/`. A successful new estimation produces three new
archives under `results/new_experiment/archives/`; it does not replace the
hash-pinned reference experiment. Training can vary across hardware/software.
The reorganized wrapper's entire expensive GPU estimation was not rerun during
packaging; its underlying scripts are the ones used for the audited run.

## Structure and result map

| Location | Purpose |
|---|---|
| `configs/config.yaml` | Seed, windows, search budget and fixed split; JSON-compatible YAML |
| `data/raw/`, `data/interim/`, `data/processed/` | Untouched imported archives, derived snapshots/runtime and final feature tables; ignored by git |
| `src/project/data.py` | Private archive/raw input import and verification |
| `src/project/features.py` | Feature/GP orchestration in the isolated runtime |
| `src/project/models.py` | Training entry points and validation/history checks |
| `src/project/evaluation.py`, `audit.py`, `analysis_math.py` | Independent checks and performance/cost statistics |
| `src/project/plotting.py` | Figures |
| `src/project/cross_study.py` | Checks the companion five-asset fold means and exports descriptive cross-study context |
| `src/project/author_code/` | Preserved author-compatible scientific implementation and license |
| `scripts/01_get_data.py` through `04_extend.py` | Thin entry points |
| `results/replication/tables/gross_fixed_lookbacks.csv` | Gross five-window and baseline comparison |
| `results/replication/tables/ex_post_15pct_metrics.csv` | Ex-post constant 15% risk standardization |
| `results/replication/tables/cost_sensitivity_0_to_5bps.csv` | Conditional one-way cost scenarios |
| `results/replication/tables/independent_audit.json` | Full audit findings and error bounds |
| `results/replication/figures/` | Cumulative and cost sensitivity plots |
| `report/phase1_results.md` | English results write-up and material limitations |
| `report/phase1_results.pdf` | Typeset English report for review/submission |
| `report/evidence/pilot_panel_observation.json` | Provenance for the confounded single-series versus 24-market observation |
| `HIERARCHY_CHECK.md` | Mapping to the previously supplied `repo_hierarchy.txt` |

`data/`, `results/`, virtual environments, credentials and caches are excluded
from git. The handoff ZIP includes generated summary tables/figures for
convenience; git still excludes them. Do not publish private WRDS files or large
experiment archives without applicable data-sharing rights. Compact tables in
the main report keep its findings readable without the ignored output directory.

`04_extend.py` rebuilds post-training diagnostics, including cost sensitivity.
These analyses also appear in the original paper; they are not claimed as a
novel research extension. No separate 2021-2025 extension backtest has been
estimated. The extension output folders are reserved.

## Scope and timing

- Training: 2010-2016 after per-market warm-up; validation: 2017-2018; test:
  2019-2020. Raw 2021-2025 observations are reserved and excluded from features.
- LSTM sequence length: 63; CPD lookbacks: 10/21/63/126/252. There are 50
  hyperparameter trials per variant, at most 300 epochs, patience 25, one
  search seed (67), and matched hyperparameter candidates across variants.
- Checkpoint and structural-window selection uses validation. Do not change
  the final model after inspecting supplementary test performance.
- The project developed through exploratory pilots. Do not describe it as
  preregistered or its full research process as free of test reuse.
- The 15% table uses a full-test constant volatility multiplier for presentation,
  not a causal trading overlay.

See [THIRD_PARTY.md](THIRD_PARTY.md) for author provenance and compatibility edits.
Section 8 of [the report](report/phase1_results.md) explicitly registers material
differences from the paper, retained choices and unverified equivalence details;
Sections 9-10 cover the universe-size hypothesis and proposed future work.
Instructor repository access and the actual course submission must still be
completed by the team; this package sends no invitations and submits no work.

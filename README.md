# Slow Momentum with Fast Reversion: completed WRDS replication

Enze Hu and Jinrui Hu | IEOR 4576

This branch contains the completed **PyTorch / SciPy-Numba port** of Wood, Roberts and Zohren (2022), [arXiv:2105.13727v3](https://arxiv.org/abs/2105.13727v3). It uses WRDS contract histories rather than Pinnacle. Author feature functions are preserved; the neural and GP fitting backends are reimplemented. It is not an exact original-framework numerical reproduction.

## Results and evidence

All **36 model/window searches and 1,800 candidate fits** completed, totaling 126,763 training epochs. Mean five-window gross Sharpe is **1.815 for LSTM** and **1.602 for validation-selected CPD**, versus the paper's **1.62 and 2.16**. The run does not recover the published benefit from validation-selected CPD. Some fixed CPD horizons perform better, but they are not substituted retrospectively for validation selection.

- [Executed final notebook](report/Paper67_Original_Study_Replication.ipynb)
- [Results, differences, trading costs and computational scope](report/RESULTS.md)
- [Full published/result comparison](report/tables/replication/published_comparison.csv)
- [Execution manifest](report/execution_manifest.json) and [search inventory](report/tables/replication/experiment_inventory.csv)
- [Original generated report](report/Paper67_Replication_Report.pdf) and [presentation](report/Research_Presentation.html)

The main exhibits average 1995-2019 over five expanding-window tests; 2020 is reported separately. Data construction, backend arithmetic, random-search realization and the evaluation endpoint can affect comparisons. These are possible explanations, not an identified causal decomposition.

## View or rerun

The final notebook defaults to **displaying committed results**. Run All reads report/tables/ and report/figures/ without training, licensed data or WRDS authentication.

For a new experiment, create a Python environment from pyproject.toml and install the suitable PyTorch build. See [GPU_RUN.md](docs/GPU_RUN.md). Configure an authorized WRDS account using environment variables and secure local authentication. Missing data and features require acquisition/computation. In the notebook, set RUN_REPRODUCTION=True; FRESH_MODELS controls whether fitted checkpoints are reused. Alternatively use make fresh or make reproduce. Full retraining is computationally substantial.

Credentials, licensed observations, environments and model weights are not published. A clone provides code and aggregate results, not access to the data service. Paper reference metrics remain separate in configs/published_exhibits.csv.

## Repository hierarchy

The final submission is under report/, with aggregate replication/extension tables and figures in its subdirectories. Reusable acquisition, modelling and evaluation code lives under src/momentum_lab/. scripts/01_get_data.py through 05_report.py are ordered entry points; configs/ contains the universe and experiment parameters. notebooks/ is for exploration. Runtime data/ and results/ are Git-ignored.

[Hierarchy](resources/repo_hierarchy.txt) | [Protocol](docs/REPLICATION.md) | [Data decisions](docs/DATA_DECISIONS.md) | [Completion status](docs/STATUS.md) | [References](resources/REFERENCES.md)

The paper's MIT-licensed source and provenance remain under vendor/ and UPSTREAM.json. make test checks GP likelihood equivalence, chronology, Sharpe loss, futures rolls and costs. This submission contains only the completed ported experiment; separate ongoing author-framework runs are not represented as completed results.

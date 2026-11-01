# Run Phase 1 on another GPU computer

Clone the `enze` branch (repository access is required while it is private):

```bash
git clone --branch enze --single-branch https://github.com/EnzeHu0120/ieor4576-slow-momentum-fast-reversion.git
cd ieor4576-slow-momentum-fast-reversion
python3.12 -m venv .venv-replication
source .venv-replication/bin/activate
python -m pip install -e '.[dev]'
python -m ipykernel install --user --name paper67-replication --display-name 'Paper67 Replication (Python 3.12)'
```

On Windows, create the environment with `py -3.12 -m venv .venv-replication` and activate `.venv-replication\Scripts\Activate.ps1` in PowerShell. The core pipeline uses a portable file lock and subprocess entry point.

Install a CUDA-compatible PyTorch build for the computer's NVIDIA driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then verify:

```bash
python scripts/check_environment.py
```

For an NVIDIA training machine, `cuda_available` should be true. The notebook also prints the selected device before fitting; the configuration uses `device: auto` (CUDA first, then Metal, then CPU).

Open `report/Paper67_Original_Study_Replication.ipynb` and select **Paper67 Replication (Python 3.12)**. Keep:

```python
RUN_REPRODUCTION = True
FRESH_MODELS = True
MODEL_WORKERS = 3
```

Choose **Run All**. The training cell displays live training/validation curves and candidate completion counts; model code remains in `src/momentum_lab`. Fresh mode preserves older checkpoints in `results/archives/` and starts every LSTM / CPD-LSTM candidate from random initialization. No old progress notebook is loaded. Save the notebook after completion. If this run is interrupted, set `FRESH_MODELS=False` before running again to resume this run rather than start over.

The equivalent terminal command is `python scripts/run_all.py --fresh-models`; it also executes and saves the final notebook. Do not start notebook and terminal fitting simultaneously.

## Data are separate from source code

The private repository still excludes `data/`, `results/`, credentials and local environments. Choose one of these routes:

- Configure your own WRDS account (`WRDS_USERNAME` plus secure libpq password authentication). Missing licensed raw data and all features are then acquired/computed by Run All. The account must be entitled to the WRDS `trdstrm` tables.
- Transfer the prepared **`data/` folder only** between your own authorized computers over a private channel, keeping the same project-relative paths. This can reuse the expensive causal GP feature construction while **all model weights still start from zero**. Do not include passwords, `.pgpass`, environments or old `results/` checkpoints. Vendor-date vintages can change if data are downloaded again.

GP-CPD feature construction uses CPU and can take much longer than model fitting when caches are absent. A stronger GPU accelerates the LSTM searches; it does not eliminate first-run GP preprocessing. No fixed runtime or exact Pinnacle Sharpe is promised.

## Phase 1 completion evidence

The full study requires 36 completed model/window searches, 50 fitted candidates each (1,800 total), and all five GP feature panels. The run generates actual Exhibits 3/4, transaction-cost tables, figures, comparison and report. `report/execution_manifest.json` records actual completion counts and data/configuration hashes. This branch contains the implementation and runnable notebook, not a claim that the fresh full experiment has already completed.


Submission default: the final notebook displays committed aggregate results without training. Set RUN_REPRODUCTION=True explicitly to start a new data/model run. The submitted completed results are a PyTorch / SciPy-Numba port.

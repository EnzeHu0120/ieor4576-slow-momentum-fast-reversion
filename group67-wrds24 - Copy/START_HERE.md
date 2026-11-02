# Group 67: private course ZIP submission

Enze Hu (eh3203) and Jinrui Hu (jh5260) | IEOR 4576

Read `report/phase1_results.pdf` for the completed 24-market partial method
replication and the descriptive comparison with the five-asset study.

## Install and rebuild results (no model training)

Extract the ZIP, open a terminal in `group67-wrds24`, and use Python 3.10-3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python run_all.py
python -m unittest discover -s tests -v
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Expected completion marker: `REPORT_REBUILD_OK`. This checks the three original
archives, 300 trial records, GP reconstruction samples, returns and cost arithmetic,
and regenerates tables and figures. It does not refit the models. Runtime depends
on the CPU. The included PDF can be read immediately without running anything.

## Rebuild features or retrain (WSL2/Linux, Python 3.10)

```bash
python -m pip install -e ".[reproduce,dev]"
python run_all.py --stage prepare
python run_all.py --stage full --workers 8
```

`prepare` rebuilds baseline features from the included fixed input snapshot.
`full` prepares features, trains ordinary LSTM, fits five GP lookbacks, trains
CPD variants, and exports new experiment results. It can take many hours and
requires the documented TensorFlow/CUDA GPU setup. Full training has not been
rerun for this packaging revision. New fits need not equal the saved results.
The original `~/paper67-author` directory is not required.

## Included inputs and scope

- `data/raw/artifacts/`: three unchanged saved experiment ZIPs with checked hashes.
- `data/interim/universe_snapshot/wrds_panel24/`: the fixed selected 24-series
  input snapshot and universe metadata; hashes in `configs/input_snapshot_sha256.json`.
- `results/replication/`: generated tables and figures.
- `report/`: PDF, editable report and source evidence.

The snapshot is derived from a WRDS download; it is not the original expanded
50-series query. Rebuilding its historical universe-selection decision from the
original query is not automated here. Feature preparation, segmentation and model
estimation begin from the fixed snapshot. No new 2021-2025 extension was performed.
The five-asset aggregate evidence is included, but its complete training project
is maintained separately in the team repository linked in README.md.

This package contains licensed inputs and saved model artifacts for authorized
course review. Do not publish it as a public GitHub repository. The code-only
package excludes these inputs; the private ZIP deliberately includes them.

## Submission

Upload this ZIP to Checkpoint 1 / Code Submission. Check the upload size limit
and confirm the platform records a successful submission. No GitHub push is
needed for the ZIP route. This archive does not itself submit work or grant access.

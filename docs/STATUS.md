# Phase 1 status

The real WRDS data pipeline, original feature construction, exact daily GP-CPD port, LSTM / CPD-LSTM searches, expanding-window backtest, evaluation, costs and report generators are implemented. The repository follows `repo_hierarchy.txt`.

The notebook is prepared for a **fresh run on a new GPU computer**. It defaults to `FRESH_MODELS=True`, shows live per-epoch training/validation curves under the training cell, and archives rather than loads older fitted checkpoints. It does not read the older progress notebook. That snapshot, local credentials, data, weights and machine-specific state are excluded from this branch.

Prior partial experiments and the prepared 50-instrument data remain local. The new fresh experiment has not yet run to completion; no final result or published Sharpe match is claimed. Completed fresh experiments will be evidenced by `report/execution_manifest.json` (36 searches and 1,800 candidates), not by this status document.

See [GPU_RUN.md](GPU_RUN.md) for a new machine. All missing licensed data and features can be acquired/computed after environment setup and WRDS authorization. CPU/Metal/CUDA device selection is automatic. The submission includes essential numerical/chronology/cost checks; the new-machine CUDA hardware and the entire fresh study have not been tested here.

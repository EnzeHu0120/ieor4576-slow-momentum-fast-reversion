# Momentum Transformer — IEOR E4576 project #67

Shared starting point for **Wood, Giegerich, Roberts & Zohren (2021), Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture**, [arXiv:2112.08534](https://arxiv.org/abs/2112.08534).

The separate *Slow Momentum with Fast Reversion* paper ([2105.13727](https://arxiv.org/abs/2105.13727), published 2022) supplies CPD/LSTM context. It is **not** catalog #67. The catalog entry has been checked directly in the supplied course list.

## Start here

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
make demo
python scripts/01_get_data.py                  # preview only; does not connect
```

On Enze's current Mac the ready environment is `../.venv/bin/python`; use `make demo PYTHON=../.venv/bin/python`. The Mac lock file records the actual Python 3.14 environment. This modern environment runs our research utilities, **not** the legacy TensorFlow models.

## What is present

- `vendor/trading-momentum-transformer/`: complete, unchanged author source at a pinned commit, including TFT, LSTM and GP-CPD; MIT license retained.
- `vendor/slow-momentum-fast-reversion/`: complete, unchanged supporting author source; MIT license retained.
- `src/momentum_lab/`: compact, commented data checks, trailing features, chronological folds, volatility-scaled exposure simulation, roll-aware costs and risk metrics.
- `scripts/`: data-selection preview, feature export, official replication status, end-to-end baseline simulation, local course-cache importer.
- `configs/`: explicit research parameters; **WRDS universe and dates are still a team decision**.
- `docs/`: replication protocol, data access, collaboration, resource map and remaining work.
- `resources/local/` (ignored): papers, course requirements and prior proposal available locally. Links/provenance are shared; licensed data and classroom files are not uploaded.

## Current evidence and limits

The portable synthetic demo and causal/cost sanity checks have run successfully. Synthetic returns are labelled everywhere. The local `Enze` workspace additionally owns the course-cache pilot, personal materials and optional extensions. No TFT/LSTM training or paper-level reproduction result is claimed at project initialization. See [STATUS.md](docs/STATUS.md).

The baseline uses 15% per-asset risk scaling divided by the fixed asset count; this is not a 15% portfolio volatility guarantee. Default execution lag=2 uses close-t information for a later close-to-close return, with a cap of 2/N notional per asset. Settlement remains a valuation proxy. Fee scenarios are one-way bps of traded notional, including entry, exit, exposure changes and roll legs. Cash interest, margin, integer contracts and terminal liquidation are not yet modeled.

## Shared versus personal

`Project/main` is branch `main`; `Project/Enze` is a linked worktree on branch `enze`. Work in the correct folder. Only `main` is authorized for GitHub upload now. Do not push all branches. Do not edit author snapshots in place; implement adapters in our own package and record differences.

Read [COLLABORATION.md](docs/COLLABORATION.md) before syncing. The teammate account and final data choice remain pending. Raw data under `Data/`, credentials and personal resources stay outside Git regardless of repository visibility. The interactive WRDS downloader is maintained only in the local Enze workspace, as requested.

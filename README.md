# Slow Momentum with Fast Reversion — Project #67

**Correct selected paper:** Wood, Roberts & Zohren (2022), [Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection](https://arxiv.org/abs/2105.13727). Group: Enze Hu (eh3203), Jinrui Hu (jh5260). 

## Research goal

Evaluate whether adding **online Gaussian-process changepoint detection (CPD)** to a 63-step LSTM Deep Momentum Network improves **out-of-sample net Sharpe** after transaction costs. Compare classical 252-day TSMOM, LSTM without CPD, and LSTM+CPD on identical futures data, test dates, volatility scaling, trading timing and cost assumptions. Core replication: Exhibit 3; core extension: Appendix C cost sensitivity (0, 1, 2, 3, 5 one-way bps). Cost-aware LSTM+CPD training is conditional on a working replication.

## Data

The original paper used 50 Pinnacle CLC continuous futures, 1990–2020. Our feasible adapted replication uses LSEG Datastream via WRDS, planned 2010–2025. The team reports a preliminary E-mini S&P 500 excess-return index **ISMCS04 with 4,033 daily observations** and contract/expiry metadata for **64 quarterly expirations**. Validate data quality, timestamps and roll conventions; expand to other futures if coverage permits. Original proprietary data are not supplied; results from WRDS will be labeled an adapted/partial replication, not exact numerical reproduction.

## Repository and scripts

- `src/momentum_lab/` — reusable causal futures-return checks, momentum/MACD features, purged chronological splits, roll-aware cost calculations, synthetic TSMOM backtest and risk metrics.
- `scripts/01_get_data.py` — *preview* of shared WRDS choices; Enze's private authenticated downloader remains local, NOT committed.
- `scripts/02_build_dataset.py` — export past-only features, separate labels.
- `scripts/03_replicate.py` — paper #67 LSTM / LSTM+CPD readiness checker (**not fitted training**).
- `scripts/04_backtest.py`, `05_course_pilot.py` — synthetic and local futures cache runs, risk/cost audit.
- `vendor/slow-momentum-fast-reversion/` — unchanged author GP-CPD / classical strategies (MIT).
- `vendor/trading-momentum-transformer/` — unchanged **supporting author LSTM/DMN/CPD implementation**; the Transformer architecture itself is out of scope. Preserve attribution/official source hashes.
- `docs/REPLICATION.md`, `docs/DATA_DECISIONS.md`, `docs/STATUS.md`, `docs/INITIAL_PROPOSAL.md` — accurate technical protocol, data caveats, project progress, assignment proposal.

## Run portable pipeline checks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
make demo
python scripts/03_replicate.py --model LSTM-CPD-21
```

**Honest status:** Previous synthetic sanity checks and futures-data utilities exist, but no actual GP-CPD features, LSTM training, reproduced published Sharpe or cost-aware neural strategy is complete. Author TensorFlow 2.3.1 / GPflow 2.2.1 need an isolated compatibility environment; modern Mac Python 3.14 is not a verified training setup. The paper's published pre-cost Sharpe **1.62 vs 2.16** is not our result.

Keep raw WRDS data, credentials and personal files out of Git. Backup of prior main: `backup-before-paper67-correction-20261011`. The existing GitHub repository *URL slug* may still reflect its old wrong name until the owner renames it in Settings.

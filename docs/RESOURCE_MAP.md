# Selected source map — project #67

- **Primary selected paper:** Wood, Roberts & Zohren (2022), [Slow Momentum with Fast Reversion](https://arxiv.org/abs/2105.13727).
- [Official selected-paper GP-CPD repo](https://github.com/kieranjwood/slow-momentum-fast-reversion): vendored under `vendor/slow-momentum-fast-reversion/`, including Matérn GP CPD / classical signals.
- [Official author shared model repo](https://github.com/kieranjwood/trading-momentum-transformer): contains `deep_momentum_network.py`, LSTM and LSTM+CPD experiment configs. We reuse the **LSTM/CPD components**; source and local-file hashes are in `UPSTREAM.json`.
- WRDS LSEG Datastream (team's ISMCS04 excess-return index; 4,033 observations; 64 quarterly-expiration metadata records) is a substitute for unavailable 50-futures Pinnacle CLC archive. Licensed local data, class assignments, raw futures and WRDS credentials must remain ignored.
- The uploaded course `ProjectTemplate.pdf` requests 2–3 pages, one proposal per group, eight specific sections. The matching Word deliverable is generated outside GitHub; `docs/INITIAL_PROPOSAL.md` summarizes the actual plan.

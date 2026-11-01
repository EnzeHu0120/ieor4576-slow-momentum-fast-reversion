# Completed experiment status

This submission records the completed 50-instrument PyTorch / SciPy-Numba port.

- Searches: 36/36; candidate fits: 1,800/1,800; training epochs: 126,763.
- CPD outputs: 346,218 per horizon, five horizons.
- Main exhibits: five test windows, 1995-2019; separately refitted 2020 test excluded.
- One search repetition; repeated-seed robustness is not claimed.

Gross Sharpe is 1.815 for LSTM and 1.602 for validation-selected CPD. Published references are 1.62 and 2.16. See report/RESULTS.md for interpretation and limitations.

The final notebook displays committed aggregate artifacts by default. A new experiment requires explicit opt-in, an authorized WRDS account and runtime dependencies. Later author-framework experiments are outside this submission. The manifest and per-search inventory establish completion.

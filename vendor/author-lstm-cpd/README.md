# Author LSTM and CPD components

Supporting implementation for Wood, Roberts and Zohren (2022), [Slow Momentum with Fast Reversion](https://arxiv.org/abs/2105.13727).

Source: [Kieran Wood’s shared author code](https://github.com/kieranjwood/trading-momentum-transformer), commit `e0352cb0bdbf8045accb1bf1a5705ae0cb9ea624`. Copyright and MIT terms remain in `LICENSE`.

This selected subset retains the author LSTM DMN, GP changepoint module, feature processing, classical strategies and backtesting. The experiment entry point accepts `LSTM`, `LSTM-CPD-21` and `LSTM-CPD-63`, defaulting to `LSTM-CPD-21`. LSTM defaults use 63 time steps. Shared backtesting imports only the LSTM model.

Local modifications and source hashes are recorded in `UPSTREAM.json` at the project root. An isolated legacy TensorFlow/GPflow environment and audited input data are still required; no fitted models are supplied.

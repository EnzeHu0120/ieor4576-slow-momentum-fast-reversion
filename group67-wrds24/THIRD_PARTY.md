# Author code and experiment provenance

Author implementation:
https://github.com/kieranjwood/trading-momentum-transformer

Inspected commit: `e0352cb0bdbf8045accb1bf1a5705ae0cb9ea624`.
The MIT license and original copyright notice are retained in
`src/project/author_code/LICENSE`. The `mom_trans` and `settings` modules are
copied into an ignored generated runtime for optional estimation, preserving
existing imports, scientific script bytes and project-relative paths.

Two explicit compatibility edits were applied to the author modules:

1. Adam `lr=` was changed to `learning_rate=` for the current optimizer API.
2. Removed NumPy `np.int` was changed to built-in `int` in batching.

The panel runner imports the author's LSTM, Sharpe loss, validation callback,
batching, feature construction and GP function. A resumable random-search
controller replaces obsolete private Keras Tuner control flow. The runner
supplies the declared 24-market policy, fixed split, fixed-N aggregation,
causal CPD cache and segmented histories. These are adaptations, not an
unchanged reproduction of the original experimental design.

The memory-bounded GP coordinator generates windows on dispatch and recycles
workers after 100 fits. This changes execution/memory behavior, not the GP
function, lookbacks or per-fit iteration cap of 200. The analysis path fix
recognizes both the live sibling LSTM directory and the exported archive layout.

The organized report wrapper was executed against the three actual run archives.
Expensive GPU optimization was not repeated merely to reorganize files.
Packaging validation is recorded in VALIDATION.json.

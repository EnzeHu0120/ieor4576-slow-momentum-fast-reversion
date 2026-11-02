# Five-asset protocol and runtime

The experiment uses the five instruments in configs/universe.csv and the settings in configs/author_five_assets.yaml. configs/author_native.yaml is an identical default alias for this five-asset setup. It does not define another universe.

Core source: vendor/author-original/, commit e0352cb0bdbf8045accb1bf1a5705ae0cb9ea624. Original files are preserved. API adaptations are in author_compat.py; reusable GPflow graph construction is isolated in author_cpd_runtime.py. Model, loss, kernel, parameter initialization and optimizer settings are retained. Numerical validation and the exact execution identity are under report/.

Test folds start in 1995, 2000, 2005, 2010 and 2015. End years are exclusive in code; the last ends at 2021. This explicitly includes 2020 and makes the last fold six calendar years. Strict 1995-2019 results are a separate sensitivity. Main metrics are equal means of fold metrics, not a pooled return-series Sharpe. CPD lookbacks are selected by pre-test diversified validation Sharpe, not test Sharpe.

Each search uses 50 candidates from the author grid, a 63-step stateless LSTM, nonoverlapping training/validation sequences, chronological per-instrument 90/10 splits, Adam, a 300-epoch cap and patience 25. TensorFlow and tuner seeds are 67 + test-start year. There is one search realization.

## Viewing

Run All in report/Paper67_Author_Five_Assets.ipynb reads committed artifacts. RUN_EXPERIMENT defaults to False. No licensed data, model weights, WRDS authentication or GPU is needed for display.

## A new data/model run

Authorized WRDS inputs and a compatible compute environment are required. The recorded GPU environment used WSL2 Ubuntu, Python 3.11, TensorFlow legacy Keras 2.21 and GPflow 2.10. Dependency versions are in configs/author_environment.lock.txt and requirements-author.txt. TensorFlow was a community Blackwell build, not an author-provided binary:

- Release: https://github.com/Syraxius/TensorflowDockerBuilder/releases/tag/v2.21.0-sm120-cuda128-py310-py313
- Wheel: tensorflow-2.21.0+selfbuild-cp311-cp311-linux_x86_64.whl
- SHA-256: a2369ec430b3d0b58ab8df2781f250ff17a4ff9ea9f7cc1c8ea2920c3f63c037

The standard CPU/GPU package appropriate to another machine must be validated before training. Install the intended TensorFlow build, then requirements-author.txt. Install this package with --no-deps if using that isolated environment. Configure resources/local/author_runtime.json from configs/author_runtime.example.json, and generate local graph validation evidence with python scripts/validate_author_graphs.py after preparing the data. Local runtime evidence and credentials are not committed. The direct Linux CLI is python scripts/03_replicate.py --config configs/author_five_assets.yaml; the notebook controller can launch the configured WSL runtime after RUN_EXPERIMENT=True.

scripts/01_get_data.py and 02_build_dataset.py prepare the five mapped WRDS inputs using configs/data_wrds.yaml. Runtime inputs and positions are stored under data/ and results/, never in Git. scripts/04_extend.py reevaluates completed model positions; scripts/05_report.py exports aggregate artifacts and regenerates figures. These entry points do not replace the author algorithms.

## Interpretation

This is an author-source five-asset WRDS study with documented compatibility and runtime changes. The original paper's 50 Pinnacle instruments, proprietary roll series, precise random-search draws and historical binary environment are not reproduced. A numerical gap cannot be assigned to any one difference without a matched-input controlled experiment. See report/RESULTS.md.


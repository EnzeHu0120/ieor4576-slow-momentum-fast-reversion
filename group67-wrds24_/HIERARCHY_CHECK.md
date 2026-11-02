# Verification against the user-supplied hierarchy

The requested structure was provided previously as **`repo_hierarchy.txt`**.
Its contents were read before this revision. An unchanged copy is included at
`docs/repo_hierarchy_requested.txt`; the original text includes example
dependency settings, which are examples rather than scientifically appropriate
replacement pins for the author's older code.

| Requested item | This ZIP | Status |
|---|---|---|
| README with paper, team, reproduction instructions and result map | `README.md` | Present |
| Python/dependency declaration | `pyproject.toml` | Present; base report dependencies and separate empirical reproduction pins |
| Makefile or one-command driver | `run_all.py` | Present; default rebuilds the audited report, optional `--stage full` estimates anew |
| Data/results/credentials/cache ignore rules | `.gitignore` | Present |
| Sample, universe, parameters, seed | `configs/config.yaml` | Present; JSON-compatible YAML |
| Untouched raw inputs | `data/raw/` | Empty directory included; original private archives are imported locally without modifying their bytes |
| Intermediate cleaned/merged data | `data/interim/` | Empty directory included; import creates the selected snapshot and isolated runtime here |
| Final analysis-ready datasets | `data/processed/` | Empty directory included; full estimation exports feature tables here |
| Importable source package | `src/project/` | Present |
| Required source modules | `__init__.py`, `paths.py`, `data.py`, `features.py`, `models.py`, `evaluation.py`, `plotting.py` | All present |
| All scientific logic inside source package | `src/project/author_code/`, `analysis_math.py`, `audit.py`, and orchestration modules | Present; preserved author scripts are resources of this source package |
| Four thin scripts | `scripts/01_get_data.py`, `02_build_dataset.py`, `03_replicate.py`, `04_extend.py` | All present |
| Exploration-only notebooks directory | `notebooks/README.md` | Present; reported results do not depend on a notebook |
| Replication tables and figures | `results/replication/tables/`, `results/replication/figures/` | Included; regenerable and ignored by git |
| Extension tables and figures | `results/extension/tables/`, `results/extension/figures/` | Empty directories included; no unperformed extension results are invented |
| Sanity checks | `tests/test_sanity.py` | Three meaningful mathematical/import-boundary checks |
| Write-up | `report/phase1_results.md` and `.pdf` | Present, English; includes paper differences, the companion five-asset evidence, and proposed universe-size research |

The ZIP explicitly records the required empty directories; ZIP tools otherwise
drop empty data/extension folders. No licensed raw observation files, saved
weights, virtual environment, GP cache, or credentials are embedded. Private
data access remains separate as described in DATA_ACCESS.md.

The optional full-estimation runtime creates additional generated directories
inside ignored paths. These do not change the requested source structure.
Rebuilding the audited reference report is verified; the reorganized expensive
GPU estimation wrapper has not been fully rerun merely for packaging.

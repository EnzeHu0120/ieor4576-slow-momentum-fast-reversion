"""Project-relative paths; no machine-specific directory is embedded."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'
ARTIFACTS=DATA/'interim/artifacts'
RUNTIME=DATA/'interim/author_runtime'
SNAPSHOT=DATA/'interim/universe_snapshot/wrds_panel24'
TABLES=ROOT/'results/replication/tables'
FIGURES=ROOT/'results/replication/figures'

def initialize():
    for p in [DATA/'raw/artifacts',SNAPSHOT,DATA/'processed',ARTIFACTS,TABLES,FIGURES,
              ROOT/'results/extension/tables',ROOT/'results/extension/figures']:
        p.mkdir(parents=True,exist_ok=True)

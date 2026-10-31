"""Paths resolved from the installed source tree, never from a home-directory cache."""
from pathlib import Path
import yaml
import os
import sys

ROOT = Path(__file__).resolve().parents[2]

def configuration(path="configs/config.yaml"):
    p = Path(path)
    return yaml.safe_load((p if p.is_absolute() else ROOT / p).read_text())

def prepare_directories():
    for p in ["data/raw/wrds", "data/interim", "data/processed", "report",
              "results/replication/tables", "results/replication/figures",
              "results/replication/checkpoints", "results/replication/logs",
              "results/extension/tables", "results/extension/figures"]:
        (ROOT / p).mkdir(parents=True, exist_ok=True)

def initialize_notebook_environment():
    """Keep any optional course-library access local to this research project."""
    os.environ["FUTSPREAD_ROOT"]=str(ROOT/"data/raw/futspread_data")
    for parent in ROOT.parents:
        if (parent/"futspread").is_dir():
            if str(parent) not in sys.path:sys.path.insert(1,str(parent))
            break

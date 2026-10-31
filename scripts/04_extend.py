"""Evaluate original-style results followed by costs and risk diagnostics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import argparse
from momentum_lab.study_evaluation import evaluate
from momentum_lab.plotting import make_figures
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/config.yaml");a=p.parse_args()
    evaluate(a.config);make_figures()

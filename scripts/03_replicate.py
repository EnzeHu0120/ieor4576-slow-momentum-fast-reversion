"""Execute and resume daily GP-CPD and expanding-window LSTM experiments."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import argparse
from momentum_lab.cpd import run as cpd
from momentum_lab.models import run as models
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/config.yaml");a=p.parse_args()
    cpd(a.config);models(a.config)

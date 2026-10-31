"""Construct same-contract futures returns and original author features."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import argparse
from momentum_lab.dataset import build
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/config.yaml");a=p.parse_args();build(a.config)

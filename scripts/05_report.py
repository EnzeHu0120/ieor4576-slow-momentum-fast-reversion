from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from momentum_lab.reporting import generate,prepare_runnable
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/config.yaml')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    (prepare_runnable if a.prepare_only else generate)(a.config)

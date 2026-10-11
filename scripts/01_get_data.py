"""Shared data-selection preview. Personal WRDS authentication/downloading lives in Enze."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/wrds_selection.example.json')
    a=p.parse_args();print(json.dumps(json.loads(a.config.read_text()),indent=2))
    print('No network access. Agree data selection before research. Enze owns the local WRDS download tool.')

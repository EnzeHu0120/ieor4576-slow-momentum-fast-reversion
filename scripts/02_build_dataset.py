"""Build a portable feature table. Real inputs require an explicitly supplied path."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from momentum_lab.data import synthetic_returns,load_returns
from momentum_lab.features import model_features

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--demo',action='store_true');g.add_argument('--returns',type=Path)
    a=p.parse_args();r=synthetic_returns() if a.demo else load_returns(a.returns)
    if r.isna().any().any(): raise ValueError('Feature export requires an audited complete panel; no silent fill')
    out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
    f=model_features(r);f.to_parquet(out/'features.parquet')
    f.drop(columns='label_date').to_csv(out/'author_features_non_cpd.csv')
    r.to_parquet(out/'returns.parquet')
    print(f'{len(f):,} feature rows; labels separate from X. source={"SYNTHETIC" if a.demo else a.returns}')

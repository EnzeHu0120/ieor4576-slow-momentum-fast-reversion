from pathlib import Path
import argparse,json,os,sys
os.environ['TF_USE_LEGACY_KERAS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from momentum_lab.author_native import configuration,evaluate
from momentum_lab.author_compat import load_author
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/author_five_assets.yaml');a=p.parse_args()
    cfg=configuration(a.config)
    receipts=[json.loads(p.read_text()) for p in (ROOT/cfg['results_directory']/'models').glob('*/*/receipt.json')]
    expected=(len(cfg['cpd_windows'])+1)*len(cfg['test_boundaries'])
    assert len(receipts)==expected,'All requested searches must be complete before evaluation.'
    load_author();evaluate(cfg,receipts)

"""Read-only experiment progress and strict completion checks."""
import json
import pyarrow.parquet as pq
from .paths import ROOT,configuration

def status():
    cfg=configuration();expected=pq.read_metadata(ROOT/'data/processed/features.parquet').num_rows
    completed=[]
    for path in (ROOT/'results/replication/checkpoints').glob('*/*/receipt.json'):
        try:
            d=json.loads(path.read_text())
            if len(d['trials'])==cfg['random_search_trials'] and d['config']==cfg:
                completed.append(f"{d['strategy']}:{d['fold']}")
        except (OSError,ValueError):pass
    fits=len(list((ROOT/'results/replication/checkpoints').glob('*/*/trial_*.json')))
    rows={}
    for l in cfg['cpd_windows']:
        try:rows[str(l)]=sum(pq.read_metadata(p).num_rows for p in (ROOT/f'data/processed/cpd/{l}').glob('*.parquet'))
        except Exception:rows[str(l)]=-1  # parquet writer may be replacing a checkpoint
    n=len(cfg['test_boundaries'])*(1+len(cfg['cpd_windows']))
    return {'completed_searches':len(completed),'expected_searches':n,'completed_candidates':fits,
        'expected_candidates':n*cfg['random_search_trials'],'cpd_rows':rows,'expected_cpd_rows_per_window':expected,
        'completed':sorted(completed),'ready':len(completed)==n and all(x==expected for x in rows.values())}

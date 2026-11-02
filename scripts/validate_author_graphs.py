"""Compare graph reuse against untouched author fitting on real windows."""
import os,sys,json,time,resource,hashlib
from pathlib import Path
os.environ.update(TF_USE_LEGACY_KERAS='1',CUDA_VISIBLE_DEVICES='-1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from momentum_lab.author_compat import load_author
load_author()
import pandas as pd,numpy as np
from mom_trans.changepoint_detection import changepoint_loc_and_score
from momentum_lab.author_cpd_runtime import install,uninstall
raw=pd.read_parquet(ROOT/'data/processed/author_five_assets/cpd_inputs.parquet')
raw=raw[raw.ticker.eq('BN')].sort_values('date').reset_index(drop=True)
records=[]
for window in [10,21,63,126,252]:
    cases=[]
    for end in [601,602,803]:
        f=raw.iloc[end-window-1:end][['date','daily_returns']].rename(columns={'daily_returns':'Y'}).copy();f['X']=f.index.astype(float);cases.append(f)
    uninstall();begin=time.perf_counter();reference=[changepoint_loc_and_score(f)[:3] for f in cases];original_seconds=time.perf_counter()-begin
    pool=install();begin=time.perf_counter();actual=[changepoint_loc_and_score(f)[:3] for f in cases];cached_seconds=time.perf_counter()-begin
    np.testing.assert_allclose(actual,reference,rtol=1e-5,atol=1e-5)
    caches=[len(opt.compile_cache) for opt in pool.optimizers.values()]
    assert caches==[1,1],caches
    rec={'window':window,'cases':len(cases),'author_seconds':original_seconds,'cached_seconds_including_first_compile':cached_seconds,'max_absolute_difference':float(np.max(np.abs(np.asarray(actual)-reference))),'compiled_graphs':sum(caches),'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    records.append(rec);print(json.dumps(rec),flush=True)
report={'validated':True,'records':records,'runtime_sha256':hashlib.sha256((ROOT/'src/momentum_lab/author_cpd_runtime.py').read_bytes()).hexdigest()}
(ROOT/'resources/local/author_graph_validation.json').write_text(json.dumps(report,indent=2))
print('AUTHOR GRAPH REUSE NUMERICAL EQUIVALENCE: PASS',flush=True)

"""Memory-bounded CPD recovery runner. Keep the original panel runner unchanged."""
import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path


def main():
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import pandas as pd
    import run_wrds_panel24 as original
    from paper67_panel_gp import initialize_worker

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--recycle',type=int,default=100)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--cache',default=str(original.PROCESSED/'cpd_cache'))
    args=parser.parse_args()
    original.require(args.workers>0 and args.recycle>0 and args.limit>=0,'Invalid worker settings.')
    original.require(Path('mom_trans/changepoint_detection.py').is_file(),'Run from ~/paper67-author.')
    features=original.read_table(original.FEATURE_PATH)
    levels=pd.read_csv(original.LEVEL_PATH,parse_dates=['signal_date'])
    cache=Path(args.cache);out=original.RESULTS/'cpd';out.mkdir(parents=True,exist_ok=True)
    cfg={'features_sha256':original.digest(original.FEATURE_PATH),
         'levels_sha256':original.digest(original.LEVEL_PATH),
         'author_cpd_sha256':original.digest(Path('mom_trans/changepoint_detection.py')),
         'lookbacks':original.LOOKBACKS,'iteration_cap':200,
         'initialization':'author defaults independently reset each window',
         'raw_return_policy':'unwinsorized percent changes within each cleaned independent segment'}
    original.freeze(cache/'config.json',cfg)
    indexed={}
    for seg,g in levels.groupby('segment'):
        g=g.sort_values('position')
        indexed[seg]=(g.signal_date.dt.strftime('%Y-%m-%d').to_numpy(),g.close.pct_change(fill_method=None).to_numpy())

    def descriptors():
        counts=defaultdict(int)
        for (seg,code),g in features.groupby(['segment','code'],sort=True):
            for row in g.itertuples():
                date=str(row.signal_date.date());pos=int(row.source_position)
                for lbw in original.LOOKBACKS:
                    key=(int(code),lbw)
                    if args.limit and counts[key]>=args.limit:continue
                    counts[key]+=1
                    yield seg,lbw,pos,date

    def path_for(seg,lbw,date):
        return cache/seg/('lbw_%d'%lbw)/(date+'.json')

    def cached(seg,lbw,pos,date):
        p=path_for(seg,lbw,date)
        if not p.is_file():return None
        row=json.loads(p.read_text())
        original.require(row['segment']==seg and row['lookback']==lbw and row['t']==pos and row['date']==date,'Mismatched cached window: '+str(p))
        return row

    # Keep only compact descriptors. Do not retain all expanded numeric windows
    # or all fit-result dictionaries in the coordinator process.
    pending=[];total=0;reused=0
    for seg,lbw,pos,date in descriptors():
        total+=1
        row=cached(seg,lbw,pos,date)
        if row is None or (args.retry_failed and row['status']=='failed'):
            pending.append((seg,lbw,pos,date))
        else:reused+=1
    print('GP windows:',total,'| Cached:',reused,'| To fit:',len(pending),flush=True)
    print('Workers:',args.workers,'| Recycle after:',args.recycle,'fits per worker',flush=True)

    def jobs():
        for seg,lbw,pos,date in pending:
            dates,returns=indexed[seg];values=returns[pos-lbw:pos+1]
            original.require(len(values)==lbw+1 and np.isfinite(values).all(),'Invalid GP input window.')
            yield seg,(lbw,pos,date,str(dates[pos-lbw]),values.tolist())

    started=time.monotonic();done=0
    if pending:
        with mp.get_context('spawn').Pool(args.workers,initializer=initialize_worker,
                initargs=(str(Path.cwd()),),maxtasksperchild=args.recycle) as pool:
            results=pool.imap_unordered(original.gp_task,jobs(),chunksize=1)
            while done<len(pending):
                try:row=results.next(timeout=15)
                except mp.TimeoutError:
                    print('Heartbeat: coordinator alive | completed %d/%d | elapsed %.1fs'%(done,len(pending),time.monotonic()-started),flush=True)
                    continue
                original.save_json(path_for(row['segment'],row['lookback'],row['date']),row)
                done+=1
                if done==1 or done%100==0 or done==len(pending):
                    elapsed=time.monotonic()-started
                    eta=(len(pending)-done)*elapsed/done/3600
                    print('GP %d/%d | %s | lbw %d | %s | elapsed %.1fs | estimated remaining %.2fh'%
                          (done,len(pending),row['segment'],row['lookback'],row['status'],elapsed,eta),flush=True)
    del pending
    # Read diagnostic fields from disk one result at a time.
    diag_path=out/('smoke_windows.csv' if args.limit else 'fit_windows.csv')
    failures=0;seconds=0.;n=0
    with diag_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['segment','lookback','date','status','elapsed_seconds']);writer.writeheader()
        for seg,lbw,pos,date in descriptors():
            row=cached(seg,lbw,pos,date)
            original.require(row is not None,'A completed window is missing from cache.')
            writer.writerow({k:row[k] for k in writer.fieldnames})
            failures+=row['status']!='completed';seconds+=float(row['elapsed_seconds']);n+=1
    if args.limit:
        original.require(n==24*len(original.LOOKBACKS)*args.limit and failures==0,'Smoke test incomplete or failed. Inspect '+str(diag_path))
        print('Mean fit seconds:',round(seconds/n,4))
        print('Smoke only: no complete CPD model feature table was emitted.')
        print('PANEL24_CPD_SMOKE_OK',flush=True)
        return
    full=features.copy();filled=[]
    for seg,g in features.groupby('segment',sort=True):
        ordered=g.sort_values('signal_date')
        for lbw in original.LOOKBACKS:
            past=None;scores=[];locations=[];sources=[]
            for row in ordered.itertuples():
                date=str(row.signal_date.date());pos=int(row.source_position)
                fit=cached(seg,lbw,pos,date)
                original.require(fit is not None,'Missing cached GP result.')
                if fit['status']=='completed':past=fit
                original.require(past is not None,'Initial GP failure has no prior fit in '+seg+'. Inspect the error and rerun with --retry-failed.')
                original.require(pd.Timestamp(past['date'])<=row.signal_date,'Noncausal GP source.')
                scores.append(float(past['cp_score']));locations.append((pos-past['cp_location'])/lbw);sources.append(past['date'])
                if fit['status']!='completed':filled.append({'segment':seg,'lookback':lbw,'date':date,'source_date':past['date']})
            full.loc[ordered.index,'cp_score_%d'%lbw]=scores
            full.loc[ordered.index,'cp_rl_%d'%lbw]=locations
            full.loc[ordered.index,'cp_source_date_%d'%lbw]=sources
    cols=['cp_score_%d'%l for l in original.LOOKBACKS]+['cp_rl_%d'%l for l in original.LOOKBACKS]
    original.require(np.isfinite(full[cols]).all().all(),'Incomplete CPD feature values.')
    temporary=original.CPD_PATH.with_suffix('.csv.tmp');full.to_csv(temporary,index=False);temporary.replace(original.CPD_PATH)
    original.save_json(out/'CPD_COMPLETE.json',{'windows':n,'failed_fits_filled_from_past':filled,
        'elapsed_seconds_this_invocation':time.monotonic()-started,'cpd_features_sha256':original.digest(original.CPD_PATH),
        'config':cfg,'execution_runner':'streamed_windows_recycled_workers','workers':args.workers,'recycle':args.recycle})
    print('PANEL24_CPD_FEATURES_OK',flush=True)


if __name__=='__main__':
    main()

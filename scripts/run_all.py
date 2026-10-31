"""One command rebuilds all results from local, licensed WRDS snapshots."""
import argparse
from pathlib import Path
import sys
import json
from datetime import datetime,timezone
from filelock import FileLock,Timeout
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from momentum_lab.paths import ROOT,prepare_directories
from momentum_lab.dataset import build
from momentum_lab.cpd import run as cpd
from momentum_lab.models import run_model
from momentum_lab.study_evaluation import evaluate
from momentum_lab.plotting import make_figures
from momentum_lab.reporting import generate,inventory_manifest,conclusion_text,METHODS
from momentum_lab.paths import configuration

def run(config_path,prepared_features=False,keep_notebook=False,model_workers=3,fresh_models=False):
    prepare_directories()
    # Keep two local Run All / terminal launches from writing the same trials.
    with FileLock(str(ROOT/'results/replication/local_run.lock'),timeout=0):
        if fresh_models:
            checkpoint=ROOT/'results/replication/checkpoints'
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            archive=ROOT/f'results/archives/before_fresh_{stamp}'
            archive.mkdir(parents=True)
            if checkpoint.exists() and any(checkpoint.iterdir()):checkpoint.rename(archive/'checkpoints')
            checkpoint.mkdir(parents=True,exist_ok=True)
            (archive/'reason.json').write_text(json.dumps({'reason':'User requested model training from random initialization',
                'config':config_path,'started_utc':datetime.now(timezone.utc).isoformat()},indent=2))
            for name in ['execution_manifest.json','RESULTS.md']:
                path=ROOT/'report'/name
                if path.exists():path.rename(archive/name)
            print('FRESH MODEL RUN: zero fitted candidates reused. Previous work preserved at',archive,flush=True)
        tickers=__import__('pandas').read_csv(ROOT/'configs/universe.csv').ticker
        raw_ready=all((ROOT/f'data/raw/wrds/{t}/{name}').exists() for t in tickers for name in ['bars.parquet','contracts.parquet'])
        raw_ready=raw_ready and all((ROOT/f'data/raw/wrds/{t}/calendar_reference.parquet').exists() for t in ['GI','ZW','ZZ'])
        if not raw_ready:
            print('Stage: acquire missing licensed WRDS data using your own WRDS authentication',flush=True)
            from momentum_lab.acquisition import download
            download(config_path)
        cfg=configuration(config_path)
        if not prepared_features or not (ROOT/'data/processed/features.parquet').exists():build(config_path)
        print('Stage: GP-CPD (reuse matching checkpoints)',flush=True)
        cpd(config_path)
        jobs=[(year,lbw) for lbw in [None]+cfg['cpd_windows'] for year in cfg['test_boundaries']]
        print('Stage: full model searches; completed candidates are reused',flush=True)
        if model_workers==1:
            for year,lbw in jobs:print('Completed',run_model(year,lbw,config_path),flush=True)
        else:
            with ProcessPoolExecutor(max_workers=model_workers) as pool:
                futures=[pool.submit(run_model,year,lbw,config_path) for year,lbw in jobs]
                for future in as_completed(futures):print('Completed',future.result(),flush=True)
        print('Stage: full backtest, costs, figures and reports',flush=True)
        evaluate(config_path);make_figures()
        if keep_notebook:
            manifest=inventory_manifest(cfg,config_path)
            if manifest['missing_experiments']:raise RuntimeError('Missing model searches: '+json.dumps(manifest['missing_experiments']))
            (ROOT/'report/RESULTS.md').write_text('# Paper 67 — Executed results\n\n'+conclusion_text()+'\n\n'+METHODS)
            from momentum_lab.deliverables import deliver
            deliver()
        else:generate(config_path)
        print('REPRODUCTION_COMPLETE',flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/config.yaml')
    p.add_argument('--prepared-features',action='store_true',help='Reuse the prepared feature panel; GP/model signatures still check compatibility')
    p.add_argument('--keep-notebook',action='store_true',help='Do not replace the notebook currently running this pipeline')
    p.add_argument('--model-workers',type=int,choices=[1,2,3],default=3)
    p.add_argument('--fresh-models',action='store_true',help='Archive old model checkpoints and train every candidate from random initialization')
    a=p.parse_args();run(a.config,a.prepared_features,a.keep_notebook,a.model_workers,a.fresh_models)

"""Author-source entry point. A full run must be started explicitly by the user."""
import argparse
import os
from pathlib import Path
import sys

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/author_native.yaml')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--status', action='store_true')
    args = p.parse_args()
    from momentum_lab.author_native import preflight, run
    if args.status:
        import json
        from filelock import FileLock, Timeout
        from momentum_lab.author_native import ROOT, configuration
        output = ROOT / configuration(args.config)['results_directory']
        output.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(output / 'run.lock'))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            session = output / 'run_session.json'
            record = json.loads(session.read_text()) if session.exists() else {}
            print(json.dumps({'running':True, **record}))
        else:
            lock.release()
            print(json.dumps({'running':False}))
    elif args.preflight:
        print(preflight(), flush=True)
    else:
        run(args.config, smoke=args.smoke)

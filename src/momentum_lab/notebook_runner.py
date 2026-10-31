"""Blocking local notebook entry point: compute uses the computer, no agent polling."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from .paths import ROOT,prepare_directories,configuration
from .training_events import PREFIX

def run_prepared(config_path='configs/config.yaml',model_workers=3,fresh_models=True):
    """Finish cached experiments and all reports without overwriting the open notebook."""
    prepare_directories()
    python=ROOT/'.venv-replication/bin/python'
    if not python.exists():python=Path(sys.executable)
    log=ROOT/'results/replication/logs/local_notebook_run.log'
    command=[str(python),'-u',str(ROOT/'scripts/run_all.py'),'--config',str(config_path),
             '--prepared-features','--keep-notebook','--model-workers',str(model_workers)]
    if fresh_models:command.append('--fresh-models')
    print('Fresh model training from random initialization.' if fresh_models else 'Resuming compatible model checkpoints.',flush=True)
    print('Raw data and causal GP feature caches may be reused; no fitted model is reused in fresh mode.',flush=True)
    print('The cell stays busy until all training, backtests and reports finish.',flush=True)
    print('Log:',log,flush=True)
    cfg=configuration(config_path)
    from .models import resolve_device
    print('Compute device:',resolve_device(cfg.get('device','auto')),'| parallel model workers:',model_workers,flush=True)
    from .inline_training import InlineTrainingPlot
    plot=InlineTrainingPlot((1+len(cfg['cpd_windows']))*len(cfg['test_boundaries'])*cfg['random_search_trials'],model_workers,fresh=fresh_models)
    with log.open('a',buffering=1) as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            start_new_session=True,text=True,bufsize=1)
        if sys.platform=='darwin' and shutil.which('caffeinate'):
            # Temporary idle-sleep assertion ends when this one pipeline exits.
            subprocess.Popen(['caffeinate','-i','-w',str(process.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            for line in process.stdout:
                stream.write(line)
                if line.startswith(PREFIX):
                    plot.update(json.loads(line[len(PREFIX):]));continue
                fields=line.split()
                # Cached GP panels need no 250-line replay in the notebook.
                if len(fields)==5 and fields[0]=='GP-CPD' and fields[3]=='0':continue
                print(line,end='',flush=True)
            returncode=process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(timeout=15)
            print('Local run interrupted; completed candidates remain cached.',flush=True)
            raise
    if returncode:
        tail='\n'.join(log.read_text(errors='replace').splitlines()[-25:])
        raise RuntimeError(f'Local reproduction exited {returncode}. Completed checkpoints are retained.\n{tail}')
    manifest=json.loads((ROOT/'report/execution_manifest.json').read_text())
    plot.draw(force=True,finished=True)
    print(f"Complete: {manifest['completed_experiments']}/{manifest['expected_experiments']} searches; {manifest['completed_trials']} candidate fits.",flush=True)
    return manifest

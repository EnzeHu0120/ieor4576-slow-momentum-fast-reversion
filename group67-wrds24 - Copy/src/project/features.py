"""Execute the preserved author-compatible feature and GP implementation."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import json
from . import paths

def config():
    # JSON is a YAML 1.2 subset and needs no additional parser here.
    value=json.loads((paths.ROOT/'configs/config.yaml').read_text())
    if value['cpd_lookbacks']!=[10,21,63,126,252] or value['sequence_length']!=63:
        raise ValueError('This replication preserves the five CPD windows and author sequence length of 63.')
    if value['sample']!={'train_end':'2016-12-31','validation_end':'2018-12-31','test_end':'2020-12-31'}:
        raise ValueError('Changing the fixed sample requires a separately declared experiment and data-policy revision.')
    return value

def prepare_runtime():
    """Keep original audited script bytes intact in an isolated generated tree."""
    paths.initialize();paths.RUNTIME.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).parent/'author_code'
    for p in source.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            dest=paths.RUNTIME/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
    for name in ['source_daily.csv','universe24.json']:
        source=paths.SNAPSHOT/name
        if not source.is_file():raise FileNotFoundError('Import private WRDS raw inputs first: '+str(source))
        dest=paths.RUNTIME/'data/raw/wrds_panel24'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)

def execute(script,*arguments):
    # Bootstrap flags belong to a single interpreter startup. Do not inherit
    # a previous script's readiness marker into a new interpreter invocation.
    env=os.environ.copy();env.pop('PAPER67_PANEL24_ENV_READY',None)
    env['PYTHONPATH']=str(paths.RUNTIME)
    subprocess.run([sys.executable,'-u',str(paths.RUNTIME/script),*map(str,arguments)],
                   cwd=paths.RUNTIME,env=env,check=True)

def build_base():
    from .data import validate_snapshot
    config();validate_snapshot();prepare_runtime();execute('run_wrds_panel24.py','prepare')
    export_processed(['features.csv','segmented_levels.csv'])

def build_cpd(workers=None):
    c=config();execute('run_wrds_panel24_cpd_safe.py','--workers',workers or c['gp_workers'])
    export_processed(['features_cpd.csv'])

def export_processed(names):
    destination=paths.DATA/'processed/wrds_panel24';destination.mkdir(parents=True,exist_ok=True)
    for name in names:
        source=paths.RUNTIME/'data/processed/wrds_panel24'/name
        if not source.is_file():raise FileNotFoundError('Processed dataset was not generated: '+str(source))
        shutil.copy2(source,destination/name)

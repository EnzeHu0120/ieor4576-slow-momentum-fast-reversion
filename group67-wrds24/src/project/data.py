"""Import the actual experiment archives and privately obtained WRDS data."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from . import paths

ARCHIVES={
 'paper67_panel24_lstm_results.zip':('lstm','c10b8e2b54cbe33c6eaaa6bb7a0e0f4764ed1b2ec980542009b9c48b5730613a'),
 'paper67_panel24_cpd_results.zip':('cpd','2eb35a40495cf29ec3953700586552f7102f8b65f3b9a4e38e6b9f74f8662943'),
 'paper67_panel24_supplement_results.zip':('supplement','776e8ce75725c8230f6f45982444a6111b5d16d8307789232e71ba2e06686613'),
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def extract_archive(source,destination):
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:raise ValueError('A ZIP member failed CRC verification.')
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate ZIP members.')
        if not all((destination/n).resolve().is_relative_to(destination) for n in z.namelist()):
            raise ValueError('Unsafe archive member path.')
        z.extractall(destination)

def import_archives(directory=None):
    paths.initialize()
    for filename,(kind,expected) in ARCHIVES.items():
        destination=paths.DATA/'raw/artifacts'/filename
        if directory:
            source=Path(directory).expanduser()/filename
            if not source.is_file():raise FileNotFoundError('Missing experiment archive: '+str(source))
            if sha(source)!=expected:raise ValueError('Archive differs from the verified experiment: '+filename)
            if source.resolve()!=destination.resolve():shutil.copy2(source,destination)
        if not destination.is_file():raise FileNotFoundError('Use --bundle-dir to import the three experiment archives. Missing '+filename)
        if sha(destination)!=expected:raise ValueError('Unexpected archive SHA-256: '+filename)
        target=paths.ARTIFACTS/kind;marker=target/'IMPORT_SHA256.json'
        if marker.is_file() and json.loads(marker.read_text()).get('sha256')==expected:continue
        extract_archive(destination,target)
        marker.write_text(json.dumps({'filename':filename,'sha256':expected},indent=2)+'\n')

def import_raw(author_repo):
    paths.initialize();source=Path(author_repo).expanduser()/'data/raw/wrds_panel24'
    for name in ['source_daily.csv','universe24.json']:
        p=source/name
        if not p.is_file():raise FileNotFoundError('Original raw input not found: '+str(p))
        destination=paths.SNAPSHOT/name
        if p.resolve()!=destination.resolve():shutil.copy2(p,destination)

def import_new_runtime_results():
    source=paths.RUNTIME/'results/wrds_panel24'
    for name,(folder,_) in ARCHIVES.items():
        relative={'lstm':'lstm_seed67/results_bundle.zip','cpd':'cpd_seed67/results_bundle.zip',
                  'supplement':'supplement_seed67/supplement_bundle.zip'}[folder]
        p=source/relative
        if not p.is_file():raise FileNotFoundError('Reproduction did not produce '+str(p))
        # A new training run is a new experiment, not the hash-pinned reference.
        # Preserve its archive separately; never overwrite the verified original.
        dest=paths.ROOT/'results/new_experiment/archives'/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    summary=source/'supplement_seed67'
    exported=paths.ROOT/'results/new_experiment'
    for p in summary.rglob('*'):
        relative=p.relative_to(summary)
        if p.is_file() and ('audit_inputs' not in relative.parts) and p.name!='all_available_test_asset_predictions.csv' and p.suffix in ['.csv','.json','.png']:
            dest=exported/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    print('New training archives saved under results/new_experiment/archives. The verified reference was preserved.')
    print('New experiment summary tables and figures: results/new_experiment/.')

"""Check tracked/candidate text without printing any matched sensitive value."""
import argparse
from pathlib import Path
import re
import subprocess
import json

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    'absolute personal path':re.compile(r'/(?:Users|home)/[A-Za-z0-9_.-]+/'),
    'private key':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'GitHub credential':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})'),
    'credential URL':re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
    'literal password assignment':re.compile(r'''(?i)(?:password|passwd|api_key|access_token)\s*[:=]\s*["'][^"'\s]{8,}["']'''),
}

def check(staged=False,commit=None):
    args=['git','ls-tree','-r','--name-only','-z',commit] if commit else ['git','diff','--cached','--name-only','--diff-filter=ACMR','-z'] if staged else ['git','ls-files','--cached','--others','--exclude-standard','-z']
    files=subprocess.check_output(args,cwd=ROOT).decode().split('\0');findings=[];count=0
    for name in sorted(set(files)-{''}):
        path=ROOT/name
        if not commit and not path.is_file():continue
        data=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT) if commit else subprocess.check_output(['git','show',f':{name}'],cwd=ROOT) if staged else path.read_bytes()
        count+=1
        if name.startswith(('data/','results/','.venv')) or Path(name).suffix in {'.pt','.parquet','.pem'} or Path(name).name in {'.env','.pgpass','.netrc'}:
            findings.append({'file':name,'issue':'excluded data/runtime file'})
        if len(data)>10_000_000:findings.append({'file':name,'issue':'file exceeds 10 MB'})
        try:text=data.decode()
        except UnicodeDecodeError:continue
        for category,pattern in PATTERNS.items():
            if pattern.search(text):findings.append({'file':name,'issue':category})
    print(json.dumps({'checked_files':count,'findings':findings},indent=2))
    return not findings

if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group();mode.add_argument('--staged',action='store_true');mode.add_argument('--commit')
    a=p.parse_args();raise SystemExit(0 if check(a.staged,a.commit) else 1)

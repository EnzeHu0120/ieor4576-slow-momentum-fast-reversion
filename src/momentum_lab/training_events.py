"""Structured progress only; no changes to fitting, selection or model randomness."""
import json

PREFIX='PAPER67_EVENT '

def emit(kind,checkpoint,**values):
    event={'kind':kind,'strategy':checkpoint.parent.parent.name,
        'fold':int(checkpoint.parent.name),'trial':int(checkpoint.stem.split('_')[1]),**values}
    print(PREFIX+json.dumps(event,separators=(',',':')),flush=True)

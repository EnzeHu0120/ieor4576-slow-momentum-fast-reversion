"""Structured progress only; no changes to fitting, selection or model randomness."""
import json

PREFIX='PAPER67_EVENT '

def parse_events(line):
    """Decode progress records even when worker output shares a line."""
    decoder = json.JSONDecoder()
    position = 0
    while True:
        start = line.find(PREFIX, position)
        if start < 0:
            return
        position = start + len(PREFIX)
        try:
            event, consumed = decoder.raw_decode(line[position:].lstrip())
        except json.JSONDecodeError:
            # A damaged display record must not abort the computation.
            continue
        if isinstance(event, dict):
            yield event

def emit(kind,checkpoint,**values):
    event={'kind':kind,'strategy':checkpoint.parent.parent.name,
        'fold':int(checkpoint.parent.name),'trial':int(checkpoint.stem.split('_')[1]),**values}
    print(PREFIX+json.dumps(event,separators=(',',':')),flush=True)

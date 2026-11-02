"""Notebook monitor: WSL GPU computation, independent per-process event files."""
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

def decode_output(raw):
    encoding = 'utf-16-le' if raw and raw.count(b'\0') > len(raw)//8 else 'utf-8'
    return raw.decode(encoding, errors='replace')


def linux_path(path):
    path = Path(path).resolve()
    if os.name != 'nt':
        return str(path)
    return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]


def command(smoke=False, config_path='configs/author_native.yaml'):
    runtime = json.loads((ROOT / 'resources/local/author_runtime.json').read_text(encoding='utf-8'))
    args = [runtime['python'], '-u', linux_path(ROOT / 'scripts/run_author.py')]
    args += ['--config', linux_path(ROOT / config_path)]
    if smoke:
        args.append('--smoke')
    if os.name == 'nt':
        return ['wsl', '-d', runtime['distribution'], '-u', runtime['user'], '--'] + args
    return args


def read_events(folder):
    events = []
    for path in sorted(Path(folder).glob('*.jsonl')):
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 131072))
            tail = stream.read().decode('utf-8', errors='replace')
        for line in tail.splitlines():
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # partial last row is retried on the next refresh
            if isinstance(event, dict):
                events.append(event)
    return sorted(events, key=lambda e: e.get('utc', ''))


def check_backend_failure(output, since=''):
    """Report a failed parent immediately, even if WSL process cleanup hangs."""
    for path in (Path(output) / 'events').glob('*.status.json'):
        try:
            event = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if event.get('kind') == 'failed' and event.get('utc', '') >= since:
            raise RuntimeError(f"Backend failed at {event.get('utc')}: "
                               f"{event.get('error')}: {event.get('message')}\n"
                               f"Saved progress retained. Log: {Path(output) / 'run.log'}")


def run_author(smoke=False, text_progress=True, config_path='configs/author_native.yaml'):
    from IPython.display import display, Image
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from io import BytesIO
    import yaml
    cfg = yaml.safe_load((ROOT / config_path).read_text(encoding='utf-8'))
    output = ROOT / ('results/author_smoke' if smoke else cfg['results_directory'])
    output.mkdir(parents=True, exist_ok=True)
    log = output / 'run.log'
    before = time.time()
    if not smoke:
        status = subprocess.run(command(False, config_path) + ['--status'], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if status.returncode:
            raise RuntimeError('Cannot inspect the WSL run lock. No new training was launched.\n' + decode_output(status.stderr))
        state = json.loads(decode_output(status.stdout).strip())
        if state['running']:
            print('Existing computation found. Attaching to progress; no duplicate process launched.', flush=True)
            from momentum_lab.author_progress import monitor_text
            return monitor_text(since=state.get('started_at', ''), config_path=config_path)
    print('Author TensorFlow/Keras + GPflow; isolated output:', output)
    print('Resume enabled. Old PyTorch weights/CPD caches are excluded.')
    print('Log:', log)
    # No stdout PIPE: output decoding / chart errors cannot break the training pipe.
    with log.open('ab', buffering=0) as stream:
        proc = subprocess.Popen(command(smoke, config_path), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if text_progress and not smoke:
        from momentum_lab.author_progress import monitor_text
        return monitor_text(process=proc, since=time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(before-1)), config_path=config_path)
    handle = None
    plotting_ok = True
    try:
        while True:
            events = [e for e in read_events(output / 'events')
                      if e.get('utc', '') >= time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(before-1))]
            if plotting_ok:
                try:
                    fig = Figure(figsize=(11, 5), layout='constrained')
                    FigureCanvasAgg(fig)
                    axes = fig.subplots(1, 2)
                    stage = events[-1]['kind'] if events else 'Starting WSL author environment'
                    receipts = list((output / 'models').glob('*/*/receipt.json'))
                    done = [json.loads(p.read_text(encoding='utf-8')) for p in receipts]
                    total = 1 if smoke else 1800
                    completed = sum(len(r['trials']) for r in done)
                    current = [e for e in events if e['kind']=='candidate_completed'
                               and not any(r['strategy']==e['strategy'] and r['fold']==e['fold'] for r in done)]
                    if current:
                        completed += current[-1]['completed_in_search']
                    axes[0].set_title(f'Author source | {completed} / {total} saved candidates')
                    axes[0].barh([0], [total], color='#e5e7eb')
                    axes[0].barh([0], [completed], color='#168a70')
                    axes[0].set_xlim(0, total)
                    axes[0].set_yticks([])
                    axes[0].set_xlabel(stage)
                    cp = [e for e in events if e['kind'] == 'cpd']
                    if cp:
                        e = cp[-1]
                        axes[0].text(.02, .1, f"GPflow: {e['ticker']} / {e['window']} days: {e['completed']}/{e['total']}",
                                     transform=axes[0].transAxes)
                    epochs = [e for e in events if e['kind'] == 'epoch']
                    if epochs:
                        last = epochs[-1]
                        current = [e for e in epochs if e['strategy'] == last['strategy'] and e['fold'] == last['fold']]
                        # Show current candidate only; epoch numbering resets between trials.
                        reset = max([i for i,e in enumerate(current) if e['epoch'] == 1] or [0])
                        current = current[reset:]
                        axes[1].plot([e['epoch'] for e in current], [e['validation_sharpe'] for e in current])
                        axes[1].set_title(f"{last['strategy']} / {last['fold']} validation Sharpe")
                        axes[1].set_xlabel('Epoch')
                    else:
                        axes[1].text(.5, .5, 'GPflow preprocessing / waiting for LSTM epochs', ha='center', transform=axes[1].transAxes)
                    buffer = BytesIO()
                    fig.savefig(buffer, format='png', dpi=100)
                    if not smoke:
                        from momentum_lab.author_progress import render_progress
                        buffer = BytesIO(render_progress())
                    if handle is None:
                        handle = display(Image(data=buffer.getvalue()), display_id=True)
                    else:
                        handle.update(Image(data=buffer.getvalue()))
                except Exception as exc:
                    plotting_ok = False
                    print(f'Chart stopped ({type(exc).__name__}); computation continues. Read {log}.')
            code = proc.poll()
            check_backend_failure(output, time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(before-1)))
            if code is not None:
                if code:
                    tail = decode_output(log.read_bytes()[-16000:])
                    raise RuntimeError(f'Author run exited {code}. Resume files retained.\n{tail}')
                print('Author run complete. Outputs:', output)
                return
            time.sleep(3)
    except KeyboardInterrupt:
        print('Notebook monitoring stopped. The WSL computation may still be running; do not launch another run.')
        raise

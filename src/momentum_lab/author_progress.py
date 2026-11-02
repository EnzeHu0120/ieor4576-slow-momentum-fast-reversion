"""Read-only live dashboard; never starts or interrupts training."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=36)
def lstm_context(strategy, year, config_path='configs/author_native.yaml'):
    """Describe per-instrument chronological splits from the saved input dates."""
    import pandas as pd
    import yaml
    cfg = yaml.safe_load((ROOT / config_path).read_text(encoding='utf-8'))
    saved = ROOT / cfg['results_directory'] / 'models' / strategy / str(year) / 'training_context.json'
    if saved.exists():
        return json.loads(saved.read_text(encoding='utf-8'))
    base = ROOT / cfg['data_directory']
    panel = pd.read_parquet(base / 'features.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel[panel.published_first_test <= year]
    if '-CPD-' in strategy:
        window = int(strategy.rsplit('-', 1)[1])
        dates = []
        for path in (base / 'cpd' / str(window)).glob('*.csv'):
            cp = pd.read_csv(BytesIO(complete_csv_bytes(path)), usecols=['date'])
            cp['date'] = pd.to_datetime(cp['date'])
            cp['ticker'] = path.stem
            dates.append(cp)
        panel = panel.merge(pd.concat(dates), on=['date', 'ticker'])
    counts = panel[panel.year < year].groupby('ticker').size()
    eligible = counts[counts * (1-cfg['train_valid_ratio']) >= cfg['sequence_length']].index
    panel = panel[panel.ticker.isin(eligible)].dropna()
    panel = panel[panel.year >= cfg['start']].sort_values(['ticker', 'date'])
    boundaries = cfg['test_boundaries']
    end = boundaries[boundaries.index(year)+1] if year != boundaries[-1] else cfg['last_test_end']
    records = []
    for ticker, rows in panel[panel.year < year].groupby('ticker'):
        split = int(cfg['train_valid_ratio'] * len(rows))
        train, valid = rows.iloc[:split], rows.iloc[split:]
        if train.empty or valid.empty:
            continue
        records.append({'instrument': ticker, 'train_start': str(train.date.min().date()),
            'train_end': str(train.date.max().date()), 'valid_start': str(valid.date.min().date()),
            'valid_end': str(valid.date.max().date()), 'test_start': f'{year}-01-01',
            'test_end': f'{end-1}-12-31'})
    return records


def render_lstm(events, config_path='configs/author_native.yaml'):
    """Return the current candidate's loss/Sharpe figure, with instrument/date labels."""
    import textwrap
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    epochs = [e for e in events if e.get('kind') == 'epoch']
    if not epochs:
        return None
    last = epochs[-1]
    current = [e for e in epochs if e['strategy']==last['strategy'] and e['fold']==last['fold']]
    reset = max([i for i,e in enumerate(current) if e['epoch']==1] or [0])
    current = current[reset:]
    records = lstm_context(last['strategy'], int(last['fold']), config_path)
    pool = ', '.join(r['instrument'] for r in records)
    period = records[0]['test_start'] + ' to ' + records[0]['test_end'] if records else 'No eligible instruments'
    fig = Figure(figsize=(13, 7))
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)
    left.plot([e['epoch'] for e in current], [e['loss'] for e in current], color='#1565c0')
    left.set_title('Training loss | current candidate')
    right.plot([e['epoch'] for e in current], [e['validation_sharpe'] for e in current], color='#168a70')
    right.set_title('Validation Sharpe | current candidate')
    for ax in (left, right):
        ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    title = f"{last['strategy']} | test {period} | epoch {last['epoch']}\nShared model | {len(records)} instruments\n" + '\n'.join(textwrap.wrap(pool, 100))
    fig.suptitle(title, fontsize=11)
    if records:
        details = f"Training input dates: {min(r['train_start'] for r in records)} to {max(r['train_end'] for r in records)}\nValidation input dates: {min(r['valid_start'] for r in records)} to {max(r['valid_end'] for r in records)}\nChronological 90/10 split per instrument; ranges show the union of instruments. Exact dates are printed in the log."
        fig.text(.07, .035, details, fontsize=9)
    fig.subplots_adjust(top=.70, bottom=.23, wspace=.25)
    buffer = BytesIO(); fig.savefig(buffer, format='png', dpi=110)
    return buffer.getvalue()


def monitor_text(interval=3, process=None, since='', once=False, config_path='configs/author_native.yaml'):
    """Append plain-text progress lines to a scrollable notebook output."""
    import csv
    from momentum_lab.author_notebook import read_events, check_backend_failure
    import yaml
    cfg = yaml.safe_load((ROOT / config_path).read_text(encoding='utf-8'))
    output = ROOT / cfg['results_directory']
    seen_rows, seen_events = {}, set()
    file_cursors = {}
    chart_handle = None
    last_epoch = None
    last_context = None
    print('CPD / LSTM progress log | saved rows are reusable | refresh every 3 seconds', flush=True)
    heartbeat = time.monotonic()
    try:
        while True:
            for path in sorted((ROOT / cfg['data_directory'] / 'cpd').glob('*/*.csv')):
                offset, count, last = file_cursors.get(str(path), (0, 0, None))
                size = path.stat().st_size
                if size == offset:
                    continue
                if size < offset:
                    offset, count, last = 0, 0, None
                with path.open('rb') as stream:
                    stream.seek(offset)
                    data = stream.read()
                end = data.rfind(b'\n') + 1
                if not end:
                    continue
                lines = data[:end].decode('utf-8').splitlines()
                if not offset:
                    lines = lines[1:]
                for row in csv.DictReader(lines, fieldnames=['date','t','cp_location','cp_location_norm','cp_score']):
                    last = row
                    count += 1
                file_cursors[str(path)] = (offset+end, count, last)
                if not last or seen_rows.get(str(path)) == count:
                    continue
                previous = seen_rows.get(str(path), 0)
                seen_rows[str(path)] = count
                print(f"[{datetime.now(timezone.utc):%H:%M:%S} UTC] CPD | instrument={path.stem} | window={path.parent.name}d | saved={count:,} | new={count-previous:,} | through={last['date']} | updated={datetime.fromtimestamp(path.stat().st_mtime, timezone.utc):%Y-%m-%d %H:%M:%S} UTC", flush=True)
                heartbeat = time.monotonic()
            events = read_events(output / 'events')
            epochs = [e for e in events if e.get('kind') == 'epoch' and e.get('utc', '') >= since]
            if epochs and epochs[-1] != last_epoch:
                last_epoch = epochs[-1]
                key = (last_epoch['strategy'], last_epoch['fold'])
                if key != last_context:
                    last_context = key
                    print('LSTM instrument/time-period details (shared model):', flush=True)
                    for record in lstm_context(key[0], int(key[1]), config_path):
                        print(json.dumps(record), flush=True)
                from IPython.display import Image, display
                chart = Image(data=render_lstm(epochs, config_path))
                if chart_handle is None:
                    chart_handle = display(chart, display_id=True)
                else:
                    chart_handle.update(chart)
            for event in events:
                identity = json.dumps(event, sort_keys=True)
                if identity in seen_events or event.get('utc', '') < since:
                    continue
                seen_events.add(identity)
                if event.get('kind') in ('preflight','features','cpd_runtime','cpd','cpd_failure','epoch','candidate_completed','failed','complete','models_started','model_started','cpd_started'):
                    print(json.dumps(event, ensure_ascii=True), flush=True)
            if once:
                for path in (output / 'events').glob('*.status.json'):
                    event = json.loads(path.read_text(encoding='utf-8'))
                    if event.get('kind') == 'failed':
                        print('BACKEND FAILED | ' + event.get('error', '') + ' | ' + event.get('message', '') + ' | Saved rows retained.', flush=True)
                return
            check_backend_failure(output, since)
            if any(e.get('kind') == 'complete' and e.get('utc', '') >= since for e in events):
                print('Author experiment complete. Final notebook report cells can now run.', flush=True)
                return
            if process is not None and process.poll() is not None:
                if process.returncode:
                    raise RuntimeError(f'Backend exit code: {process.returncode}. Read {output / "run.log"}. Saved progress retained.')
                print('Author run complete.', flush=True)
                return
            if time.monotonic() - heartbeat >= 30:
                print(f'[{datetime.now(timezone.utc):%H:%M:%S} UTC] WAITING | No newly saved CPD rows in the last 30 seconds. This is not evidence of compute progress.', flush=True)
                heartbeat = time.monotonic()
            time.sleep(interval)
    except KeyboardInterrupt:
        print('Display stopped. No backend process was terminated.', flush=True)


def complete_csv_bytes(path):
    data = path.read_bytes()
    return data[:data.rfind(b'\n') + 1]


def snapshot():
    """Copy complete CSV rows only while the original workers keep writing."""
    import hashlib
    output = ROOT / 'results/author_native'
    folder = output / 'progress_snapshots' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True)
    files = {}
    sources = list((ROOT / 'data/processed/author_native/cpd').glob('*/*.csv'))
    sources += [output / 'run_identity.json', ROOT / 'configs/author_native.yaml']
    for source in sources:
        if not source.exists():
            continue
        relative = source.relative_to(ROOT)
        data = complete_csv_bytes(source) if source.suffix == '.csv' else source.read_bytes()
        destination = folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        files[str(relative)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (folder / 'manifest.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'files': files}, indent=2), encoding='utf-8')
    return folder


def render_progress():
    import pandas as pd
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from momentum_lab.author_notebook import read_events
    base = ROOT / 'data/processed/author_native'
    output = ROOT / 'results/author_native'
    inputs = pd.read_parquet(base / 'cpd_inputs.parquet')
    counts = inputs.groupby('ticker').size().sort_index()
    windows = [10, 21, 63, 126, 252]
    progress = np.zeros((len(counts), len(windows)))
    saved = expected = 0
    frames = []
    for i, (ticker, count) in enumerate(counts.items()):
        for j, window in enumerate(windows):
            total = max(int(count) - window - 1, 0)
            expected += total
            path = base / 'cpd' / str(window) / f'{ticker}.csv'
            if not path.exists():
                continue
            data = pd.read_csv(BytesIO(complete_csv_bytes(path)))
            rows = data['t'].nunique()
            saved += rows
            progress[i, j] = rows / total if total else 1
            if rows:
                frames.append((ticker, window, rows, total, data))
    events = read_events(output / 'events')
    statuses = []
    for path in (output / 'events').glob('*.status.json'):
        try:
            statuses.append(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            pass
    terminal = sorted([e for e in statuses if e.get('kind') in ('failed', 'complete')], key=lambda e:e.get('utc',''))
    epochs = [e for e in events if e.get('kind') == 'epoch']
    fig = Figure(figsize=(15, 10), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    last_update = events[-1].get('utc', 'Unavailable') if events else 'Unavailable'
    fig.suptitle(f'Author-source progress | Viewed {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC\nGP-CPD: {saved:,} / {expected:,} saved daily results ({saved/max(expected,1):.2%}) | Last event: {last_update}')
    heat = axes[0, 0].imshow(progress, aspect='auto', vmin=0, vmax=1, cmap='YlGnBu')
    axes[0, 0].set_xticks(range(5), windows)
    axes[0, 0].set_yticks(range(len(counts)), counts.index, fontsize=6)
    axes[0, 0].set_title('All instruments / CPD windows')
    axes[0, 0].set_xlabel('Lookback window (trading days)')
    fig.colorbar(heat, ax=axes[0, 0], label='Saved fraction')
    active = [(t, w, n, total, d) for t, w, n, total, d in frames if n < total][-8:]
    axes[0, 1].set_title('Partial CPD jobs (CSV rows saved continuously)')
    if active:
        labels = [f'{t} / {w}d: {n:,}/{total:,}' for t, w, n, total, d in active]
        axes[0, 1].barh(labels, [n/total for t,w,n,total,d in active], color='#168a70')
        axes[0, 1].set_xlim(0, 1)
    for ticker, window, rows, total, data in active[:4] or frames[-4:]:
        axes[1, 0].plot(pd.to_datetime(data['date']), data['cp_score'], label=f'{ticker} / {window}d', linewidth=.8)
    axes[1, 0].set_title('Computed change-point scores (available dates only)')
    axes[1, 0].set_ylabel('CP score')
    if frames:
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].tick_params(axis='x', rotation=20)
    ax = axes[1, 1]
    if epochs:
        last = epochs[-1]
        current = [e for e in epochs if e['strategy']==last['strategy'] and e['fold']==last['fold']]
        reset = max([i for i,e in enumerate(current) if e['epoch']==1] or [0])
        current = current[reset:]
        ax.plot([e['epoch'] for e in current], [e['validation_sharpe'] for e in current])
        year = int(last['fold'])
        end = min(year+5, 2021)
        panel = pd.read_parquet(base / 'features.parquet')
        panel = panel[(panel.year < year) & (panel.published_first_test <= year)]
        sizes = panel.groupby('ticker').size()
        tickers = sizes[sizes * .1 >= 63].index.tolist()
        ax.set_title(f"{last['strategy']} | test {year}-01-01 to {end-1}-12-31")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Sharpe')
        import textwrap
        context = f'Shared LSTM across {len(tickers)} eligible instruments (not one model per instrument).\nPool: ' + ', '.join(tickers) + f'\nPre-test data: 1990 to {year-1}; chronological 90% training / 10% validation per instrument. Exact validation dates vary by instrument.'
        fig.text(.51, .01, '\n'.join(textwrap.wrap(context, 100, replace_whitespace=False)), fontsize=8)
    else:
        ax.axis('off')
        state = ('BACKEND FAILED: ' + terminal[-1].get('error', terminal[-1]['kind'])) if terminal else 'Latest recorded stage: GPflow preprocessing.'
        ax.text(.05, .65, 'LSTM has not started.\n' + state + '\n\nSaved GPflow CSV rows can be resumed.\nPrevious SciPy/Numba CPD caches are excluded.\n\nDuring LSTM: instrument pool, test period,\ntraining/validation split and epoch curves appear here.\n\nClosing this monitor does not stop computation.', transform=ax.transAxes)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=120)
    return buffer.getvalue()


def monitor_progress(interval=10):
    from IPython.display import Image, display
    print('Read-only monitor. No new training process is launched. Refresh: 10 seconds.')
    print('Recovery snapshot:', snapshot())
    handle = None
    try:
        while True:
            image = Image(data=render_progress())
            if handle is None:
                handle = display(image, display_id=True)
            else:
                handle.update(image)
            from momentum_lab.author_notebook import check_backend_failure
            check_backend_failure(ROOT / 'results/author_native')
            time.sleep(interval)
    except KeyboardInterrupt:
        print('Monitor stopped. Background computation is unchanged.')

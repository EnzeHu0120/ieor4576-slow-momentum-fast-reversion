"""Author-source experiment orchestration, isolated from the ported experiment."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from .author_compat import ROOT, load_author, provenance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    from .checkpoint_io import replace_checkpoint
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    replace_checkpoint(temporary, path)


def configuration(path='configs/author_native.yaml'):
    import yaml
    return yaml.safe_load((ROOT / path).read_text(encoding='utf-8'))


def cpd_job_complete(data, ticker, window, fingerprint):
    receipt = Path(data) / 'cpd_receipts' / str(window) / f'{ticker}.json'
    if not receipt.exists() or not (Path(data) / 'cpd' / str(window) / f'{ticker}.csv').exists():
        return False
    saved = json.loads(receipt.read_text(encoding='utf-8'))
    return saved.get('fingerprint') == fingerprint and saved.get('rows') == saved.get('expected')


def write_data_audit(cfg):
    import pandas as pd
    from .author_protocol import test_intervals
    data = ROOT / cfg['data_directory']
    panel = pd.read_parquet(data / 'features.parquet')
    universe = pd.read_csv(ROOT / 'configs/universe.csv')
    rows = []
    for ticker, values in panel.groupby('ticker'):
        counts = [(year, len(values[values.year < year])) for year, _ in test_intervals(cfg)]
        eligible = [year for year, count in counts if count * (1-cfg['train_valid_ratio']) >= cfg['sequence_length']
                    and year >= int(values.published_first_test.iloc[0])]
        rows.append({'ticker':ticker, 'first_feature_date':str(values.date.min()),
            'last_feature_date':str(values.date.max()), 'feature_rows':len(values),
            'first_eligible_test_year':min(eligible) if eligible else None})
    output = ROOT / cfg['results_directory'] / 'tables'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).merge(universe, on='ticker', how='left').to_csv(output / 'data_panel_audit.csv', index=False)
    write_json(output / 'runtime_data_manifest.json', {
        'files':{str(path.relative_to(ROOT)):digest(path) for path in
            [data/'features.parquet',data/'cpd_inputs.parquet',ROOT/'configs/universe.csv']},
        'limitations':['Pinnacle original prices, roll calendar and missing-data masks are unavailable.',
            'WRDS universe mapping and same-contract return construction are retained; equivalence to Pinnacle is not claimed.']})


def emit(output, kind, **values):
    folder = Path(output) / 'events'
    folder.mkdir(parents=True, exist_ok=True)
    event = {'kind': kind, 'utc': datetime.now(timezone.utc).isoformat(), **values}
    # Each process owns its own file; stdout is never parsed as JSON.
    with (folder / f'{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, default=str) + '\n')
    write_json(folder / f'{os.getpid()}.status.json', event)


def preflight(require_gpu=True):
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    import tensorflow as tf
    load_author()
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if require_gpu and not gpus:
        raise RuntimeError('TensorFlow GPU unavailable. Use the verified WSL author environment; CPU fallback is disabled.')
    # Exercise the author LSTM and gradient path, not just CUDA device discovery.
    import keras_tuner as kt
    from mom_trans.deep_momentum_network import LstmDeepMomentumNetworkModel
    from settings.fixed_params import MODLE_PARAMS
    import tempfile
    with tempfile.TemporaryDirectory(prefix='paper67-preflight-') as folder:
        params = {**MODLE_PARAMS, 'architecture': 'LSTM', 'total_time_steps': 63, 'input_size': 8, 'output_size': 1,
                  'multiprocessing_workers': 1, 'random_search_iterations': 1}
        dmn = LstmDeepMomentumNetworkModel('preflight', folder, **params)
        model = dmn.model_builder(kt.HyperParameters())
        x = tf.random.normal((4, 63, 8))
        y = tf.random.normal((4, 63, 1)) * .01
        with tf.GradientTape() as tape:
            loss = model.loss(y, model(x, training=True))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if require_gpu and 'GPU' not in model.trainable_variables[0].device.upper():
            raise RuntimeError('Author LSTM was not placed on GPU.')
        model.save_weights(str(Path(folder) / 'roundtrip'))
        model.load_weights(str(Path(folder) / 'roundtrip'))
        result = {'tensorflow': tf.__version__, 'gpu': [d.name for d in gpus],
                  'variable_device': model.trainable_variables[0].device,
                  'lstm_gradient_and_checkpoint': True}
        tf.keras.backend.clear_session()
        return result


def prepare_features(cfg):
    import numpy as np
    import pandas as pd
    load_author()
    from mom_trans.data_prep import deep_momentum_strategy_features
    data = ROOT / cfg['data_directory']
    data.mkdir(parents=True, exist_ok=True)
    # Reuse audited WRDS same-contract returns, not ported CPD/model outputs.
    old = pd.read_parquet(ROOT / 'data/processed/features.parquet')
    if cfg.get('tickers'):
        missing = set(cfg['tickers']) - set(old.ticker.unique())
        if missing:
            raise RuntimeError(f'Requested instruments are unavailable: {sorted(missing)}')
        old = old[old.ticker.isin(cfg['tickers'])]
    frames, inputs, comparisons = [], [], []
    columns = ['norm_daily_return', 'norm_monthly_return', 'norm_quarterly_return',
               'norm_biannual_return', 'norm_annual_return', 'macd_8_24', 'macd_16_48',
               'macd_32_96', 'daily_vol', 'target_returns']
    for ticker, metadata in old.groupby('ticker', sort=True):
        rolled = pd.read_parquet(ROOT / f'data/interim/{ticker}_rolls.parquet')
        raw = rolled.raw_return.dropna()
        close = 100 * (1 + raw).cumprod()
        fresh = deep_momentum_strategy_features(pd.DataFrame({'close': close}))
        fresh['ticker'] = ticker
        # Retain only pre-audited, contiguous next-observation labels and universe metadata.
        shared = metadata.set_index('date', drop=False).index.intersection(fresh.index)
        base = metadata.set_index('date', drop=False).loc[shared].copy()
        for col in columns:
            reference = fresh.loc[shared, col].to_numpy()
            comparisons.append({'ticker': ticker, 'column': col,
                                'max_abs_difference': float(np.max(np.abs(base[col].to_numpy() - reference)))})
            base[col] = reference
        base.index.name = 'observation_date'
        frames.append(base)
        # Author CPD example uses raw close returns, not winsorized LSTM labels.
        inputs.append(pd.DataFrame({'date': raw.index, 'ticker': ticker, 'daily_returns': raw.to_numpy()}))
    panel = pd.concat(frames).sort_values(['ticker', 'date'])
    panel.to_parquet(data / 'features.parquet')
    pd.concat(inputs, ignore_index=True).to_parquet(data / 'cpd_inputs.parquet')
    out = ROOT / cfg['results_directory']
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparisons).to_csv(out / 'tables/feature_revalidation.csv', index=False)
    return panel


def compute_cpd(job):
    ticker, lbw, cfg, fingerprint, limit = job
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPflow preprocessing on CPU, LSTM owns GPU.
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    import numpy as np
    import pandas as pd
    load_author()
    from .author_cpd_runtime import install
    install()
    from mom_trans.changepoint_detection import changepoint_loc_and_score
    data = ROOT / cfg['data_directory']
    output = ROOT / cfg['results_directory']
    path = data / 'cpd' / str(lbw) / f'{ticker}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.read_parquet(data / 'cpd_inputs.parquet')
    returns = returns[returns.ticker.eq(ticker)].sort_values('date').reset_index(drop=True)
    columns = ['date', 't', 'cp_location', 'cp_location_norm', 'cp_score']
    completed = set()
    if path.exists():
        existing = pd.read_csv(path)
        # Only complete CSV rows are considered committed.
        existing = existing.dropna(subset=['date', 't'])
        completed = set(existing.t.astype(int))
    else:
        with path.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(columns)
    expected = max(0, len(returns) - lbw - 1)
    smoke_start = None
    if limit:
        smoke_features = pd.read_parquet(data / 'features.parquet', columns=['ticker','date'])
        smoke_start = smoke_features.loc[smoke_features.ticker.eq(ticker),'date'].min()
    attempts = 0
    batch_rows = max(1, int(os.environ.get('PAPER67_CPD_BATCH_ROWS', '500')))
    emit(output, 'cpd', ticker=ticker, window=lbw, completed=len(completed), total=expected)
    with path.open('a', newline='', encoding='utf-8', buffering=1) as f:
        writer = csv.writer(f)
        for end in range(lbw + 1, len(returns)):
            t = end - 1
            if smoke_start is not None and returns.date.iloc[t] < smoke_start:
                continue
            if t in completed:
                continue
            window = returns.iloc[end-lbw-1:end][['date', 'daily_returns']].copy()
            window['X'] = window.index.astype(float)
            window = window.rename(columns={'daily_returns': 'Y'})
            try:
                score, location, relative, _, _ = changepoint_loc_and_score(window)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                # Same NA convention as the author's run_module; fill causally on read.
                score = location = relative = np.nan
                emit(output, 'cpd_failure', ticker=ticker, window=lbw, t=t, error=type(exc).__name__)
            writer.writerow([str(returns.date.iloc[t].date()), t, location, relative, score])
            attempts += 1
            completed.add(t)
            if attempts % 10 == 0 or len(completed) == expected:
                emit(output, 'cpd', ticker=ticker, window=lbw, completed=len(completed), total=expected)
            if limit and attempts >= limit:
                return ticker, lbw, 'smoke'
            if not limit and attempts >= batch_rows and len(completed) < expected:
                emit(output, 'cpd', ticker=ticker, window=lbw, completed=len(completed), total=expected)
                return ticker, lbw, 'partial'
    write_json(data / 'cpd_receipts' / str(lbw) / f'{ticker}.json',
               {'fingerprint': fingerprint, 'rows': len(completed), 'expected': expected})
    return ticker, lbw, 'complete'


def load_panel(cfg, lbw):
    import pandas as pd
    from mom_trans.data_prep import include_changepoint_features
    path = ROOT / cfg['data_directory']
    frame = pd.read_parquet(path / 'features.parquet')
    if cfg.get('tickers'):
        frame = frame[frame.ticker.isin(cfg['tickers'])]
    if lbw is not None:
        frame = include_changepoint_features(frame, str(path / 'cpd' / str(lbw)), lbw)
    frame.index = pd.to_datetime(frame.date)
    frame.index.name = 'observation_date'
    return frame.sort_values(['ticker', 'date'])


def run_model(cfg, lbw, year, end, fingerprint, smoke=False):
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    load_author()
    from mom_trans.model_inputs import ModelFeatures
    from mom_trans.deep_momentum_network import LstmDeepMomentumNetworkModel, SharpeValidationLoss
    from settings.fixed_params import MODLE_PARAMS
    name = 'LSTM' if lbw is None else f'LSTM-CPD-{lbw}'
    output = ROOT / cfg['results_directory']
    dest = output / 'models' / name / str(year)
    receipt = dest / 'receipt.json'
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding='utf-8'))
        if old.get('fingerprint') != fingerprint:
            if not smoke:
                raise RuntimeError(f'Configuration/source mismatch in {dest}; preserve this run before changing settings.')
        else:
            return old
    dest.mkdir(parents=True, exist_ok=True)
    raw = load_panel(cfg, lbw)
    raw = raw[raw.published_first_test.le(year)].copy()
    counts = raw[raw.year.lt(year)].groupby('ticker').size()
    eligible = counts[counts * (1-cfg['train_valid_ratio']) >= cfg['sequence_length']].index
    raw = raw[raw.ticker.isin(eligible)]
    if smoke:
        raw = raw[raw.ticker.isin(list(eligible)[:2])]
    # Author formatter owns padding, active flags, chronological splits and scaling defaults.
    features = ModelFeatures(raw, cfg['sequence_length'], start_boundary=cfg['start'],
        test_boundary=year, test_end=end, changepoint_lbws=None if lbw is None else [lbw],
        train_valid_ratio=cfg['train_valid_ratio'], split_tickers_individually=True,
        time_features=False, lags=None)
    context = []
    dated = raw.dropna()
    dated = dated[(dated.year >= cfg['start']) & (dated.year < year)]
    for ticker, rows in dated.groupby('ticker'):
        split = int(cfg['train_valid_ratio'] * len(rows))
        train, valid = rows.iloc[:split], rows.iloc[split:]
        if len(train) and len(valid):
            context.append({'instrument':ticker,
                'train_start':str(train.date.min().date()), 'train_end':str(train.date.max().date()),
                'valid_start':str(valid.date.min().date()), 'valid_end':str(valid.date.max().date()),
                'test_start':f'{year}-01-01', 'test_end':f'{end-1}-12-31'})
    write_json(dest / 'training_context.json', context)
    params = {**MODLE_PARAMS, **features.input_params, 'architecture': 'LSTM',
        'total_time_steps': cfg['sequence_length'], 'num_epochs': cfg['max_epochs'],
        'random_search_iterations': cfg['random_search_trials'],
        'early_stopping_patience': cfg['patience'], 'multiprocessing_workers': 1,
        'evaluate_diversified_val_sharpe': cfg['evaluate_diversified_val_sharpe']}
    tf.keras.utils.set_random_seed(cfg['seed'] + year)
    emit(output, 'model_started', strategy=name, fold=year, test_end=end,
         instruments=sorted(raw.ticker.unique().tolist()), training_start=cfg['start'],
         train_valid_ratio=cfg['train_valid_ratio'])
    # Instrument existing author callbacks, without changing their objective/selection.
    original = SharpeValidationLoss.on_epoch_end
    def on_epoch_end(self, epoch, logs=None):
        original(self, epoch, logs)
        emit(output, 'epoch', strategy=name, fold=year, epoch=epoch+1,
             loss=float(logs.get('loss', np.nan)), validation_sharpe=float(logs['sharpe']))
    SharpeValidationLoss.on_epoch_end = on_epoch_end
    try:
        # Frequent Keras checkpoint updates stay on the Linux filesystem, away
        # from Windows sharing locks. Keep this persistent across notebook restarts.
        scratch = Path(os.environ.get('PAPER67_CHECKPOINT_SCRATCH', str(Path.home()/'.cache/paper67-author')))
        scratch = scratch / fingerprint / name / str(year)
        scratch.mkdir(parents=True, exist_ok=True)
        write_json(dest / 'runtime_checkpoint_location.json', {'linux_scratch': str(scratch)})
        from .author_protocol import seeded_model
        author_module = importlib.import_module('mom_trans.deep_momentum_network')
        model_seed = cfg['seed'] + year
        dmn = seeded_model(author_module, name, str(scratch / 'tuner'), params, model_seed)
        write_json(dest / 'reproducibility.json', {'tensorflow_seed':model_seed,
            'tuner_seed':model_seed, 'fold':year, 'test_end_exclusive':end,
            'source_commit':cfg['author_commit'],
            'note':'Controlled new experiment; original paper trial draws/weights are unavailable.'})
        trial_end = dmn.tuner.on_trial_end
        def on_trial_end(trial):
            trial_end(trial)
            count = sum(t.status == 'COMPLETED' for t in dmn.tuner.oracle.trials.values())
            emit(output, 'candidate_completed', strategy=name, fold=year,
                 trial_id=trial.trial_id, completed_in_search=count)
        dmn.tuner.on_trial_end = on_trial_end
        best_hp, model = dmn.hyperparameter_search(features.train, features.valid)
        validation = float(dmn.evaluate(features.valid, model))
        positions, _ = dmn.get_positions(features.test_sliding, model, sliding_window=True,
                                        years_geq=year, years_lt=end)
        positions.to_parquet(dest / 'positions.parquet', index=False)
        # A unique final save avoids replacing an open .pt checkpoint on Windows.
        model.save_weights(str(scratch / 'best.weights.h5'))
        shutil.copy2(scratch / 'best.weights.h5', dest / 'best.weights.h5')
        trials = {k: {'status': t.status, 'score': t.score, 'parameters': t.hyperparameters.values}
                  for k, t in dmn.tuner.oracle.trials.items()}
        successful = sum(t['status'] == 'COMPLETED' for t in trials.values())
        if successful != cfg['random_search_trials']:
            raise RuntimeError(f'{name}/{year}: {successful} completed trials; expected {cfg["random_search_trials"]}.')
        shutil.copytree(scratch / 'tuner', dest / 'tuner', dirs_exist_ok=True)
        result = {'fingerprint': fingerprint, 'strategy': name, 'fold': year, 'end': end,
                  'validation_sharpe': validation, 'best_hp': best_hp, 'trials': trials,
                  'rows': len(positions), 'eligible_tickers': sorted(raw.ticker.unique().tolist()),
                  'author_modules': ['ModelFeatures', 'LstmDeepMomentumNetworkModel', 'SharpeValidationLoss']}
        write_json(receipt, result)
        emit(output, 'model_completed', strategy=name, fold=year, trials=successful)
        return result
    finally:
        SharpeValidationLoss.on_epoch_end = original
        tf.keras.backend.clear_session()


def evaluate(cfg, receipts):
    import numpy as np
    import pandas as pd
    from mom_trans.classical_strategies import calc_performance_metrics, calc_net_returns, MACDStrategy
    output = ROOT / cfg['results_directory']
    rows, cost_rows, professor_rows, panels = [], [], [], []
    raw = pd.read_parquet(ROOT / cfg['data_directory'] / 'features.parquet')
    selection = {}
    for r in receipts:
        if r['strategy'] != 'LSTM':
            current = selection.get(r['fold'])
            if current is None or r['validation_sharpe'] > current['validation_sharpe']:
                selection[r['fold']] = r
    for r in receipts:
        p = pd.read_parquet(output / 'models' / r['strategy'] / str(r['fold']) / 'positions.parquet')
        p = p.merge(raw.reset_index(drop=True).rename(columns={'ticker':'identifier', 'date':'time'}),
                    on=['identifier', 'time'], validate='one_to_one', suffixes=('', '_feature'))
        names = [r['strategy']]
        if selection.get(r['fold']) is r:
            names.append('LSTM-CPD-Optimised')
        versions = [(name, p.copy()) for name in names]
        if r['strategy'] == 'LSTM':
            positions = {'Long Only': np.ones(len(p)),
                         'MACD': MACDStrategy.scale_signal(p[['macd_8_24','macd_16_48','macd_32_96']].mean(axis=1)),
                         'TSMOM w=0': np.sign(p.norm_annual_return),
                         'TSMOM w=0.5': .5*(np.sign(p.norm_annual_return)+np.sign(p.norm_monthly_return)),
                         'TSMOM w=1': np.sign(p.norm_monthly_return)}
            for name, position in positions.items():
                q = p.copy()
                q['position'] = position
                q['captured_returns'] = q.position*q.returns
                versions.append((name, q))
        for name, q in versions:
            metrics = calc_performance_metrics(q.set_index('time'))
            rows.append({'strategy': name, 'fold': r['fold'], **metrics})
            panels.append((name, r['fold'], q))
            original_costs = calc_net_returns(q, [0.5, 1, 2, 2.5, 3, 5])
            for bp in [0.5, 1, 2, 2.5, 3, 5]:
                costed = original_costs.copy()
                costed['captured_returns'] = costed[f'captured_returns_{str(bp).replace(".", "_")}_bps']
                cost_rows.append({'strategy':name, 'fold':r['fold'], 'bps':bp,
                                  **calc_performance_metrics(costed.set_index('time'))})
            # Professor-requested accounting is a separate sensitivity, not author loss/training.
            q = q.sort_values(['identifier','time']).copy()
            q['exposure'] = q.position*.15/(q.daily_vol*np.sqrt(252))
            previous = q.groupby('identifier').exposure.shift(1).fillna(0.)
            old_contract = q.groupby('identifier').decision_contract.shift(1)
            rolled = q.decision_contract.ne(old_contract) & previous.ne(0.)
            turnover = (q.exposure-previous).abs().where(~rolled,q.exposure.abs()+previous.abs())
            last = q.groupby('identifier').tail(1).index
            turnover.loc[last] += q.loc[last,'exposure'].abs()  # final liquidation
            for bp in [0, .5, 1, 2, 2.5, 3, 5]:
                net = q.copy()
                net['captured_returns'] = q.exposure*q.raw_target_return - bp*1e-4*turnover
                professor_rows.append({'strategy':name, 'fold':r['fold'], 'bps':bp,
                                      **calc_performance_metrics(net.set_index('time'))})
    table = pd.DataFrame(rows)
    table.to_csv(output / 'tables/exhibits_by_fold.csv', index=False)
    mean = table.groupby('strategy').mean(numeric_only=True).drop(columns='fold')
    mean.to_csv(output / 'tables/exhibit_3.csv')
    pd.DataFrame(cost_rows).to_csv(output / 'tables/author_transaction_costs.csv', index=False)
    pd.DataFrame(professor_rows).to_csv(output / 'tables/professor_roll_cost_sensitivity.csv', index=False)
    scales = .15 / mean.annual_volatility
    rescaled = []
    for name, fold, p in panels:
        q = p.copy()
        q['captured_returns'] *= scales[name]
        rescaled.append({'strategy':name, 'fold':fold, **calc_performance_metrics(q.set_index('time'))})
    pd.DataFrame(rescaled).groupby('strategy').mean(numeric_only=True).drop(columns='fold').to_csv(output / 'tables/exhibit_4.csv')
    # Show the strict 25-year interpretation alongside the inclusive-end-2020 main table.
    strict = []
    for name, fold, p in panels:
        dates = pd.to_datetime(p.time)
        q = p[(dates >= '1995-01-01') & (dates < '2020-01-01')]
        if len(q):
            strict.append({'strategy':name,'fold':fold,**calc_performance_metrics(q.set_index('time'))})
    if strict:
        pd.DataFrame(strict).groupby('strategy').mean(numeric_only=True).drop(columns='fold').to_csv(output / 'tables/strict_1995_2019_sensitivity.csv')
    from .author_protocol import test_intervals
    write_json(output / 'tables/evaluation_protocol.json', {
        'start_inclusive':f"{min(cfg['test_boundaries'])}-01-01", 'end_exclusive':f"{cfg['last_test_end']}-01-01",
        'test_intervals':test_intervals(cfg), 'aggregation':'equal mean of per-fold metrics',
        'last_fold_note':'End year is exclusive in code. Main full protocol includes all of 2020 in the 2015-start fold (6 calendar years); strict 1995-2019 sensitivity is saved separately.',
        'costs':'Paper-style costs and professor roll-cost extension are separate tables.'})
    published = pd.read_csv(ROOT / 'configs/published_exhibits.csv')
    # Save author-native results independently; never insert published values into predictions.
    published.to_csv(output / 'tables/published_reference.csv', index=False)
    pd.DataFrame([{'fold': y, 'strategy': r['strategy'], 'validation_sharpe': r['validation_sharpe']}
                  for y, r in selection.items()]).to_csv(output / 'tables/selected_cpd_windows.csv', index=False)
    report = ROOT / cfg['report_directory']
    report.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': cfg['protocol'], 'author_commit': cfg['author_commit'],
                'instruments':sorted({ticker for receipt in receipts for ticker in receipt['eligible_tickers']}),
                'comparison_scope':'Five-instrument subset validation against a 50-instrument published reference; not a matched-universe reproduction.' if cfg.get('tickers') else 'Full configured WRDS universe.',
                'completed_experiments': len(receipts),
                'completed_trials': sum(len(r['trials']) for r in receipts),
                'expected_experiments': (len(cfg['cpd_windows'])+1)*len(cfg['test_boundaries']),
                'source': provenance(), 'data_note': 'WRDS same-contract returns; Pinnacle proprietary data not supplied.'}
    write_json(report / 'execution_manifest.json', manifest)
    return manifest


def run(config_path='configs/author_native.yaml', smoke=False):
    from filelock import FileLock
    import multiprocessing
    cfg = configuration(config_path)
    if smoke:
        cfg = {**cfg, 'protocol': 'author_source_SMOKE_ONLY',
               'data_directory': 'data/processed/author_smoke', 'results_directory': 'results/author_smoke',
               'report_directory': 'report/author_smoke', 'cpd_windows': [10],
               'test_boundaries': [2000], 'last_test_end': 2001, 'random_search_trials': 1,
               'max_epochs': 2, 'patience': 1}
    output = ROOT / cfg['results_directory']
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / 'run.lock'), timeout=0):
        write_json(output / 'run_session.json', {'pid':os.getpid(), 'started_at':datetime.now(timezone.utc).isoformat()})
        try:
            emit(output, 'preflight')
            readiness = preflight(cfg['require_gpu'])
            write_json(output / 'preflight.json', readiness)
            validation_path = ROOT / 'resources/local/author_graph_validation.json'
            validation = json.loads(validation_path.read_text(encoding='utf-8'))
            if not validation.get('validated') or validation.get('runtime_sha256') != digest(Path(__file__).with_name('author_cpd_runtime.py')):
                raise RuntimeError('GPflow runtime acceleration has not passed numerical equivalence validation for this exact source.')
            source = provenance()
            feature_source = ROOT / 'data/processed/features.parquet'
            identity = {'config': cfg, 'author': source,
                'input_features_sha256': digest(feature_source),
                'framework_versions': {name:importlib.metadata.version(name) for name in
                    ['tensorflow','tf-keras','gpflow','keras-tuner','numpy','pandas','scipy']},
                'harness_hashes': {name:digest(Path(__file__).with_name(name))
                    for name in ['author_native.py','author_compat.py','author_cpd_runtime.py','author_protocol.py']}}
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            meta = output / 'run_identity.json'
            if not smoke and meta.exists() and json.loads(meta.read_text())['fingerprint'] != fingerprint:
                previous = json.loads(meta.read_text())
                from .author_protocol import cpd_only_config_transition
                config_ok = previous['config'] == cfg or cpd_only_config_transition(previous['config'], cfg)
                legacy_fingerprints = []
                for old_hash in ['c19d64cba1ed855a32a574f137122ade010d9f72f370bb921bd9c3f992ff6441',
                                 '187a2d5708763b77e707b1f97d96c1e676b561525ae9a11d31393c749d075dfb',
                                 '6e51d866b1e9169819b9edf114c4767fa7318e603cc8666aa6f0b8c69b2d4519',
                                 '43cc654c8929d2bbdcf05474d37ec70c166c3d973ce32eb785dccffe83ce5339']:
                    legacy = {**identity, 'config':previous['config'], 'harness_hashes': {
                        'author_native.py': old_hash,
                        'author_compat.py': identity['harness_hashes']['author_compat.py']}}
                    legacy_fingerprints.append(hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest())
                    legacy['harness_hashes']['author_cpd_runtime.py'] = identity['harness_hashes']['author_cpd_runtime.py']
                    legacy_fingerprints.append(hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest())
                    legacy['harness_hashes']['author_protocol.py'] = identity['harness_hashes']['author_protocol.py']
                    legacy_fingerprints.append(hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest())
                if not config_ok or previous['fingerprint'] not in legacy_fingerprints or list((output / 'models').glob('*/*/receipt.json')):
                    raise RuntimeError('Existing author run uses different source/data/settings; use a separate run directory.')
                write_json(output / 'memory_fix_migration.json', {'previous_identity': previous,
                    'new_fingerprint': fingerprint, 'reason': 'CPD-only run: numerically validated graph reuse and bounded workers; identical author code, data, settings and framework versions verified.', 'equivalence_validation': validation})
            write_json(meta, {'fingerprint': fingerprint, 'config': cfg, 'provenance': source})
            emit(output, 'features')
            data = ROOT / cfg['data_directory']
            if not (data / 'features.parquet').exists():
                prepare_features(cfg)
            write_data_audit(cfg)
            import pandas as pd
            tickers = sorted(pd.read_parquet(data / 'features.parquet', columns=['ticker']).ticker.unique())
            if cfg.get('tickers'):
                missing = set(cfg['tickers']) - set(tickers)
                if missing:
                    raise RuntimeError(f'Requested instruments are missing from prepared features: {sorted(missing)}')
                tickers = sorted(cfg['tickers'])
            if smoke:
                tickers = tickers[:2]
            emit(output, 'cpd_started', windows=cfg['cpd_windows'], assets=len(tickers))
            jobs = [(t, w, cfg, fingerprint, 2 if smoke else None) for w in cfg['cpd_windows'] for t in tickers]
            jobs = [job for job in jobs if smoke or not cpd_job_complete(data, job[0], job[1], fingerprint)]
            # Two compiled graphs per lookback/worker. Recycle bounded batches as
            # an additional memory guard; parallelize independent assets only.
            workers = min(4, cfg['cpd_workers'], len(jobs))
            emit(output, 'cpd_runtime', workers=workers, batch_rows=int(os.environ.get('PAPER67_CPD_BATCH_ROWS', '500')),
                 mode='author_functions_reusable_gpflow_graphs')
            if jobs:
                with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                    iterator = iter(jobs)
                    pending = {pool.submit(compute_cpd, job): job for job in [next(iterator) for _ in range(workers)]}
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            job = pending.pop(future)
                            result = future.result()
                            print('Author GPflow CPD:', result, flush=True)
                            next_job = job if result[2] == 'partial' else next(iterator, None)
                            if next_job is not None:
                                pending[pool.submit(compute_cpd, next_job)] = next_job
            if smoke:
                # Train a real author baseline on real data. CPD numerical smoke above is separate.
                emit(output, 'models_started')
                r = run_model(cfg, None, 2000, 2001, fingerprint, smoke=True)
                merged = load_panel(cfg, 10)
                if merged.empty or merged[['cp_score_10','cp_rl_10']].isna().any().any():
                    raise RuntimeError('Author CPD feature integration failed.')
                write_json(output / 'smoke_passed.json', {'gpu': readiness, 'model': r,
                    'cpd_windows_per_asset': 2, 'author_cpd_feature_merge_rows': len(merged)})
                emit(output, 'complete', smoke=True)
                return
            emit(output, 'models_started')
            receipts = []
            years = cfg['test_boundaries']
            for lbw in [None]+cfg['cpd_windows']:
                for i, year in enumerate(years):
                    end = years[i+1] if i+1 < len(years) else cfg['last_test_end']
                    receipts.append(run_model(cfg, lbw, year, end, fingerprint))
            manifest = evaluate(cfg, receipts)
            emit(output, 'complete', **{k:v for k,v in manifest.items() if k.startswith('completed_')})
        except BaseException as exc:
            emit(output, 'failed', error=type(exc).__name__, message=str(exc))
            raise

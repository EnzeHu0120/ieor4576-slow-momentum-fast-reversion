"""Load immutable author modules; apply only recorded API compatibility edits.

No PyTorch model or SciPy/Numba CPD implementation is imported here.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / 'vendor/author-original'
COMMIT = 'e0352cb0bdbf8045accb1bf1a5705ae0cb9ea624'
PATCHES = {
    'mom_trans/model_inputs.py': [('dtype=np.int)', 'dtype=int)')],
    'mom_trans/classical_strategies.py': [('.fillna(method="bfill")', '.bfill()')],
    'mom_trans/data_prep.py': [
        ('.fillna(method="ffill")', '.ffill()'),
        ('.index.weekofyear', '.index.isocalendar().week.to_numpy()'),
    ],
    'mom_trans/deep_momentum_network.py': [
        ('from keras_tuner.tuners.randomsearch import RandomSearch',
         'from keras_tuner import RandomSearch'),
        ('self._get_checkpoint_fname(trial.trial_id, self._reported_step)',
         'self._get_checkpoint_fname(trial.trial_id)'),
        ('self._build_and_fit_model(trial, args, copied_fit_kwargs)',
         'self._build_and_fit_model(trial, *args, **copied_fit_kwargs)'),
        ('kt.engine.tuner_utils.TunerCallback',
         '__import__("keras_tuner.src.engine.tuner_utils", fromlist=["TunerCallback"]).TunerCallback'),
        ('class TunerDiversifiedSharpe(kt.tuners.RandomSearch):',
         'class TunerDiversifiedSharpe(kt.tuners.RandomSearch):\n    _reported_step = 0'),
        ('            max_trials,\n            seed,\n            hyperparameters,\n            tune_new_entries,\n            allow_new_entries,',
         '            max_trials=max_trials,\n            seed=seed,\n            hyperparameters=hyperparameters,\n            tune_new_entries=tune_new_entries,\n            allow_new_entries=allow_new_entries,'),
        ('keras.optimizers.Adam(lr=learning_rate, clipnorm=max_gradient_norm)',
         'keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)'),
        ('        super(TunerValidationLoss, self).run_trial(trial, *args, **kwargs)',
         '        return super(TunerValidationLoss, self).run_trial(trial, *args, **kwargs)'),
    ],
}


def load_author():
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    # Keras Tuner detects standalone Keras 3 even with TF_USE_LEGACY_KERAS.
    # Bind its public keras import to the same legacy implementation as the author model.
    import tf_keras
    sys.modules['keras'] = tf_keras
    if 'mom_trans' in sys.modules:
        loaded = sys.modules['mom_trans']
        if getattr(loaded, '__author_original__', False):
            return
        raise RuntimeError('Another mom_trans implementation is loaded; restart this kernel.')
    for name in ['mom_trans', 'settings']:
        module = types.ModuleType(name)
        module.__path__ = [str(AUTHOR / name)]
        module.__author_original__ = True
        sys.modules[name] = module
    # Settings and model inputs are executed directly from immutable source.
    for name in ['settings.hp_grid', 'settings.fixed_params',
                 'mom_trans.classical_strategies', 'mom_trans.data_prep',
                 'mom_trans.model_inputs', 'mom_trans.changepoint_detection',
                 'mom_trans.deep_momentum_network']:
        relative = name.replace('.', '/') + '.py'
        path = AUTHOR / relative
        source = path.read_text(encoding='utf-8')
        for old, new in PATCHES.get(relative, []):
            if old not in source:
                raise RuntimeError(f'Compatibility patch no longer matches author source: {relative}: {old[:50]}')
            source = source.replace(old, new)
        module = types.ModuleType(name)
        module.__file__ = str(path)
        module.__package__ = name.rpartition('.')[0]
        sys.modules[name] = module
        exec(compile(source, str(path), 'exec'), module.__dict__)
        setattr(sys.modules[module.__package__], name.rsplit('.', 1)[1], module)


def provenance():
    paths = sorted(p for p in AUTHOR.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {'repository': 'https://github.com/kieranjwood/trading-momentum-transformer',
            'commit': COMMIT,
            'files': {p.relative_to(AUTHOR).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'compatibility_edits': PATCHES,
            'runtime_bindings': {'keras': 'tf_keras (legacy Keras for both TensorFlow and Keras Tuner)'},
            'meaning': 'Author source with explicit dependency API adaptations; WRDS data adaptation, not exact Pinnacle rerun.'}

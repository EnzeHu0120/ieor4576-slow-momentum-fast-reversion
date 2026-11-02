"""Explicit experiment boundaries and reproducible author tuner construction."""


def test_intervals(config):
    years = config['test_boundaries']
    return [(year, years[i+1] if i+1 < len(years) else config['last_test_end'])
            for i, year in enumerate(years)]


def seeded_model(module, name, directory, params, seed):
    """Supply the public Keras Tuner seed without altering author model/loss code."""
    original = module.TunerDiversifiedSharpe
    def constructor(*args, **kwargs):
        kwargs['seed'] = seed
        return original(*args, **kwargs)
    module.TunerDiversifiedSharpe = constructor
    try:
        return module.LstmDeepMomentumNetworkModel(name, directory, **params)
    finally:
        module.TunerDiversifiedSharpe = original


def cpd_only_config_transition(previous, current):
    """Permit only the reviewed five-fold end-2020 correction, with no model reuse."""
    old = {**previous, 'test_boundaries': current['test_boundaries']}
    return (previous.get('test_boundaries') == [1995,2000,2005,2010,2015,2020]
            and current.get('test_boundaries') == [1995,2000,2005,2010,2015]
            and old == current)

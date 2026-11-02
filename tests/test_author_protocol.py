from types import SimpleNamespace
import pytest
from momentum_lab.author_protocol import test_intervals as intervals, seeded_model, cpd_only_config_transition


def test_protocol_includes_2020_once_in_the_final_fold():
    folds = intervals({'test_boundaries':[1995,2000,2005,2010,2015], 'last_test_end':2021})
    assert len(folds) == 5
    assert [year for start,end in folds for year in range(start,end)] == list(range(1995,2021))
    assert folds[-1] == (2015,2021)


def test_tuner_receives_public_seed_and_original_class_is_restored():
    captured = []
    def tuner(**kwargs):
        captured.append(kwargs)
        return kwargs
    module = SimpleNamespace(TunerDiversifiedSharpe=tuner)
    module.LstmDeepMomentumNetworkModel = lambda name,directory,**kwargs: module.TunerDiversifiedSharpe(max_trials=50)
    assert seeded_model(module,'LSTM','test',{},2062)['seed'] == 2062
    assert captured == [{'max_trials':50,'seed':2062}]
    assert module.TunerDiversifiedSharpe is tuner


def test_cpd_reuse_transition_rejects_unrelated_setting_changes():
    old = {'test_boundaries':[1995,2000,2005,2010,2015,2020], 'seed':67,'last_test_end':2021}
    new = {**old,'test_boundaries':[1995,2000,2005,2010,2015]}
    assert cpd_only_config_transition(old,new)
    assert not cpd_only_config_transition(old,{**new,'seed':68})


def test_author_baseexception_is_saved_as_na_without_aborting_cpd(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    import pandas as pd
    import momentum_lab.author_native as native
    import momentum_lab.author_cpd_runtime as runtime
    monkeypatch.setattr(native, 'ROOT', tmp_path)
    monkeypatch.setattr(native, 'load_author', lambda: None)
    monkeypatch.setattr(runtime, 'install', lambda: None)
    parent = ModuleType('mom_trans')
    module = ModuleType('mom_trans.changepoint_detection')
    def fails(window):
        raise BaseException('Author optimizer retry exhausted')
    module.changepoint_loc_and_score = fails
    monkeypatch.setitem(sys.modules, 'mom_trans', parent)
    monkeypatch.setitem(sys.modules, 'mom_trans.changepoint_detection', module)
    (tmp_path/'data').mkdir()
    pd.DataFrame({'ticker':['AN']*3,'date':pd.date_range('2020-01-01',periods=3),
        'daily_returns':[.01,.02,-.01]}).to_parquet(tmp_path/'data/cpd_inputs.parquet')
    result = native.compute_cpd(('AN',1,{'data_directory':'data','results_directory':'results'},'test',None))
    assert result == ('AN',1,'complete')
    saved = pd.read_csv(tmp_path/'data/cpd/1/AN.csv')
    assert len(saved)==1 and saved.cp_score.isna().all()

from pathlib import Path
import hashlib
import json
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_completed_five_asset_budget_and_search_records():
    config = yaml.safe_load((ROOT/'configs/author_five_assets.yaml').read_text())
    assert config['tickers']==['SP','TY','BN','ZG','ZU']
    assert config['random_search_trials']==50 and config['max_epochs']==300 and config['sequence_length']==63
    assert config['cpd_windows']==[10,21,63,126,252]
    manifest = json.loads((ROOT/'report/execution_manifest.json').read_text())
    assert manifest['completed_experiments']==manifest['expected_experiments']==30
    assert manifest['completed_trials']==1500
    assert set(manifest['instruments'])==set(config['tickers'])
    trials = pd.read_csv(ROOT/'report/tables/replication/candidate_search_inventory.csv')
    assert len(trials)==1500 and trials.status.eq('COMPLETED').all()
    assert trials.groupby(['strategy','fold']).size().eq(50).all()


def test_published_artifact_hashes_and_upstream_source_are_exact():
    manifest = json.loads((ROOT/'report/result_artifacts_manifest.json').read_text())
    for relative, item in manifest['files'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==item['sha256'],relative
    upstream = json.loads((ROOT/'report/execution_manifest.json').read_text())['source']['files']
    for relative, digest in upstream.items():
        assert hashlib.sha256((ROOT/'vendor/author-original'/relative).read_bytes()).hexdigest()==digest,relative


def test_comparison_numbers_are_calculated_from_completed_outputs():
    data = pd.read_csv(ROOT/'report/tables/replication/exhibit_3.csv').set_index('strategy')
    comparison = pd.read_csv(ROOT/'report/paper_reference_comparison.csv').set_index('strategy')
    assert abs(data.loc['LSTM','sharpe_ratio']-.5889242419736435)<1e-12
    assert abs(data.loc['LSTM-CPD-Optimised','sharpe_ratio']-.5182497862589879)<1e-12
    pd.testing.assert_series_equal(data.sharpe_ratio.sort_index(),comparison.iloc[:,0].sort_index(),check_names=False)

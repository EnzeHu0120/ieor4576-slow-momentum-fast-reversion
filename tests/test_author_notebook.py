import ast
import json
import pytest
from pathlib import Path

from momentum_lab.author_notebook import read_events, decode_output
from momentum_lab.author_compat import AUTHOR, PATCHES


def test_backend_failure_is_reported_without_waiting_for_process_exit(tmp_path):
    from momentum_lab.author_notebook import check_backend_failure
    events = tmp_path / 'events'
    events.mkdir()
    (events / 'parent.status.json').write_text(json.dumps({
        'kind': 'failed', 'utc': '2026-11-01T01:23:55',
        'error': 'BrokenProcessPool', 'message': 'Worker exited abruptly.'}))
    with pytest.raises(RuntimeError, match='BrokenProcessPool'):
        check_backend_failure(tmp_path, '2026-11-01T01:00:00')
    check_backend_failure(tmp_path, '2026-11-02')


def test_author_compatibility_edits_match_immutable_source():
    for relative, edits in PATCHES.items():
        source = (AUTHOR / relative).read_text(encoding='utf-8')
        for old, new in edits:
            assert old in source, (relative, old)
            source = source.replace(old, new)
        ast.parse(source)


def test_event_reader_ignores_partial_rows_and_merges_workers(tmp_path):
    (tmp_path / '1.jsonl').write_text(json.dumps({'kind':'epoch','utc':'b'})+'\n{"kind":',encoding='utf-8')
    (tmp_path / '2.jsonl').write_text(json.dumps({'kind':'cpd','utc':'a'})+'\n',encoding='utf-8')
    assert [e['kind'] for e in read_events(tmp_path)] == ['cpd','epoch']


def test_new_runner_does_not_import_ported_model_or_cpd():
    source = (Path(__file__).resolve().parents[1] / 'src/momentum_lab/author_native.py').read_text()
    tree = ast.parse(source)
    imports = [node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert 'momentum_lab.models' not in imports
    assert 'momentum_lab.cpd' not in imports
    assert 'mom_trans.deep_momentum_network' in imports
    assert 'mom_trans.changepoint_detection' in imports


def test_utf8_python_and_utf16_wsl_diagnostics():
    text = '\u62d2\u7edd\u8bbf\u95ee WSL'
    assert decode_output(text.encode('utf-8')) == text
    assert decode_output(text.encode('utf-16-le')) == text


def test_cpd_resume_skips_only_verified_completed_jobs(tmp_path):
    from momentum_lab.author_native import cpd_job_complete
    folder = tmp_path / 'cpd_receipts/10'
    folder.mkdir(parents=True)
    receipt = folder / 'AN.json'
    receipt.write_text(json.dumps({'fingerprint':'run-a','rows':100,'expected':100}))
    assert not cpd_job_complete(tmp_path, 'AN', 10, 'run-a')
    csv = tmp_path / 'cpd/10/AN.csv'
    csv.parent.mkdir(parents=True)
    csv.write_text('saved data')
    assert cpd_job_complete(tmp_path, 'AN', 10, 'run-a')
    assert not cpd_job_complete(tmp_path, 'AN', 10, 'run-b')
    receipt.write_text(json.dumps({'fingerprint':'run-a','rows':99,'expected':100}))
    assert not cpd_job_complete(tmp_path, 'AN', 10, 'run-a')

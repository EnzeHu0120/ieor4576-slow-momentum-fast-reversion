import json

from momentum_lab.training_events import PREFIX, parse_events


def test_concurrent_records_and_plain_output_share_a_line():
    first = {"kind": "epoch", "epoch": 251}
    second = {"kind": "epoch", "epoch": 44}
    line = "epoch output " + PREFIX + json.dumps(first) + PREFIX + json.dumps(second) + "\n"
    assert list(parse_events(line)) == [first, second]


def test_damaged_record_does_not_hide_next_record():
    event = {"kind": "candidate_completed"}
    assert list(parse_events(PREFIX + '{broken' + PREFIX + json.dumps(event))) == [event]
    assert list(parse_events(PREFIX + '{truncated')) == []

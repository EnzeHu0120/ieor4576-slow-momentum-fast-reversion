from unittest.mock import Mock

import pytest

from momentum_lab.checkpoint_io import replace_checkpoint


def test_replace_retries_transient_sharing_error(tmp_path):
    source = Mock()
    source.replace.side_effect = [PermissionError('sharing lock'), None]
    replace_checkpoint(source, tmp_path / 'model.pt', delay=0)
    assert source.replace.call_count == 2


def test_replace_keeps_old_file_if_permission_error_persists(tmp_path):
    destination = tmp_path / 'model.pt'
    destination.write_bytes(b'old')
    source = Mock()
    source.replace.side_effect = PermissionError('denied')
    with pytest.raises(PermissionError):
        replace_checkpoint(source, destination, attempts=2, delay=0)
    assert destination.read_bytes() == b'old'


def test_replace_existing_checkpoint(tmp_path):
    source = tmp_path / 'model.pt.tmp'
    destination = tmp_path / 'model.pt'
    source.write_bytes(b'new')
    destination.write_bytes(b'old')
    replace_checkpoint(source, destination)
    assert destination.read_bytes() == b'new'

import sys

from momentum_lab.notebook_runner import _start_pipeline


def test_child_output_uses_utf8_and_survives_invalid_bytes():
    program = (
        "import sys; "
        "print('\\u8bad\\u7ec3 \\u2014 progress', flush=True); "
        "sys.stderr.buffer.write(b'bad byte: \\x82\\n'); "
        "sys.stderr.buffer.flush(); "
        "print('still running', flush=True)"
    )
    process = _start_pipeline([sys.executable, '-c', program])
    output, _ = process.communicate(timeout=15)
    assert process.returncode == 0
    assert '\u8bad\u7ec3 \u2014 progress' in output
    assert 'bad byte: \ufffd' in output
    assert 'still running' in output

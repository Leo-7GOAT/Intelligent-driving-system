from scripts.watch_existing_pair import command_identifies_worker


def test_native_control_is_an_explicit_guarded_worker(tmp_path):
    batch = tmp_path/'native batch'
    command = f'python -u "{batch}/source/scripts/native_curve_control.py" --batch "{batch}"'
    assert command_identifies_worker(command, batch)
    assert not command_identifies_worker(command, tmp_path/'different batch')


def test_allowed_worker_name_in_note_or_python_expression_cannot_authorize_process(tmp_path):
    assert not command_identifies_worker(f'python unrelated.py --note native_curve_control.py --batch "{tmp_path}"', tmp_path)
    assert not command_identifies_worker(f'python -c "native_curve_control.py" --batch "{tmp_path}"', tmp_path)
    assert not command_identifies_worker(f'python fake_native_curve_control.py --batch "{tmp_path}"', tmp_path)

import hashlib
import json
import zipfile
from pathlib import PureWindowsPath

import pytest

from scripts.package_native_continuation import operational_guard_files, package_guard_snapshot, portable_metadata, scan_payload


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(tmp_path):
    batch = tmp_path/'batch'
    source = batch/'launcher_override/source/scripts'
    source.mkdir(parents=True)
    entries = {}
    for name in ('watch_existing_pair.py', 'probe_carla37.py', 'stop_owned_carla.ps1'):
        path = source/name
        path.write_text('# exact captured operational source\n')
        entries['scripts/'+name] = digest(path)
    manifest = batch/'launcher_override/manifest.json'
    manifest.write_text(json.dumps({'files': entries}))
    execution = {'guard_current_override': True, 'guard_manifest_sha256': digest(manifest),
                 'guard_source_relative': 'launcher_override/source/scripts/watch_existing_pair.py',
                 'guard_source_sha256': entries['scripts/watch_existing_pair.py']}
    (batch/'launcher_execution.json').write_text(json.dumps(execution))
    (batch/'guard_final.json').write_text(json.dumps({'exit_code': 0, 'reason': None}))
    (batch/'infrastructure').mkdir()
    (batch/'infrastructure/server_stop.json').write_text(json.dumps({'status': 'owned_empty_server_stopped'}))
    return batch


def test_guard_supplement_preserves_probe_bytes_and_binds_original_capsule(tmp_path):
    batch = snapshot(tmp_path)
    output = tmp_path/'guard.zip'
    result = package_guard_snapshot(batch, output, 'a'*64)
    assert result['new_controls'] == result['optimizer_updates'] == 0
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['parent_archive_sha256'] == 'a'*64
        name = 'batch/launcher_override/source/scripts/probe_carla37.py'
        assert archive.read(name) == (batch/'launcher_override/source/scripts/probe_carla37.py').read_bytes()
        for name, record in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == record['sha256']
    with pytest.raises(FileExistsError):package_guard_snapshot(batch, output, 'a'*64)


@pytest.mark.parametrize('failure', ['manifest', 'probe', 'missing_probe', 'executed_guard', 'escape', 'undeclared'])
def test_guard_export_rejects_mismatched_or_unregistered_sources_before_writing(tmp_path, failure):
    batch = snapshot(tmp_path)
    manifest_path = batch/'launcher_override/manifest.json'
    execution_path = batch/'launcher_execution.json'
    execution = json.loads(execution_path.read_text())
    if failure == 'manifest':manifest_path.write_text('{}')
    elif failure == 'probe':(batch/'launcher_override/source/scripts/probe_carla37.py').write_text('# changed')
    elif failure == 'missing_probe':(batch/'launcher_override/source/scripts/probe_carla37.py').unlink()
    elif failure == 'executed_guard':execution['guard_source_sha256'] = '0'*64
    elif failure == 'undeclared':execution['guard_current_override'] = False
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['../../outside.py'] = '0'*64
        manifest_path.write_text(json.dumps(manifest))
        execution['guard_manifest_sha256'] = digest(manifest_path)
    execution_path.write_text(json.dumps(execution))
    output = tmp_path/'never.zip'
    with pytest.raises(ValueError):package_guard_snapshot(batch, output, 'a'*64)
    assert not output.exists() and not output.with_name('never.zip.partial').exists()


def test_historical_run_without_operational_override_needs_no_new_snapshot(tmp_path):
    assert operational_guard_files(tmp_path) == set()


@pytest.mark.parametrize('style', ['windows', 'unix', 'embedded_json', 'invalid_json_prose'])
def test_auxiliary_error_path_redaction_preserves_original_hash_and_failure_status(style):
    private = str(PureWindowsPath('C:/', 'Users', 'fixture', 'project', 'missing.json'))
    if style == 'unix':private = '/'+'/'.join(('home', 'fixture', 'project', 'missing.json'))
    error = "Cannot find path '"+private+"' because it does not exist."
    if style == 'invalid_json_prose':error = '[failed] '+error
    if style == 'embedded_json':error = json.dumps({'error': error})
    original = {'status': 'LAUNCH_OR_SUPERVISION_FAILURE', 'error': error}
    changes = []
    exported = portable_metadata(original, changes)
    assert original['error'] == error and exported['status'] == original['status']
    assert changes and all(len(row['original_value_sha256']) == 64 for row in changes)
    if style != 'embedded_json':
        assert changes[0]['original_value_sha256'] == hashlib.sha256(error.encode()).hexdigest()
    scan_payload('launcher_failure.json', json.dumps(exported).encode())

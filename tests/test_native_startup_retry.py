import json
import gzip

import pytest

from scripts import retry_unstarted_native as retry


def fixture(tmp_path, monkeypatch):
    batch = tmp_path / 'failed'; infrastructure = batch / 'infrastructure'; infrastructure.mkdir(parents=True)
    def write(path, value): path.write_text(json.dumps(value))
    write(batch / 'launcher_failure.json', {'error': 'Connected server is not matching-version and empty'})
    write(infrastructure / 'readiness_probe_02.json', {'status': 'CONNECTION_FAILED'})
    write(infrastructure / 'server_start.json', {'owned_by_task': True, 'shipping_pid': 999991, 'launcher_pid': 999992})
    probe = infrastructure / 'stop_probe.json'
    write(probe, {'status': 'CONNECTED', 'client_version': '0.9.15', 'server_version': '0.9.15', 'actors': []})
    write(infrastructure / 'server_stop.json', {'status': 'owned_empty_server_stopped', 'shipping_pid': 999991, 'probe': str(probe)})
    monkeypatch.setattr(retry, 'pid_alive', lambda pid: False)
    return batch


def test_verified_timeout_is_zero_control_and_read_only(tmp_path, monkeypatch):
    batch = fixture(tmp_path, monkeypatch)
    before = {p.relative_to(batch): p.read_bytes() for p in batch.rglob('*') if p.is_file()}
    evidence = retry.startup_only_timeout(batch)
    assert evidence['applied_controls'] == 0 and not evidence['worker_started']
    assert before == {p.relative_to(batch): p.read_bytes() for p in batch.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault', ['worker', 'bridge', 'connected_probe', 'nonempty_cleanup', 'live_process'])
def test_worker_activity_foreign_world_or_live_identity_cannot_be_retried_as_startup(tmp_path, monkeypatch, fault):
    batch = fixture(tmp_path, monkeypatch)
    if fault == 'worker': (batch / 'worker_process.json').write_text('{}')
    if fault == 'bridge': (batch / 'infrastructure/bridge_unknown_responses.jsonl.gz').touch()
    if fault == 'connected_probe': (batch / 'infrastructure/readiness_probe_02.json').write_text('{"status":"CONNECTED"}')
    if fault == 'nonempty_cleanup':
        path = batch / 'infrastructure/stop_probe.json'; value = json.loads(path.read_text()); value['actors'] = [{'id': 42}]; path.write_text(json.dumps(value))
    if fault == 'live_process': monkeypatch.setattr(retry, 'pid_alive', lambda pid: True)
    with pytest.raises(ValueError): retry.startup_only_timeout(batch)


def metadata_fixture(tmp_path, monkeypatch):
    batch = fixture(tmp_path, monkeypatch)
    for name, value in {
        'protocol.json': {'evaluation_only': True, 'freeze_observation_statistics': True, 'methods': ['B1']},
        'failure.json': {'error': "ValueError('Frozen observation statistics require registered learning continuation')"},
        'guard_final.json': {'exit_code': 1, 'reason': None, 'worker_pid': 999990},
        'worker_process.json': {'pid': 999990}, 'guard.json': {'pid': 999989},
    }.items(): (batch / name).write_text(json.dumps(value))
    (batch / 'B1').mkdir(); (batch / 'B1/config.json').write_text('{}')
    for kind in ('requests', 'responses'):
        with gzip.open(batch / f'infrastructure/bridge_test_{kind}.jsonl.gz', 'wt') as stream: pass
    return batch


def test_metadata_repair_requires_complete_empty_streams_and_known_error(tmp_path, monkeypatch):
    batch = metadata_fixture(tmp_path, monkeypatch)
    evidence = retry.evaluation_metadata_preflight(batch)
    assert evidence['worker_started'] and evidence['applied_controls'] == 0
    assert len(evidence['empty_bridge_streams']) == 2


@pytest.mark.parametrize('fault', ['request', 'response', 'trace', 'wrong_error', 'live'])
def test_metadata_repair_refuses_any_partial_execution_or_unknown_failure(tmp_path, monkeypatch, fault):
    batch = metadata_fixture(tmp_path, monkeypatch)
    if fault in ('request', 'response'):
        with gzip.open(batch / f'infrastructure/bridge_test_{fault}s.jsonl.gz', 'wt') as stream:
            stream.write(json.dumps({'op': 'begin'} if fault == 'request' else {'frame': 42})+'\n')
    if fault == 'trace': (batch / 'B1/evaluation_trace.jsonl.gz').touch()
    if fault == 'wrong_error': (batch / 'failure.json').write_text('{"error":"different"}')
    if fault == 'live': monkeypatch.setattr(retry, 'pid_alive', lambda pid: True)
    with pytest.raises(ValueError): retry.evaluation_metadata_preflight(batch)

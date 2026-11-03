"""A failed restart must never discard saved learning or duplicate a live course."""
import json
import pytest
from scripts import prepare_native_reference_restart as restart


@pytest.fixture
def interrupted(tmp_path, monkeypatch):
    parent = tmp_path / 'parent'
    parent.mkdir()
    documents = {
        'protocol.json': {'reference_context': {'version': restart.VERSION}, 'methods': ['H', 'H0']},
        'guard_final.json': {'exit_code': 1, 'worker_pid': 10},
        'worker_process.json': {'pid': 10}, 'guard.json': {'pid': 11},
        'failure.json': {'error': "InfrastructureFailure('bridge exited')"},
        'infrastructure/server_start.json': {'owned_by_task': True, 'shipping_pid': 12, 'launcher_pid': 13},
        'infrastructure/server_stop.json': {'shipping_pid': 12, 'status': 'owned_empty_server_stopped',
                                           'post_stop_tcp': [], 'post_stop_udp': [], 'epoch': 100},
        'interruption_audit.json': {'methods': {'H': {'state': 'interrupted', 'checkpoints': []},
                                                'H0': {'state': 'not_started'}}},
    }
    for name, data in documents.items():
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(restart, 'recorded_lifecycle_gone', lambda *args: True)
    return parent


@pytest.mark.parametrize('change', ['saved_checkpoint', 'started_H0', 'occupied_port', 'live_worker', 'already_claimed'])
def test_refuses_restart_that_could_lose_or_duplicate_learning(interrupted, tmp_path, monkeypatch, change):
    parent = interrupted
    if change == 'live_worker':
        monkeypatch.setattr(restart, 'recorded_lifecycle_gone', lambda *args: False)
    elif change == 'already_claimed':
        (parent / 'recovery_registration.json').write_text('{}')
    else:
        name = 'infrastructure/server_stop.json' if change == 'occupied_port' else 'interruption_audit.json'
        path = parent / name
        data = json.loads(path.read_text())
        if change == 'occupied_port':
            data['post_stop_tcp'] = [2100]
        elif change == 'saved_checkpoint':
            data['methods']['H']['checkpoints'] = [{'folder': 'checkpoint_320', 'physics_steps': 320}]
        else:
            data['methods']['H0']['state'] = 'interrupted'
        path.write_text(json.dumps(data))
    output = tmp_path / 'child'
    with pytest.raises((ValueError, FileExistsError)):
        restart.prepare(parent, output)
    assert not output.exists()
    if change != 'already_claimed':
        assert not (parent / 'recovery_registration.json').exists()

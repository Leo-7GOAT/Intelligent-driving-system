import json
import pytest
from navstack.experiments import write_json
from scripts import audit_native_training_only as auditor
from test_native_method_endpoint_audit import _response, _tick, _write_gzip, _sha


@pytest.fixture
def training_batch(tmp_path, monkeypatch):
    batch = tmp_path / 'batch'
    method = batch / 'B1'
    checkpoint = method / 'checkpoint_000000001'
    checkpoint.mkdir(parents=True)
    (batch / 'infrastructure').mkdir()
    write_json(batch / 'protocol.json', dict(methods=['B1'], source_hashes={},
        deterministic_training_route_evaluation=False))
    write_json(batch / 'summary.json', {'status': 'complete'})
    write_json(batch / 'guard_final.json', {'exit_code': 0, 'reason': None})
    write_json(batch / 'infrastructure/server_stop.json', {'status': 'owned_empty_server_stopped'})
    write_json(method / 'config.json', {'env': {}})
    response = _response('train', 2, 10, 1.)
    begin = dict(ok=True, episode_id='train', request_id=1, reset_ticks=2,
                 frame=9, scene={'condition': 0})
    _write_gzip(batch / 'infrastructure/bridge_responses.jsonl.gz', [{'response': begin}, {'response': response}])
    tick = _tick(response)
    tick['reward_components'] = {'progress': .25}
    _write_gzip(method / 'transitions_000.jsonl.gz', [{'trace': [tick], 'physics_steps': 1}])
    for name in ('model.zip', 'normalization.pkl', 'resume_state.pkl'):
        (checkpoint / name).write_bytes(name.encode())
    (method / 'model.zip').write_bytes(b'model.zip')
    (method / 'normalization.pkl').write_bytes(b'normalization.pkl')
    metadata = dict(physics_steps=1, actions=1, optimizer_steps=2)
    for name, key in [('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256'), ('resume_state.pkl', 'resume_state_sha256')]:
        metadata[key] = _sha(checkpoint / name)
    write_json(checkpoint / 'metadata.json', metadata)
    write_json(method / 'latest_checkpoint.json', {'folder': checkpoint.name})
    summary = dict(physics_steps=1, actions=1, optimization_phases=1, optimizer_steps=2,
                   stopped_reason=None, world_ticks=3, reward_components={'progress': .25})
    write_json(method / 'summary.json', summary)
    (method / 'optimization.jsonl').write_text(json.dumps(summary) + '\n')
    monkeypatch.setattr(auditor, 'verify_registered_native_begin', lambda begin, p, case: None)
    return batch


def test_training_only_audit_accounts_for_all_controls_resets_and_rewards(training_batch):
    result = auditor.audit(training_batch)
    assert result['training_control_frames_verified'] == 1
    assert result['reset_ticks_verified'] == 2
    assert result['methods']['B1']['reward_components'] == {'progress': .25}


@pytest.mark.parametrize('fault', ['changed_measured_state', 'wrong_reward_total', 'unrecorded_control',
                                    'wrong_cumulative_count', 'missing_first_frame'])
def test_training_only_audit_rejects_inconsistent_physical_or_saved_evidence(training_batch, fault):
    batch = training_batch
    if fault == 'wrong_reward_total':
        path = batch / 'B1/summary.json'
        data = auditor.read(path)
        data['reward_components']['progress'] = .5
        write_json(path, data)
    elif fault == 'wrong_cumulative_count':
        path = batch / 'B1/transitions_000.jsonl.gz'
        rows, _ = auditor.read_prefix(path)
        rows[0]['physics_steps'] = 2
        _write_gzip(path, rows)
    else:
        path = batch / 'infrastructure/bridge_responses.jsonl.gz'
        rows, _ = auditor.read_prefix(path)
        if fault == 'changed_measured_state':
            rows[1]['response']['state']['x'] = 42.
        elif fault == 'unrecorded_control':
            rows.append({'response': _response('train', 3, 11, 1.1)})
        else:
            rows[0]['response']['frame'] = 8
        _write_gzip(path, rows)
    with pytest.raises(ValueError):
        auditor.audit(batch)

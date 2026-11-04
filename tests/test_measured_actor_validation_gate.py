import json

import pytest

import scripts.prepare_measured_actor_validation_probe as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.mark.parametrize('fault', ['offline_only', 'PPO_updates', 'unsafe', 'missing_case', 'different_seed', 'learning_enabled', 'raw_hash'])
def test_known24_probe_requires_actual_fixed_canonical_basic_gate(tmp_path, monkeypatch, fault):
    parent = tmp_path / 'basic'
    p = dict(seed=22, measured_actor_basic_probe=dict(version='v1'), evaluation_only=True)
    episodes = [dict(case_id=i, success=True, termination_reason='success') for i in range(4)]
    summary = dict(physics_steps=0, optimizer_steps=0)
    if fault == 'offline_only': p.pop('measured_actor_basic_probe')
    if fault == 'PPO_updates': summary['optimizer_steps'] = 1
    if fault == 'unsafe': episodes[1].update(success=False, termination_reason='out_of_bounds')
    if fault == 'missing_case': episodes.pop()
    if fault == 'different_seed': p['seed'] = 11
    if fault == 'learning_enabled': p['evaluation_only'] = False
    write(parent / 'protocol.json', p)
    write(parent / 'B1/evaluation.json', dict(episodes=episodes))
    write(parent / 'B1/summary.json', summary)
    write(parent / 'B1/endpoint_raw_audit.json', dict(protocol_sha256='changed' if fault == 'raw_hash' else module.sha(parent / 'protocol.json')))
    monkeypatch.setattr(module, 'require_finished_parent', lambda path: None)
    monkeypatch.setattr(module, 'delivery_audit', lambda path: dict(strict_delivery_gate_passed=True))
    output = tmp_path / 'unstarted'
    with pytest.raises(ValueError, match='actual canonical22 basic gate'):
        module.prepare(parent, output)
    assert not output.exists()


def test_known24_preparation_cannot_overwrite_existing_registration(tmp_path):
    with pytest.raises(FileExistsError): module.prepare(tmp_path / 'unused', tmp_path)

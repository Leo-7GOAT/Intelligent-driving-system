import json

import pytest

import scripts.analyze_native_broader_b1_course as module


@pytest.mark.parametrize('fault', ['unregistered', 'wrong_budget', 'boolean_budget', 'hash', 'controls', 'ack', 'protocol'])
def test_short_segment_cannot_claim_full_course_without_valid_original_evidence(tmp_path, monkeypatch, fault):
    parent = tmp_path / 'original'
    batch = tmp_path / 'recovery'
    parent.mkdir(); batch.mkdir()
    (parent / 'protocol.json').write_text('{}')
    prefix = dict(saved_controls=11520, verified_controls=11520, all_engine_controls_acknowledged=True)
    if fault == 'controls': prefix['verified_controls'] = 11519
    if fault == 'ack': prefix['all_engine_controls_acknowledged'] = 1
    saved = parent / 'verified_saved_prefix.json'
    saved.write_text(json.dumps(prefix))
    recovery = dict(saved_original_controls=11520, resumed_controls=8960,
        saved_prefix_sha256=module.sha(saved), original_protocol_sha256=module.sha(parent / 'protocol.json'))
    p = dict(parent_batch='original', broader_B1_PPO=dict(new_actual_PPO_controls=8960),
             broader_course_recovery=recovery)
    if fault == 'unregistered': p.pop('broader_course_recovery')
    if fault == 'wrong_budget': p['broader_B1_PPO']['new_actual_PPO_controls'] = 8959
    if fault == 'boolean_budget': p['broader_B1_PPO']['new_actual_PPO_controls'] = True
    if fault == 'hash': saved.write_text(saved.read_text() + ' ')
    if fault == 'protocol': (parent / 'protocol.json').write_text('{"changed":true}')
    (batch / 'protocol.json').write_text(json.dumps(p))
    monkeypatch.setattr(module, 'require_finished_parent', lambda path: None)
    with pytest.raises(ValueError):
        module.analyze(batch)

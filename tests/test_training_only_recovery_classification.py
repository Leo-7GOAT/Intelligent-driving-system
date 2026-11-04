import json
import pytest
from scripts import audit_interrupted_native as interruption
from test_native_training_only_audit import training_batch
from test_native_method_endpoint_audit import _response, _tick, _write_gzip


@pytest.mark.parametrize('fault', [None, 'short_budget', 'reload_not_verified', 'wrong_optimizer_endpoint'])
def test_finished_training_only_is_not_replayed_but_inconsistent_endpoints_remain_unfinished(training_batch, fault):
    batch = training_batch
    protocol = json.loads((batch/'protocol.json').read_text())
    protocol['method_budgets'] = {'B1': 1}
    if fault == 'short_budget':
        protocol['method_budgets']['B1'] = 2
    (batch/'protocol.json').write_text(json.dumps(protocol))
    summary = json.loads((batch/'B1/summary.json').read_text())
    summary['reload_exact'] = fault != 'reload_not_verified'
    if fault == 'wrong_optimizer_endpoint':
        summary['optimizer_steps'] += 1
    (batch/'B1/summary.json').write_text(json.dumps(summary))
    (batch/'B1/heartbeat.json').write_text(json.dumps({'pid': 123, 'physics_steps': 1}))
    interruption.audit(batch)
    result = json.loads((batch/'interruption_audit.json').read_text())['methods']['B1']
    assert result['state'] == ('complete' if fault is None else 'interrupted')
    if fault is None:
        assert result['training_only_endpoint_retained'] is True
        assert result['endpoint_evaluation'] == 'NOT_EXECUTED'
        assert not (batch/'B1/evaluation.json').exists()


@pytest.mark.parametrize('extra_rollout', [False, True])
def test_completion_uses_the_recorded_final_rollout_instead_of_a_hierarchical_only_overshoot_limit(training_batch, extra_rollout):
    batch = training_batch
    protocol = json.loads((batch/'protocol.json').read_text())
    protocol['method_budgets'] = {'B1': 1}
    (batch/'protocol.json').write_text(json.dumps(protocol))
    rows = []
    for i in range(321):
        tick = _tick(_response('train', 2+i, 10+i, 1.+i))
        rows.append({'trace': [tick], 'physics_steps': i+1})
    _write_gzip(batch/'B1/transitions_000.jsonl.gz', rows)
    summary = json.loads((batch/'B1/summary.json').read_text())
    summary.update(physics_steps=321, actions=321, reload_exact=True,
                   optimization_phases=2 if extra_rollout else 1)
    (batch/'B1/summary.json').write_text(json.dumps(summary))
    metadata_path = batch/'B1/checkpoint_000000001/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(physics_steps=321, actions=321)
    metadata_path.write_text(json.dumps(metadata))
    optimization = ([dict(summary, physics_steps=320, optimization_phases=1)] if extra_rollout else []) + [summary]
    (batch/'B1/optimization.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in optimization))
    (batch/'B1/heartbeat.json').write_text(json.dumps({'pid': 123, 'physics_steps': 321}))
    interruption.audit(batch)
    result = json.loads((batch/'interruption_audit.json').read_text())['methods']['B1']
    assert result['state'] == ('interrupted' if extra_rollout else 'complete')

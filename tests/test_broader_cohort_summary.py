from copy import deepcopy
import json

import pytest

from scripts.summarize_broader_b1_cohort import summarize, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(tmp_path, include55=False):
    original = tmp_path / 'runs/B1_additional_broader_cohort_20261102'
    execution = tmp_path / 'runs/recovery_execution'
    pilot_name = '20261101T131830295783Z_native_B1_broader_shared_parking_PPO'
    baseline = dict(routes=[{}], native_case_manifest=dict(cases=[{}] * 36),
        native_validation=dict(manifest=dict(cases=[{}] * 24)), calibration_sha256='calibration',
        broader_B1_PPO=dict(preserved_low_level_source_hashes={}), recovery=dict(B1={}))
    cores = {name: dict(cases=8, successes=4, termination_counts=dict(success=4, out_of_bounds=4))
             for name in ('curve_parking', 'static_vehicle', 'crossing_walker')}
    completed, rows, batches, audits = [], [], {}, {}
    for seed in (11, 22, 33, 44) + ((55,) if include55 else ()):
        name = pilot_name if seed == 22 else 'seed' + str(seed)
        batch = tmp_path / 'runs' / name
        p = deepcopy(baseline); p['seed'] = seed
        if seed == 44:
            parent = tmp_path / 'runs/original44'
            write(parent / 'protocol.json', {})
            prefix = dict(saved_controls=11520, verified_controls=11520,
                all_engine_controls_acknowledged=True, saved_Adam=dict(parameter_states=35, step_increment_each=360))
            write(parent / 'verified_saved_prefix.json', prefix)
            p['parent_batch'] = parent.name
            p['recovery']['B1']['unknown_unflushed_parent_tail'] = True
            recovery = dict(saved_original_controls=11520, saved_original_Adam_steps=360,
                saved_prefix_sha256=sha(parent / 'verified_saved_prefix.json'), original_protocol_sha256=sha(parent / 'protocol.json'))
        write(batch / 'protocol.json', p)
        (batch / 'B1').mkdir()
        (batch / 'B1/model.zip').write_bytes(b'model')
        (batch / 'B1/normalization.pkl').write_bytes(b'normalizer')
        write(batch / 'B1/summary.json', dict(lineage_control_steps=51200))
        write(batch / 'summary.json', dict(status='complete'))
        write(batch / 'guard_final.json', dict(exit_code=0, reason=None))
        write(batch / 'infrastructure/server_stop.json', dict(status='owned_empty_server_stopped'))
        result = dict(protocol_sha256=sha(batch / 'protocol.json'), final_model_sha256=sha(batch / 'B1/model.zip'),
            normalization_sha256=sha(batch / 'B1/normalization.pkl'), lineage_PPO_controls=51200,
            validation_cases=24, measured_start_actions_verified=24, all_engine_controls_acknowledged=True,
            observation_RMS_exact=True, saved_Adam_proof=dict(parameter_states=35, step_increment_each=280 if seed == 44 else 640),
            new_actual_PPO_controls=8960 if seed == 44 else 20480, new_actual_Adam_steps=280 if seed == 44 else 640,
            all_actual_control_frames_verified=20000 if seed == 44 else 30000,
            validation_termination_counts=dict(success=12, out_of_bounds=12), validation_successes=12, by_core=cores)
        if seed == 44: result['explicit_recovery'] = recovery
        audit = (tmp_path / 'reports/native_B1_broader_seed22_20261102.json' if seed == 22 else
                 (original if seed in (11, 33) else execution) / f'seed{seed}_learning_audit.json')
        write(audit, result)
        batches[seed], audits[seed] = batch, audit
        if seed != 22:
            rows.append(dict(seed=seed, batch=name))
            completed.append(dict(seed=seed, batch=name, audit_sha256=sha(audit)))
    if not include55:
        rows.append(dict(seed=55, batch='pending55'))
    write(execution / 'registration.json', dict(rows=rows))
    write(execution / 'status.json', dict(completed=completed))
    return execution, batches, audits


def test_partial_summary_retains_missing_seed_and_exact_recovery_costs(tmp_path):
    execution, _, _ = fixture(tmp_path)
    result = summarize(tmp_path, execution, True)
    assert result['pending_seeds'] == [55] and not result['complete_five_seed_cohort']
    assert result['policy_condition_evaluations'] == 96 and result['unique_known_validation_conditions'] == 24
    assert result['effective_new_PPO_controls'] == 81920 and result['effective_new_Adam_steps'] == 2560
    recovered = next(r for r in result['seeds'] if r['seed'] == 44)
    assert recovered['interrupted_and_recovered'] and recovered['inherited_unknown_unflushed_tail']
    assert recovered['all_bridge_controls_verified'] == 31520


def test_incomplete_cohort_cannot_be_reported_as_five_completed_seeds(tmp_path):
    execution, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='incomplete'):
        summarize(tmp_path, execution)


def test_complete_five_seed_summary_retains_only24unique_conditions(tmp_path):
    execution, _, _ = fixture(tmp_path, include55=True)
    result = summarize(tmp_path, execution)
    assert result['complete_five_seed_cohort'] and result['pending_seeds'] == []
    assert [r['seed'] for r in result['seeds']] == [11, 22, 33, 44, 55]
    assert result['policy_condition_evaluations'] == 120 and result['unique_known_validation_conditions'] == 24
    assert result['effective_new_PPO_controls'] == 102400
    assert result['qualified_development_seeds'] == 0 and result['research_acceptance'] == 'UNMET'


@pytest.mark.parametrize('fault', ['model', 'prefix', 'audit', 'owned_cleanup'])
def test_cohort_summary_rejects_changed_evidence(tmp_path, fault):
    execution, batches, audits = fixture(tmp_path)
    if fault == 'model': (batches[11] / 'B1/model.zip').write_bytes(b'changed')
    if fault == 'prefix': (tmp_path / 'runs/original44/verified_saved_prefix.json').write_text('{}')
    if fault == 'audit': audits[33].write_text('{}')
    if fault == 'owned_cleanup': write(batches[44] / 'infrastructure/server_stop.json', dict(status='not_clean'))
    with pytest.raises(ValueError):
        summarize(tmp_path, execution, True)

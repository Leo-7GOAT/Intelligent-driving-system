"""Protect fixed-budget endpoint selection and complete development inventories."""
import json
from types import SimpleNamespace
import pytest
from scripts import prepare_native_reference_evaluation as preparation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def finished_course(tmp_path, monkeypatch):
    root, parent = tmp_path / 'root', tmp_path / 'parent'
    root.mkdir()
    p = dict(reference_context={'version': preparation.VERSION}, methods=['H', 'H0'],
             method_budgets={'H': 10240, 'H0': 10240}, resource_limits={}, source_hashes={},
             freeze_observation_statistics=True, lineage_targets={'H': 10240, 'H0': 10240},
             routes=[{'old_training_route': True}], native_case_manifest={'cases': []},
             recovery={m: {'retained_optimizer_steps': 10} for m in ('H', 'H0')})
    write(parent / 'protocol.json', p)
    (parent / 'source/scripts').mkdir(parents=True)
    (parent / 'source/scripts/native_road_learning.py').write_text(preparation.LEGACY_EVALUATION)
    (parent / 'source/physics.txt').write_text('frozen physics bytes')
    (parent / 'calibration').mkdir()
    auditor = root / 'scripts/audit_native_reference_inference.py'
    auditor.parent.mkdir()
    auditor.write_text('# read-only auditor')
    teacher = dict(routes=[{'route_id': i} for i in range(12)], native_case_manifest={
        'version': 'native_cases_v1', 'split': 'development',
        'cases': [{'route_index': i % 12, 'original_id': i} for i in range(36)]})
    write(root / 'runs/20261102T101413887816Z_native_diverse_TRAIN36_teacher/protocol.json', teacher)
    validation = {'routes': [{'validation_route': i} for i in range(8)],
                  'cases': [{'original_validation_id': i} for i in range(24)]}
    write(root / 'runs/20261103T043242Z_native_recovery/protocol.json', {'native_validation': validation})
    for method in ('H', 'H0'):
        folder = parent / method
        write(folder / 'summary.json', dict(physics_steps=10450, optimizer_steps=330,
            stopped_reason=None, lineage_control_steps=31000, charged_control_steps_lower_bound=42000))
        write(folder / 'latest_checkpoint.json', {'folder': 'checkpoint_final'})
        write(folder / 'checkpoint_final/metadata.json', dict(physics_steps=10450,
            model_sha256=method + '_model_hash', normalization_sha256=method + '_norm_hash'))
        (folder / 'checkpoint_final/model.zip').write_bytes(b'fixed ' + method.encode())
    monkeypatch.setattr(preparation, 'ROOT', root)
    monkeypatch.setattr(preparation, 'require_finished_parent', lambda path: None)
    monkeypatch.setattr(preparation, 'verified_checkpoint_metadata', lambda path: preparation.read(path / 'metadata.json'))
    monkeypatch.setattr(preparation.shutil, 'disk_usage', lambda path: SimpleNamespace(free=32 * 1024**3))
    return parent, tmp_path / 'evaluation', validation


def test_static_keeps_every_original_case_including_known_failure(finished_course):
    parent, output, _ = finished_course
    preparation.prepare(parent, output, 'static')
    p = preparation.read(output / 'protocol.json')
    assert [c['original_id'] for c in p['native_case_manifest']['cases']] == list(range(12, 24))
    assert 14 in [c['original_id'] for c in p['native_case_manifest']['cases']]
    assert p['method_budgets'] == {'H': 0, 'H0': 0}
    assert p['evaluation_only'] and not p['checkpoint_thresholds']
    assert p['resource_limits']['wall_seconds']==14400
    assert 'freeze_observation_statistics' not in p and 'lineage_targets' not in p
    assert (output / 'source/physics.txt').read_bytes() == (parent / 'source/physics.txt').read_bytes()
    assert (output / 'source/scripts/native_road_learning.py').read_text()==preparation.FIXED_EVALUATION
    assert (parent / 'source/scripts/native_road_learning.py').read_text()==preparation.LEGACY_EVALUATION
    for method in ('H', 'H0'):
        assert (output / 'resume' / method / 'model.zip').read_bytes() == (parent / method / 'checkpoint_final/model.zip').read_bytes()


def test_validation_keeps_all_separate_known_conditions(finished_course):
    parent, output, validation = finished_course
    preparation.prepare(parent, output, 'known_validation')
    p = preparation.read(output / 'protocol.json')
    assert p['native_validation'] == validation
    assert len(p['native_validation']['cases']) == 24
    assert p['resource_limits']['wall_seconds']==3600
    assert p['reference_context_endpoint']['new_final_conditions'] == 0


@pytest.mark.parametrize('fault', ['short_budget', 'wrong_saved_endpoint', 'stopped_training'])
def test_refuses_an_endpoint_that_is_not_the_declared_finished_budget(finished_course, fault):
    parent, output, _ = finished_course
    summary = parent / 'H/summary.json'
    data = preparation.read(summary)
    if fault == 'short_budget':
        data['physics_steps'] = 10239
    elif fault == 'stopped_training':
        data['stopped_reason'] = 'wall_budget_reached'
    else:
        path = parent / 'H/checkpoint_final/metadata.json'
        metadata = preparation.read(path)
        metadata['physics_steps'] = 10100
        write(path, metadata)
    write(summary, data)
    with pytest.raises(ValueError):
        preparation.prepare(parent, output, 'static')
    assert not (output / 'prepared.json').exists()
    assert not (output / 'protocol.json').exists()


def test_unknown_frozen_worker_is_not_silently_replaced(tmp_path):
    worker=tmp_path/'worker.py';worker.write_text('unknown preserved source')
    with pytest.raises(ValueError,match='Unrecognized'):
        preparation.bind_evaluation_observation_source(worker)
    assert worker.read_text()=='unknown preserved source'


def test_already_fixed_worker_is_byte_preserved(tmp_path):
    worker=tmp_path/'worker.py';worker.write_text(preparation.FIXED_EVALUATION)
    before=worker.read_bytes();proof=preparation.bind_evaluation_observation_source(worker)
    assert worker.read_bytes()==before
    assert proof['worker_sha256_before']==proof['worker_sha256_after']


def test_historical_fix_changes_only_the_constructor_bytes_including_mixed_newlines(tmp_path):
    worker=tmp_path/'worker.py'
    before=b'# original CRLF\r\n'+preparation.LEGACY_EVALUATION.encode()+b'\n# frozen LF\n'
    worker.write_bytes(before)
    preparation.bind_evaluation_observation_source(worker)
    assert worker.read_bytes()==before.replace(preparation.LEGACY_EVALUATION.encode(),preparation.FIXED_EVALUATION.encode())

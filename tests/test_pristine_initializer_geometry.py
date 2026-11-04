from pathlib import Path
import json
import pytest
from scripts.audit_native_method_endpoint import sha
from scripts.prepare_native_initialization_geometry import require_pristine


def initializer(tmp_path):
    protocol = {'evaluation_only': True, 'methods': ['H', 'H0'],
        'shared_endpoint_mask_v1': True, 'road_geometry_version': 'carla_driving_indexed_union_64_v3',
        'controlled_budget_per_method': 0, 'method_budgets': {'H': 0, 'H0': 0}}
    (tmp_path/'infrastructure').mkdir()
    (tmp_path/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    (tmp_path/'prepared.json').write_text(json.dumps({'status': 'PREPARED_NOT_STARTED',
        'protocol_sha256': sha(tmp_path/'protocol.json')}), encoding='utf-8')
    return protocol


@pytest.mark.parametrize('evidence', ['activation.json', 'launcher_execution.json',
    'guard_final.json', 'H', 'infrastructure/raw.jsonl.gz'])
def test_never_relabels_an_executed_or_activated_initializer_as_pristine(tmp_path, evidence):
    protocol = initializer(tmp_path)
    require_pristine(tmp_path, protocol)
    (tmp_path/evidence).touch()
    with pytest.raises(ValueError, match='execution evidence'):
        require_pristine(tmp_path, protocol)


def test_refuses_modified_registration_and_nonzero_training_budget(tmp_path):
    protocol = initializer(tmp_path)
    (tmp_path/'protocol.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='pristine registered'):
        require_pristine(tmp_path, protocol)
    protocol['method_budgets']['H'] = 640
    (tmp_path/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    (tmp_path/'prepared.json').write_text(json.dumps({'status': 'PREPARED_NOT_STARTED',
        'protocol_sha256': sha(tmp_path/'protocol.json')}), encoding='utf-8')
    with pytest.raises(ValueError, match='pristine registered'):
        require_pristine(tmp_path, protocol)

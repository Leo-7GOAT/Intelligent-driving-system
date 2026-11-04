import hashlib
import json

import pytest

from scripts.prepare_native_initialization_geometry import require_pristine


def initializer(tmp_path, methods):
    protocol={'methods':methods,'evaluation_only':True,'shared_endpoint_mask_v1':methods==['H','H0'],
              'road_geometry_version':'carla_driving_indexed_union_64_v3',
              'controlled_budget_per_method':0,'method_budgets':{m:0 for m in methods}}
    payload=json.dumps(protocol).encode();(tmp_path/'protocol.json').write_bytes(payload)
    (tmp_path/'prepared.json').write_text(json.dumps({'status':'PREPARED_NOT_STARTED',
        'protocol_sha256':hashlib.sha256(payload).hexdigest()}))
    (tmp_path/'infrastructure').mkdir()
    return protocol


@pytest.mark.parametrize('methods',[['H','H0'],['B1']])
def test_accept_only_untouched_zero_ppo_initializers(tmp_path,methods):
    require_pristine(tmp_path,initializer(tmp_path,methods))


@pytest.mark.parametrize('evidence',['B1','activation.json','guard_final.json','failure.json'])
def test_completed_or_started_b1_is_never_relabelled_pristine(tmp_path,evidence):
    protocol=initializer(tmp_path,['B1']);(tmp_path/evidence).touch()
    with pytest.raises(ValueError,match='execution evidence'):require_pristine(tmp_path,protocol)


def test_other_methods_remain_rejected(tmp_path):
    with pytest.raises(ValueError,match='pristine registered'):
        require_pristine(tmp_path,initializer(tmp_path,['B2']))

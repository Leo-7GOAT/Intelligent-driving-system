import json

import pytest

from navstack.config import VehicleConfig
from navstack.experiments import write_json
from scripts.native_road_learning import training_config
import scripts.prepare_native_route_learning as staging


@pytest.fixture
def parent(tmp_path, monkeypatch):
    monkeypatch.setattr(staging,'ROOT',tmp_path)
    root = tmp_path/'runs/rule'
    for folder in ('source/scripts','calibration','infrastructure'):
        (root/folder).mkdir(parents=True)
    write_json(root/'summary.json',{'status':'complete'})
    write_json(root/'guard_final.json',{'exit_code':0,'reason':None})
    write_json(root/'infrastructure/server_stop.json',{'status':'owned_empty_server_stopped'})
    config = training_config({'feedback_ablation':True,'road_candidate_policy':True,
        'road_geometry_version':'carla_driving_indexed_union_64_v3',
        'subgoal_mode':'route_frame_terminal','episode_seconds':45.},'H',VehicleConfig())
    write_json(root/'protocol.json',{'config':config.to_dict(),'routes':[{}, {}, {}, {}],
        'calibration_sha256':'fixture','carla_rpc_host':'127.0.0.1','carla_rpc_port':2100,
        'carla_rpc_port_span':[2100,2101,2102]})
    raw={'cases_verified':4,'termination_counts':{'success':4}}
    write_json(root/'raw_audit.json',raw)
    monkeypatch.setattr(staging,'audit',lambda path:json.loads((path/'raw_audit.json').read_text()))
    monkeypatch.setattr(staging,'resources_ok',lambda path:(True,None))
    (root/'model.zip').write_bytes(b'Never inherit weights into the new semantics')
    return root


def test_route_staging_starts_fresh_and_refuses_duplicate_registration(parent):
    batch=staging.prepare(parent)
    protocol=json.loads((batch/'protocol.json').read_text())
    assert protocol['methods']==['H0','H','B1']
    assert 'recovery' not in protocol and not (batch/'model.zip').exists()
    assert all(training_config(protocol,m,VehicleConfig()).subgoal_mode=='route_frame_terminal'
               for m in protocol['methods'])
    with pytest.raises(FileExistsError,match='existing'):
        staging.prepare(parent)


def test_route_staging_preserves_failed_coverage_and_creates_no_training(parent):
    write_json(parent/'raw_audit.json',{'cases_verified':4,'termination_counts':{'success':3,'deadline':1}})
    with pytest.raises(ValueError,match='all four'):
        staging.prepare(parent)
    assert not (parent.parent/'native_route_learning.json').exists()
    assert not list(parent.parent.glob('*_native_route_learning'))

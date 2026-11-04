from copy import deepcopy
import pytest
from navstack.config import VehicleConfig
from navstack.experiments import signature
from navstack.native_route import REFINED_GRID_VERSION
from scripts.native_road_learning import training_config
from scripts.prepare_native_geometry_migration import migrated_config


def test_migration_allows_only_explicit_geometry_field_and_preserves_other_semantics():
    p={'episode_seconds':45,'road_geometry_version':'carla_driving_indexed_union_64_v3',
       'task_progress_weight':.5,'unsafe_event_penalty':-90.,'flat_hold':1}
    vehicle=VehicleConfig();old=training_config(p,'B1',vehicle)
    saved={'env':old.to_dict(),'signature':signature(old),'seed':11}
    p['road_geometry_version']=REFINED_GRID_VERSION
    result=migrated_config(saved,p,vehicle,'B1')
    assert result['seed']==11 and result['env']['road_geometry_version']==REFINED_GRID_VERSION
    assert result['signature']['observation_shape']==saved['signature']['observation_shape']
    changed=deepcopy(p);changed['unsafe_event_penalty']=-20.
    with pytest.raises(ValueError,match='beyond road geometry'):migrated_config(saved,changed,vehicle,'B1')
    corrupt=deepcopy(saved);corrupt['signature']['road_geometry']='unregistered'
    with pytest.raises(ValueError,match='signature'):migrated_config(corrupt,p,vehicle,'B1')

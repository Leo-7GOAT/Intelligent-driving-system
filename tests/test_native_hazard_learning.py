"""Synthetic registration and evidence checks, without CARLA or claimed learning."""
from dataclasses import asdict
import hashlib
import json
from copy import deepcopy
import pytest
from navstack.config import VehicleConfig
from navstack.experiments import write_json
from navstack.native_hazards import resolve_native_hazard,verify_native_hazard_begin
from scripts.native_road_learning import registered_native_cases,training_config
from scripts.prepare_native_hazard_learning import learning_protocol


def parent(tmp_path):
    points=[{'x':float(i),'y':0.,'z':0.,'yaw_degrees_carla':0.} for i in range(26)]
    digest=hashlib.sha256(json.dumps([[round(p[k],3) for k in ('x','y','yaw_degrees_carla')] for p in points]).encode()).hexdigest()
    route={'map':'fixture','opendrive_sha256':'fixture-map','points':points,'geometry_id':digest}
    template={'episode_seconds':45.,'road_candidate_policy':True,'continuation_feedback':True,
        'subgoal_mode':'route_frame_terminal','road_geometry_version':'carla_driving_indexed_union_64_v3',
        'road_collision_backend':'batch_sat_v1','planner_nodes':7200,'planner_heuristic':'dubins_v1',
        'route_goal_candidate':True,'reference_reuse':True}
    config=training_config(template,'H',VehicleConfig())
    (tmp_path/'calibration').mkdir()
    write_json(tmp_path/'calibration/calibration.json',{'vehicle':asdict(VehicleConfig())})
    write_json(tmp_path/'protocol.json',{'config':config.to_dict(),'routes':[route]*4,
        'native_hazard':{'version':'native_hazard_v1','kind':'static_vehicle','distance_m':10.},
        'carla_rpc_host':'127.0.0.1','carla_rpc_port':2200,'carla_rpc_port_span':[2200,2201,2202]})
    return config


def test_fresh_methods_share_parent_physics_and_actual_hazard_cases_without_inherited_models(tmp_path):
    original=parent(tmp_path)
    protocol=learning_protocol(tmp_path,4096,7200,11,['H','H0','B1'])
    assert training_config(protocol,'H',VehicleConfig()).to_dict()==original.to_dict()
    assert training_config(protocol,'H0',VehicleConfig()).execution_feedback=='continuation_ablated'
    flat=training_config(protocol,'B1',VehicleConfig())
    assert flat.method=='B1' and flat.flat_action_catalog=='speed_steer_3x7_v1'
    assert not set(protocol)&{'recovery','own_success_replay','jobs','curriculum','evaluation_only'}
    cases=registered_native_cases(protocol)
    assert [c['condition'] for c in cases]==[0,1,2,3]
    assert all(c['native_hazard']==protocol['native_hazard'] for c in cases)
    cases[0]['native_hazard']['distance_m']=9.
    assert cases[1]['native_hazard']['distance_m']==10.
    broken=deepcopy(protocol);broken['native_hazard']['distance_m']=2.
    with pytest.raises(ValueError):registered_native_cases(broken)


@pytest.mark.parametrize('controls,wall,seed,methods',[(True,7200,11,['H']),(4096,10,11,['H']),
    (4096,7200,True,['H']),(4096,7200,11,['H','H']),(4096,7200,11,['B3'])])
def test_unregistered_hazard_learning_budget_or_method_is_rejected(tmp_path,controls,wall,seed,methods):
    parent(tmp_path)
    with pytest.raises(ValueError):learning_protocol(tmp_path,controls,wall,seed,methods)


def test_native_reset_binds_registered_task_actual_actor_and_observed_dynamic_state():
    layout={'path':[[6.+i,10.,0.] for i in range(26)],'width':40.,'height':24.}
    spec={'version':'native_hazard_v1','kind':'crossing_walker','lateral_m':-2.5}
    observed={**resolve_native_hazard(layout,spec),'actor_id':42,'actor_type':'walker.pedestrian.0001'}
    x,y,_=observed['spawn_pose'];state={'x':x,'y':y,'vx':0.,'vy':0.,'radius':.5}
    begin={'scene':{'native_layout':layout,'native_hazard':observed,'static':[]},'dynamic':[state],
           'native_dynamic_actors':[{'actor_id':42,'actor_type':observed['actor_type'],'state':state}]}
    verify_native_hazard_begin(begin,spec)
    changed=deepcopy(begin);changed['native_dynamic_actors'][0]['actor_id']=43
    with pytest.raises(ValueError,match='identity/state'):verify_native_hazard_begin(changed,spec)
    with pytest.raises(ValueError,match='presence'):verify_native_hazard_begin(begin,None)


def test_endpoint_experiment_changes_only_declared_policy_filter_under_parent_physics(tmp_path):
    original=parent(tmp_path)
    protocol=learning_protocol(tmp_path,4096,7200,11,['H'],endpoint_filter=True)
    assert protocol['candidate_head']=='unshared_endpoint_v1'
    resolved=training_config(protocol,'H',VehicleConfig()).to_dict()
    assert resolved.pop('endpoint_candidate_filter') is True
    assert resolved==original.to_dict() and 'policy_intervention' in protocol

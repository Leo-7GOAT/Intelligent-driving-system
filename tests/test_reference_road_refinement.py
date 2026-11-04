from types import SimpleNamespace
import numpy as np
import pytest
from navstack.reference_road_refinement import VERSION,settings,prepare_reference,query_points,pose_digest


def fixture():
    owner=SimpleNamespace(road=SimpleNamespace(size=4,dx=1.,dy=1.,collision_drivable=np.ones((4,4),bool)))
    env=SimpleNamespace(world=owner,scene_evidence=dict(native_layout=dict(angle=0.,origin=[0.,0.])),
        reference=None,_reference_selection=None,_rule_plan_cache=None,episode={'plan_failures':0})
    return env,owner


def test_private_constraints_do_not_edit_actual_world_or_replace_policy_action():
    env,owner=fixture();calls=[];replies=[]
    def prepare(action):
        calls.append((action,env.world.road.collision_drivable[1,1]));env.reference=SimpleNamespace(x=[1.],y=[1.],yaw=[0.])
        return 'reference',dict(status='success',expanded_nodes=3)
    def validate(poses):
        replies.append(poses)
        return {'reference_road_probe':dict(version=VERSION,pose_sha256=pose_digest(poses),rejected_world=[[1.5,-1.5,0.]] if len(replies)==1 else [],truncated=False)}
    mode,detail=prepare_reference(env,17,prepare,validate,{'max_attempts':3})
    assert mode=='reference' and calls==[(17,True),(17,False)]
    assert env.world is owner and owner.road.collision_drivable.all()
    assert detail['road_refinement']['cells']==[[1,1]] and len(detail['road_refinement']['probes'])==2
    env.reusable_reference=lambda *args:True;env.decode_action=lambda action:('same target',1.5)
    def reuse(action):
        assert env.world is owner
        return 'reference',dict(status='reference_reused',expanded_nodes=0)
    mode,detail=prepare_reference(env,17,reuse,lambda poses:pytest.fail('A retained certificate was queried again'),{'max_attempts':3})
    assert mode=='reference' and not detail['road_refinement']['probes']


def test_query_failure_restores_actual_world_and_has_no_retry_loop():
    env,owner=fixture()
    def prepare(action):raise RuntimeError('planner failed')
    with pytest.raises(RuntimeError):prepare_reference(env,4,prepare,lambda poses:pytest.fail('Unexpected query'),{'max_attempts':3})
    assert env.world is owner and owner.road.collision_drivable.all()


def test_rejection_budget_clears_unaccepted_reference_instead_of_executing_it():
    env,owner=fixture();calls=[]
    def prepare(action):
        calls.append(action);env.reference=SimpleNamespace(x=[1.],y=[1.],yaw=[0.]);return 'reference',dict(status='success')
    def reject(poses):return {'reference_road_probe':dict(version=VERSION,pose_sha256=pose_digest(poses),rejected_world=[[1.5,-1.5,0.]],truncated=False)}
    mode,detail=prepare_reference(env,26,prepare,reject,{'max_attempts':3})
    assert mode=='plan_failed' and env.reference is None and calls==[26]*3
    assert env.world is owner and owner.road.collision_drivable.all() and env.episode['plan_failures']==1


def test_default_off_and_reject_training_or_ambiguous_configuration():
    assert settings({}) is None
    valid=dict(reference_road_refinement=dict(version=VERSION,max_attempts=3),evaluation_only=True,methods=['H'])
    assert settings(valid)['max_attempts']==3
    for bad in ({**valid,'evaluation_only':False},{**valid,'methods':['B1']},{**valid,'reference_road_refinement':dict(version=VERSION,max_attempts=3.)}):
        with pytest.raises(ValueError):settings(bad)


def test_query_is_bounded_and_returns_actual_query_coordinates():
    calls=[]
    def contains(p):calls.append(p);return False
    poses=[[1.,1.,0.]]
    result=query_points(poses,dict(front=2.,rear=1.,width=1.),dict(angle=0.,origin=[0.,0.],spawn_z=.6),contains,SimpleNamespace)
    assert len(calls)==result['sample_count']==108 and result['pose_sha256']==pose_digest(poses)
    assert result['rejected_world'] and not result['truncated']
    with pytest.raises(ValueError):query_points([[float('nan'),0.,0.]],{}, {},contains,SimpleNamespace)


def test_optin_deterministic_refinement_stops_a_known_rejected_repeated_path():
    env,owner=fixture();env.config=SimpleNamespace(planner=SimpleNamespace(max_seconds=None));calls=[];queries=[]
    def prepare(action):
        calls.append(action);env.reference=SimpleNamespace(x=[1.],y=[1.],yaw=[0.]);return 'reference',dict(status='success',expanded_nodes=3)
    def reject(poses):
        queries.append(poses);return {'reference_road_probe':dict(version=VERSION,pose_sha256=pose_digest(poses),rejected_world=[[1.5,-1.5,0.]],truncated=False)}
    mode,detail=prepare_reference(env,26,prepare,reject,dict(max_attempts=3,stop_repeated_references=True))
    assert mode=='plan_failed' and len(calls)==2 and len(queries)==1
    assert env.reference is None and env.world is owner and owner.road.collision_drivable.all()
    assert env.episode['plan_failures']==1 and detail['road_refinement']['shortcut']=='identical_reference_already_rejected_in_this_frame'
    # A later decision still queries the native predicate afresh. The existing
    # rejected cell means it can then stop after one unchanged-model attempt.
    prepare_reference(env,26,prepare,reject,dict(max_attempts=3,stop_repeated_references=True))
    assert len(calls)==3 and len(queries)==2 and env.episode['plan_failures']==2


def test_wall_budget_searches_do_not_use_deterministic_shortcuts():
    env,owner=fixture();env.config=SimpleNamespace(planner=SimpleNamespace(max_seconds=.1));queries=[]
    def prepare(action):
        env.reference=SimpleNamespace(x=[1.],y=[1.],yaw=[0.]);return 'reference',dict(status='success')
    def reject(poses):
        queries.append(poses);return {'reference_road_probe':dict(version=VERSION,pose_sha256=pose_digest(poses),rejected_world=[],truncated=True)}
    prepare_reference(env,26,prepare,reject,dict(max_attempts=3,stop_repeated_references=True))
    assert len(queries)==3

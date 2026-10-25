"""Synthetic evidence tests rejection/reporting, never actual CARLA outcomes."""
from copy import deepcopy

import pytest

from navstack.config import EnvConfig
from scripts.compare_native_rule_collision import CONTROL_FILES,validate_pair,execution_deltas


def protocols():
    names=set(CONTROL_FILES)|{'navstack/planning.py','navstack/road_grid_batch.py',
                              'navstack/config.py','scripts/evaluate_native_rule.py'}
    a=dict(config=EnvConfig(road_observation=True).to_dict(),routes=[dict(i=i) for i in range(4)],
        source_hashes={n:'0'*64 for n in names},calibration_sha256='same',rule_speed_policy='native_score')
    b=deepcopy(a);b['config']['road_collision_backend']='batch_sat_v1'
    return a,b


def test_collision_pair_requires_only_the_registered_backend_to_change():
    a,b=protocols();validate_pair(a,b)
    assert 'road_collision_backend' not in a['config']


@pytest.mark.parametrize('change',('backend','planner','task','inventory','source','calibration','speed','incomplete_source'))
def test_collision_pair_refuses_different_code_or_task_conditions(change):
    a,b=protocols()
    if change=='backend':b['config']['road_collision_backend']='scalar'
    elif change=='planner':b['config']['planner']['max_nodes']+=1
    elif change=='task':b['routes'][0]['i']=9
    elif change=='inventory':b['routes'].pop()
    elif change=='source':b['source_hashes']['navstack/world.py']='1'*64
    elif change=='calibration':b['calibration_sha256']='another'
    elif change=='speed':b['rule_speed_policy']='paired_low_speed'
    else:
        a['source_hashes'].clear();b['source_hashes'].clear()
    with pytest.raises(ValueError):validate_pair(a,b)


def rows():
    state=dict(x=1.,y=2.,yaw=.1,v=1.,steering=0.,acceleration=0.,time=.05)
    command=dict(acceleration=0.,steering=0.)
    tick=dict(state=state,command=command,requested_command=command,reward=0.,reward_components={},
              physical={},tracking={},intervention=False,controller_status='solved',
              backend_evidence=dict(frame=100,episode_id='a',vehicle_control=dict(throttle=.1),
                speed_setpoint=1.,actuation_mode='acceleration_ff_v2',actuation_state={}))
    return [dict(policy_action=1,info=dict(trace=[tick],plan=dict(status='success',planning_time=2.,expanded_nodes=3)))]


def test_execution_comparison_ignores_timing_identity_but_preserves_physical_differences():
    a=rows();b=deepcopy(a)
    b[0]['info']['trace'][0]['backend_evidence'].update(frame=999,episode_id='another')
    b[0]['info']['plan']['planning_time']=.1
    result=execution_deltas(a,b)
    assert result['exact_observed_execution'] and result['same_plan_metadata']
    b[0]['info']['trace'][0]['state']['x']+=.25
    result=execution_deltas(a,b)
    assert not result['exact_observed_execution']
    assert result['maximum_absolute_state_deltas']['x']==.25
    b[0]['info']['trace'][0]['state']['x']-=.25
    b[0]['info']['trace'][0]['backend_evidence']['vehicle_control']['throttle']=.2
    assert not execution_deltas(a,b)['exact_observed_execution']


def test_execution_comparison_keeps_different_actions_and_unmatched_failed_tail():
    a=rows();b=deepcopy(a);b.append(deepcopy(b[0]));b[0]['policy_action']=0
    result=execution_deltas(a,b)
    assert not result['same_action_sequence'] and not result['exact_observed_execution']
    assert result['compared_controls']==1 and result['unpaired_batch_controls']==1
    assert result['unpaired_scalar_controls']==0
    assert not execution_deltas([],[])['exact_observed_execution']

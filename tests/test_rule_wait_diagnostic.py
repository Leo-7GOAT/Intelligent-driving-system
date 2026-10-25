"""Fabricated frames verify diagnostic rejection, not real CARLA results."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from navstack.vehicle import VehicleState
from scripts.diagnose_rule_wait import first_wait_state


def fixture():
    actuator={'actuator_observation':'ff_state_v1','speed_setpoint':.2,
        'actuation_state':{'signed_pedal':.1,'filtered_acceleration':.2,'feedback_integral':.03}}
    initial={'initial':asdict(VehicleState(6.,16.)),'actuator_evidence':deepcopy(actuator)}
    begin={**deepcopy(actuator),'episode_id':'fixture','request_id':1,'frame':30,'reset_ticks':30,
        'state':initial['initial']}
    actual={};acks={('fixture',1):begin};rows=[]
    for index,action in enumerate((26,0)):
        response={**deepcopy(actuator),'episode_id':'fixture','request_id':2+index,'frame':31+index,
            'imu_frame':31+index,'command':{'acceleration':0.,'steering':0.},
            'state':asdict(VehicleState(6.1+.1*index,16.,time=.05*(index+1)))}
        actual['fixture',31+index]=deepcopy(response);acks['fixture',2+index]=deepcopy(response)
        tick={'state':deepcopy(response['state']),'command':deepcopy(response['command']),
            'backend_frame':31+index,'backend_evidence':{k:v for k,v in response.items() if k not in ('state','command')}}
        rows.append({'case_id':3,'policy_action':action,
            'info':{'trace':[tick],'initial_scenario':deepcopy(initial)}})
    return rows,actual,acks


def test_selected_pre_wait_state_is_the_acknowledged_previous_state():
    rows,actual,acks=fixture()
    result=first_wait_state(rows,3,actual,acks,'ff_state_v1')
    assert result['decision']==1 and result['selected_control_frames_verified']==2
    assert result['state']==actual['fixture',31]['state']


@pytest.mark.parametrize('change',['state','frame','begin','actuator'])
def test_selected_state_diagnostic_rejects_unbound_evidence(change):
    rows,actual,acks=fixture()
    if change=='state':rows[0]['info']['trace'][0]['state']['x']=99.
    elif change=='frame':actual.pop(('fixture',31))
    elif change=='begin':acks['fixture',1]['state']['x']=99.
    else:rows[0]['info']['trace'][0]['backend_evidence']['actuation_state']['feedback_integral']=.1
    with pytest.raises(ValueError):first_wait_state(rows,3,actual,acks,'ff_state_v1')


def test_first_action_wait_uses_measured_begin_without_unexecuted_predecessor():
    rows,actual,acks=fixture();rows=rows[:1];rows[0]['policy_action']=0
    result=first_wait_state(rows,3,actual,acks,'ff_state_v1')
    assert result['decision']==0 and result['state']==acks['fixture',1]['state']
    rows[0]['policy_action']=26
    with pytest.raises(ValueError,match='No recorded first wait'):
        first_wait_state(rows,3,actual,acks,'ff_state_v1')

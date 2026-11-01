from dataclasses import asdict,replace
import numpy as np
import pytest
from navstack.config import EnvConfig
from navstack.experiments import signature
from navstack.parking_assist import goal_parking_command
from navstack.vehicle import Command,VehicleState
from test_route_catalog_environment import RouteFixture,config
from scripts.native_road_learning import training_config
from navstack.config import VehicleConfig


@pytest.mark.parametrize('method',['H','B1','B2','B3'])
def test_assistance_is_shared_and_only_changes_physical_acceleration_inside_goal_pose(method):
    cfg=EnvConfig(method=method,goal_parking_assist=True)
    state=VehicleState(9.5,10.,.1,1.4);before=asdict(state);command=Command(.2,-.14)
    result,changed,eligible=goal_parking_command(state,(10.,10.,0.),command,cfg,True)
    assert result==Command(-cfg.vehicle.max_decel,command.steering) and changed and eligible
    assert asdict(state)==before
    for other,progress in [(VehicleState(8.,10.,0.,1.4),True),(VehicleState(9.5,10.,1.,1.4),True),(state,False)]:
        result,changed,eligible=goal_parking_command(other,(10.,10.,0.),command,cfg,progress)
        assert result is command and not changed


def test_disabled_configuration_keeps_historical_serialization_and_signature():
    old=EnvConfig();assert 'goal_parking_assist' not in old.to_dict() and 'goal_parking_assist' not in signature(old)
    with pytest.raises(ValueError):EnvConfig(goal_parking_assist=True,parking=False)
    with pytest.raises(ValueError):EnvConfig(goal_parking_assist='true')
    cfg=EnvConfig(goal_parking_assist=True)
    c=Command(-cfg.vehicle.max_decel,.2)
    result,changed,eligible=goal_parking_command(VehicleState(9.5,10.,v=1.),(10.,10.,0.),c,cfg,True)
    assert result is c and not changed and eligible
    assert signature(cfg)!=signature(old)


@pytest.mark.parametrize('method',['H','H0','B1','B2','B3'])
def test_registered_native_factory_propagates_same_assistance_to_every_method(method):
    cfg=training_config({'episode_seconds':45.,'goal_parking_assist':True},method,VehicleConfig())
    assert cfg.goal_parking_assist
    assert signature(cfg)['goal_parking_assist']['release_speed']==cfg.goal_speed


def test_flat_assist_issues_a_measured_plant_command_and_retains_unassisted_proposal():
    cfg=EnvConfig.from_dict({**config('B1').to_dict(),'flat_hold':1,'goal_parking_assist':True})
    env=RouteFixture(cfg)
    try:
        env.reset(options={'start_x':23.4})
        env.plant._state=replace(env.state,v=.3)
        env.ever_controlled=True;env.episode['distance']=.2
        before=env.state
        _,_,done,_,info=env.step(np.array([.5,.1],dtype=np.float32))
        tick=info['trace'][0]
        assert tick['unassisted_command']['acceleration']>0
        assert tick['requested_command']['acceleration']==-cfg.vehicle.max_decel
        assert tick['goal_parking_intervention'] and tick['intervention']
        assert env.state.time>before.time and env.episode['goal_parking_controls']==1
        assert tick['state']==asdict(env.state)
        assert not done and abs(env.state.v)<=cfg.goal_speed
        # The unchanged success gate rejects the intervened braking frame.
        # Release on the next cycle lets the real unassisted control succeed.
        _,_,done,_,info=env.step(np.array([.5,.1],dtype=np.float32))
        assert done and info['success']
        assert not info['trace'][0]['goal_parking_intervention']
        assert not info['trace'][0]['intervention']
        assert env.episode['goal_parking_controls']==1
    finally:env.close()

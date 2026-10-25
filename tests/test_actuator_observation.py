"""Fabricated frames exercise the shared interface, never claim CARLA execution."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from navstack.carla_actuation import actuator_state_values, validate_actuator_interface
from navstack.carla_bridge import BridgePlant, CarlaWorld
from navstack.config import EnvConfig, PlannerConfig, VehicleConfig
from navstack.env import NavigationEnv
from navstack.experiments import signature
from navstack.native_route import sample_road_grid
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehiclePlant, VehicleState, Command
from navstack.world import World
from scripts.audit_native_method_endpoint import _verify_ticks, verify_initial_actuator
from scripts.audit_native_policy_sequence import observation_env, advance_observation
from scripts.native_road_learning import training_config


def evidence():
    return {'actuator_observation':'ff_state_v1','speed_setpoint':.8,
        'actuation_state':{'signed_pedal':.2,'filtered_acceleration':.1,'feedback_integral':.03}}


def config(method='H', feedback='enabled', catalog='ego_arcs'):
    return EnvConfig(method=method,execution_feedback=feedback,candidate_observation=True,
        road_observation=True,actuator_observation='ff_state_v1',subgoal_mode=catalog,
        deadline=4.,flat_hold=10,planner=PlannerConfig(max_seconds=None,max_nodes=32,analytic_expansion=True))


class FixtureEnv(NavigationEnv):
    def _create_episode(self,options=None):
        grid=sample_road_grid({'width':32.,'height':32.,'geometry_id':'state-fixture'},lambda x,y:True)
        self.scene_evidence={'width':32.,'height':32.,'goal':[24.,16.,0.],'static':[],
            'road_grid':grid,'native_layout':{'path':[[6.,16.,0.],[24.,16.,0.]]}}
        plant=VehiclePlant(self.config.vehicle,VehicleState(6.,16.))
        plant.evidence=evidence()
        return World(32.,32.,road=RoadGrid(grid)),plant,self.scene_evidence['goal'],'fixture'


def test_shared_state_survives_feedback_ablation_and_versions_old_models():
    observations=[]
    for method,feedback in [('H','enabled'),('H','ablated'),('B1','enabled'),('B2','enabled')]:
        env=FixtureEnv(config(method,feedback));obs,_=env.reset()
        assert obs.shape==(4705,) and signature(env.config)['observation_shape']==[4705]
        np.testing.assert_array_equal(obs[-4:],np.asarray([.8,.2,.1,.03],dtype=np.float32))
        observations.append(obs);env.close()
    np.testing.assert_array_equal(observations[0],observations[2])
    np.testing.assert_array_equal(observations[0],observations[3])
    np.testing.assert_array_equal(observations[0][-4:],observations[1][-4:])
    old=EnvConfig();assert 'actuator_observation' not in old.to_dict()
    assert EnvConfig.from_dict(old.to_dict()).to_dict()==old.to_dict()
    with pytest.raises(ValueError,match='requires shared'):
        EnvConfig(actuator_observation='ff_state_v1')
    for method in ('H0','H','B1'):
        trained=training_config({'feedback_ablation':True,'road_candidate_policy':True,
            'actuator_observation':'ff_state_v1','episode_seconds':45},method,VehicleConfig())
        assert trained.actuator_observation=='ff_state_v1'


@pytest.mark.parametrize('change',[lambda e:e.pop('actuation_state'),
    lambda e:e['actuation_state'].update(feedback_integral=float('nan')),
    lambda e:e.update(speed_setpoint=None),lambda e:e.update(speed_setpoint=4)])
def test_missing_or_invalid_actuator_state_is_never_replaced_with_zero(change):
    env=FixtureEnv(config());env.reset();change(env.plant.evidence)
    with pytest.raises(ValueError):env._observation()
    env.close()


def test_bridge_plant_updates_state_from_the_acknowledged_frame():
    result={**evidence(),'state':asdict(VehicleState(6.,16.)),'command':asdict(Command()),
        'dynamic':[],'collisions':[],'frame':31,'request_id':1,'episode_id':'fixture',
        'imu_frame':31,'sensor_health':'frame_heartbeat_ok'}
    world=CarlaWorld(32.,32.)
    plant=BridgePlant(None,result,world,'ff_state_v1')
    changed=deepcopy(result);changed['frame']=changed['imu_frame']=32
    changed['actuation_state']['filtered_acceleration']=.45
    plant.update(changed)
    assert actuator_state_values(plant.evidence)==[.8,.2,.45,.03]
    del changed['actuation_state']
    with pytest.raises(ValueError):plant.update(changed)
    assert plant.frame==32


@pytest.mark.parametrize('catalog',['ego_arcs','route_frame_terminal'])
def test_complete_observation_history_reconstructs_with_shared_actuator_state(catalog):
    env=FixtureEnv(config(catalog=catalog));obs,_=env.reset()
    replay=observation_env(env.config,env.scene_evidence,env.initial_scenario)
    np.testing.assert_array_equal(replay._observation(),obs)
    for action in (6,0,16):
        obs,_,done,_,info=env.step(action)
        # The lightweight fixture has no backend; explicitly inject fabricated evidence.
        info['trace'][-1]['backend_evidence']=deepcopy(env.plant.evidence)
        advance_observation(replay,{'policy_action':action,'info':info})
        np.testing.assert_array_equal(replay._observation(),obs)
        if done:break
    env.close()


def test_audit_rejects_missing_changed_or_wrong_reset_actuator_values():
    response={**evidence(),'episode_id':'fixture','request_id':2,'frame':31,'imu_frame':31,
        'state':{'time':.05},'command':{'acceleration':1.,'steering':0.}}
    tick={'backend_evidence':deepcopy(response),'backend_frame':31,
        'state':response['state'],'command':response['command']}
    def verify(t):
        return _verify_ticks([t],{('fixture',31):response},{},{},set(),{},set(),'ff_state_v1')
    assert verify(tick)==1
    missing=deepcopy(tick);missing['backend_evidence'].pop('actuation_state')
    with pytest.raises(ValueError):verify(missing)
    changed=deepcopy(tick);changed['backend_evidence']['actuation_state']['feedback_integral']=.1
    with pytest.raises(ValueError,match='Actuator evidence'):verify(changed)
    initial={'actuator_evidence':evidence()}
    begin={**evidence(),'episode_id':'fixture','reset_ticks':30}
    verify_initial_actuator(initial,'fixture',{('fixture',1):begin})
    initial['actuator_evidence']['speed_setpoint']=.9
    with pytest.raises(ValueError,match='reset acknowledgement'):
        verify_initial_actuator(initial,'fixture',{('fixture',1):begin})


def test_bridge_requires_explicit_state_opt_in_for_feedforward_training():
    gains={'longitudinal_mode':'acceleration_ff_v2','drivetrain_mode':'manual_first_v1',
        'throttle_model':{'throttle_gain':7.3,'speed_coefficient':-.08,'intercept':-.65}}
    validate_actuator_interface(gains,'ff_state_v1',True,'native_curve')
    validate_actuator_interface(gains,'off',False,'native_curve')
    with pytest.raises(ValueError,match='explicit actuator'):
        validate_actuator_interface(gains,'off',True,'native_curve')
    with pytest.raises(ValueError,match='measured feedforward'):
        validate_actuator_interface({},'ff_state_v1',True,'native_curve')

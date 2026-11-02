from copy import deepcopy
from dataclasses import asdict
import math
import pickle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.config import EnvConfig,VehicleConfig
from navstack.experiments import signature
from navstack.fixed_reward_units import initialize_fixed_reward_units
from navstack.road_candidate_policy import OrderedFlatRoadCandidatePolicy,RoadCandidateVecNormalize
from navstack.route_error import route_error_features
from navstack.route_feedback_policy import promote,RouteFeedbackFlatPolicy
from navstack.vehicle import VehicleState


def test_route_errors_are_rigid_transform_invariant_and_never_move_the_vehicle():
    vehicle=VehicleConfig();state=VehicleState(x=1.,y=.6,yaw=.1)
    path=np.array([[0.,0.,0.],[2.,0.,0.],[4.,0.,0.]])
    before=asdict(state);copy=path.copy();value=route_error_features(state,path,vehicle)
    np.testing.assert_allclose(value,[.2,-math.sin(.1),math.cos(.1),0.],atol=1e-7)
    rotation=.7;co,si=math.cos(rotation),math.sin(rotation)
    xy=path[:,:2]@np.array([[co,si],[-si,co]])+np.array([20.,-4.])
    moved=VehicleState(x=co*state.x-si*state.y+20,y=si*state.x+co*state.y-4,yaw=state.yaw+rotation)
    np.testing.assert_allclose(route_error_features(moved,np.c_[xy,path[:,2]+rotation],vehicle),value,atol=1e-7)
    assert asdict(state)==before;np.testing.assert_array_equal(path,copy)
    with pytest.raises(ValueError):route_error_features(state,[[0,0,0],[0,0,0]],vehicle)


def test_disabled_route_feature_flag_preserves_historical_configuration_and_signature():
    original=EnvConfig()
    assert 'flat_route_observation' not in original.to_dict() and 'flat_route_observation' not in signature(original)
    with pytest.raises(ValueError):EnvConfig(flat_route_observation=True)
    cfg=EnvConfig(method='B1',road_observation=True,candidate_observation=True,execution_feedback='continuation',
        subgoal_mode='route_frame_terminal',actuator_observation='ff_state_v1',flat_route_observation=True)
    assert signature(cfg)['observation_shape']==[4705] and 'flat_route_observation' in signature(cfg)


class Empty(gym.Env):
    observation_space=spaces.Box(-np.inf,np.inf,(4705,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(2,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):return np.zeros(4705,np.float32),{}
    def step(self,action):return np.zeros(4705,np.float32),0.,False,False,{}


def test_zero_residual_preserves_actions_values_old_adam_rms_and_serialized_ppo(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(15)
    model=PPO(OrderedFlatRoadCandidatePolicy,None,_init_setup_model=False)
    model.observation_space=Empty.observation_space;model.action_space=Empty.action_space;model.n_envs=1;model._setup_model()
    norm=RoadCandidateVecNormalize(DummyVecEnv([Empty]),norm_reward=False)
    initialize_fixed_reward_units(norm,.05)
    norm.obs_rms.mean[599:603]=[1.,2.,3.,4.];norm.obs_rms.var[599:603]=4.
    raw=np.random.default_rng(5).normal(size=(3,4705)).astype(np.float32);raw[:,599:603]=0.
    old=torch.as_tensor(norm.normalize_obs(raw))
    loss=model.policy.get_distribution(old).distribution.mean.square().sum()+model.policy.predict_values(old).square().sum()
    loss.backward();model.policy.optimizer.step();model.policy.optimizer.zero_grad()
    weights={k:v.clone() for k,v in model.policy.state_dict().items()};adam=deepcopy(model.policy.optimizer.state_dict())
    rms=pickle.dumps(norm.obs_rms)
    with torch.no_grad():action=model.policy.get_distribution(old).distribution.mean.clone();value=model.policy.predict_values(old).clone()
    report,params=promote(model,norm)
    changed=raw.copy();changed[:,599:603]=[.3,-.2,.98,.6]
    new=torch.as_tensor(norm.normalize_obs(changed))
    np.testing.assert_array_equal(new[:,599:603],changed[:,599:603])
    with torch.no_grad():
        assert torch.equal(action,model.policy.get_distribution(new).distribution.mean)
        assert torch.equal(value,model.policy.predict_values(new))
    assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in weights.items())
    from scripts.fit_broader_measured_actor import exact_state
    current=model.policy.optimizer.state_dict()
    assert exact_state(adam['state'],current['state']) and current['param_groups'][:-1]==adam['param_groups']
    assert pickle.dumps(norm.obs_rms)==rms and report['new_learned_parameters']==2758 and len(params)==5
    model.save(tmp_path/'model.zip');loaded=PPO.load(tmp_path/'model.zip',device='cpu')
    assert isinstance(loaded.policy,RouteFeedbackFlatPolicy)
    with torch.no_grad():assert torch.equal(action,loaded.policy.get_distribution(new).distribution.mean)
    restored=pickle.loads(pickle.dumps(norm));np.testing.assert_array_equal(restored.normalize_obs(changed),new.numpy())

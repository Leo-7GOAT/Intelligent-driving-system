import gymnasium as gym
import numpy as np
import torch
import pytest
from stable_baselines3 import PPO
from navstack.discrete_flat_policy import DiscreteFlatRoadPolicy
from navstack.road_candidate_policy import FlatRoadCandidateActorCriticPolicy
from navstack.env import flat_action_values
from navstack.config import EnvConfig
from navstack.experiments import signature


class Probe(gym.Env):
    observation_space=gym.spaces.Box(-np.inf,np.inf,(4705,),np.float32)
    def __init__(self,discrete=False):
        self.action_space=gym.spaces.MultiDiscrete([3,7]) if discrete else gym.spaces.Box(-1.,1.,(2,),np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.steps=0
        return np.zeros(4705,dtype=np.float32),{}
    def step(self,action):
        assert self.action_space.contains(action);self.steps+=1
        return np.zeros(4705,dtype=np.float32),float(np.sum(action)),self.steps==8,False,{}


def test_categorical_flat_keeps_seeded_encoders_critic_rng_and_exact_likelihood_reload(tmp_path):
    torch.set_num_threads(1)
    old=PPO(FlatRoadCandidateActorCriticPolicy,Probe(),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    rng=torch.get_rng_state().clone()
    model=PPO(DiscreteFlatRoadPolicy,Probe(True),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.policy.state_dict().items():
        if k!='log_std' and not k.startswith('action_net.'):
            assert torch.equal(v,model.policy.state_dict()[k])
    obs=torch.zeros((16,4705))
    with torch.no_grad():
        action,_,before=model.policy(obs)
        _,after,_=model.policy.evaluate_actions(obs,action)
    torch.testing.assert_close(before,after)
    assert (action>=0).all() and (action[:,0]<3).all() and (action[:,1]<7).all()
    state={k:v.clone() for k,v in model.policy.state_dict().items()};model.learn(16)
    assert model._n_updates==4 and any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in state.items())
    model.save(tmp_path/'model.zip');loaded=PPO.load(tmp_path/'model.zip',device='cpu')
    assert isinstance(loaded.policy,DiscreteFlatRoadPolicy)
    assert all(torch.equal(v,loaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
    np.testing.assert_array_equal(model.predict(obs.numpy(),deterministic=True)[0],loaded.predict(obs.numpy(),deterministic=True)[0])


def test_discrete_direct_action_executes_actual_lightweight_plant_without_planner():
    from test_road_candidate_policy import RoadPlantEnv,config
    cfg=config('B1',actuator='ff_state_v1');cfg.deadline=3.
    cfg.flat_longitudinal_mode='speed_target_v1';cfg.flat_action_catalog='speed_steer_3x7_v1';cfg.__post_init__()
    env=RoadPlantEnv(cfg);env.reset(seed=11)
    assert np.array_equal(env.action_space.nvec,[3,7])
    _,_,_,_,info=env.step(np.array([1,3]))
    assert info['plan'] is None and info['mode']=='flat'
    assert all(t['controller_status']=='direct' for t in info['trace'])
    assert info['trace'][0]['requested_command']['acceleration']==.75 and env.state.v>0
    moving=env.state.v;env.step(np.array([0,3]));assert env.state.v<moving
    np.testing.assert_array_equal(flat_action_values([2,6],cfg),[0,1])
    env.close()
    assert signature(cfg)['flat_action_catalog']=='speed_steer_3x7_v1'


def test_invalid_catalog_actions_and_legacy_serialization():
    cfg=EnvConfig(method='B1',flat_longitudinal_mode='speed_target_v1',flat_action_catalog='speed_steer_3x7_v1')
    for value in ([3,0],[-1,3],[1,7],[1.,3.],0):
        with pytest.raises(ValueError):flat_action_values(value,cfg)
    assert 'flat_action_catalog' not in EnvConfig().to_dict()
    assert 'flat_action_catalog' not in signature(EnvConfig())
    with pytest.raises(ValueError):EnvConfig(flat_action_catalog='speed_steer_3x7_v1')


def test_continuation_signature_retains_vector_start_without_collapsing_semantics():
    from scripts.continue_native_learning import _space_signature
    first=gym.spaces.MultiDiscrete([3,7])
    shifted=gym.spaces.MultiDiscrete([3,7],start=[1,0])
    assert _space_signature(first)['start']['shape']==[2]
    assert _space_signature(first)!=_space_signature(shifted)
    assert _space_signature(gym.spaces.Discrete(3))['start']==0

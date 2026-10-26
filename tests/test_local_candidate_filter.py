import copy
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from navstack.candidate_filter import local_witness_mask
from navstack.road_candidate_policy import LocalWitnessRoadPolicy,UnsharedRoadCandidatePolicy
from navstack.config import EnvConfig
from navstack.experiments import signature


def inputs():
    obs=np.zeros(4705,dtype=np.float32)
    candidates=obs[351:599].reshape(31,8)
    candidates[[3,9],5:7]=1
    # Every global continuation witness is absent, as in the failed road.
    return obs


class Probe(gym.Env):
    observation_space=gym.spaces.Box(-np.inf,np.inf,(4705,),np.float32)
    action_space=gym.spaces.Discrete(31)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.count=0
        return inputs(),{}
    def step(self,action):
        assert action in (0,3,9)
        self.count+=1
        return inputs(),float(action)/30.,self.count==8,False,{}


def test_local_evidence_keeps_wait_and_does_not_require_global_continuation():
    c=inputs()[351:599].reshape(31,8)
    assert np.flatnonzero(local_witness_mask(c)).tolist()==[0,3,9]
    c[:,6]=0
    assert local_witness_mask(c).all()  # no known connector is not impossibility


def test_numpy_policy_masks_likelihood_and_fallback_agree(tmp_path):
    torch.set_num_threads(1)
    baseline=PPO(UnsharedRoadCandidatePolicy,Probe(),seed=11,n_steps=8,batch_size=8)
    rng=torch.get_rng_state().clone()
    model=PPO(LocalWitnessRoadPolicy,Probe(),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in model.policy.state_dict().items():
        old=name.replace('mlp_extractor.backbone.','mlp_extractor.').replace('action_net.head.','action_net.')
        assert torch.equal(value,baseline.policy.state_dict()[old])
    obs=torch.as_tensor(np.stack([inputs()]*32))
    distribution=model.policy.get_distribution(obs)
    probs=distribution.distribution.probs.detach().numpy()
    assert np.flatnonzero(probs[0]>0).tolist()==[0,3,9]
    with torch.no_grad():
        actions,_,before=model.policy(obs)
        _,after,_=model.policy.evaluate_actions(obs,actions)
    torch.testing.assert_close(before,after)
    fallback=obs.clone();fallback[:,351:599]=0
    assert (model.policy.get_distribution(fallback).distribution.probs>0).all()
    state=copy.deepcopy(model.policy.state_dict());model.learn(16)
    assert model._n_updates==4 and any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in state.items())
    model.save(tmp_path/'model.zip');loaded=PPO.load(tmp_path/'model.zip',device='cpu')
    assert isinstance(loaded.policy,LocalWitnessRoadPolicy)
    assert all(torch.equal(v,loaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
    np.testing.assert_array_equal(model.predict(obs.numpy(),deterministic=True)[0],loaded.predict(obs.numpy(),deterministic=True)[0])


def test_rule_receives_same_cached_local_choices_and_default_config_stays_portable():
    from test_road_candidate_policy import RoadPlantEnv,config
    from navstack.env import rule_action
    cfg=config('B3');cfg.local_candidate_filter=True;cfg.__post_init__()
    env=RoadPlantEnv(cfg);obs,_=env.reset(seed=11)
    expected=local_witness_mask(obs[351:599].reshape(31,8))
    np.testing.assert_array_equal(env._local_candidate_mask,expected)
    action=rule_action(env)
    assert expected[action]
    env.close()
    assert 'local_candidate_filter' not in EnvConfig().to_dict()
    assert 'local_candidate_filter' not in signature(EnvConfig())

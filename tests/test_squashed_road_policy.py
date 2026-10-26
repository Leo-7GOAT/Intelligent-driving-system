import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from navstack.road_candidate_policy import FlatRoadCandidateActorCriticPolicy
from navstack.squashed_road_policy import SquashedFlatRoadPolicy
from scripts.native_road_learning import registered_jobs
from scripts.prepare_native_flat_distribution_pair import short_route


class ActuatorProbe(gym.Env):
    """Synthetic bounded-action plumbing; never vehicle performance evidence."""
    observation_space = gym.spaces.Box(-np.inf,np.inf,(4705,),np.float32)
    action_space = gym.spaces.Box(-1.,1.,(2,),np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ticks=0
        self.actions=[]
        return np.zeros(4705,dtype=np.float32),{}

    def step(self, action):
        self.actions.append(action.copy())
        assert self.action_space.contains(action)
        self.ticks+=1
        return np.zeros(4705,dtype=np.float32),-float(np.square(action-.2).sum()),self.ticks==32,False,{}


def test_correct_tanh_density_and_identical_initial_network():
    torch.set_num_threads(1)
    gaussian=PPO(FlatRoadCandidateActorCriticPolicy,ActuatorProbe(),seed=11,n_steps=8,batch_size=8)
    bounded=PPO(SquashedFlatRoadPolicy,ActuatorProbe(),seed=11,n_steps=8,batch_size=8)
    assert all(torch.equal(p,bounded.policy.state_dict()[n]) for n,p in gaussian.policy.state_dict().items())
    assert bounded.policy.squash_output
    observations=torch.zeros((5,4705))
    dist=bounded.policy.get_distribution(observations)
    actions=torch.tensor([[-.9,.7],[-.5,.4],[.1,-.1],[.6,.3],[.8,-.8]])
    actual=dist.log_prob(actions)
    expected=(dist.distribution.log_prob(torch.atanh(actions))-
              torch.log(1-actions.square()+dist.epsilon)).sum(dim=1)
    torch.testing.assert_close(actual,expected)
    with torch.no_grad():
        sampled,_,old_log_prob=bounded.policy(observations)
        _,reevaluated,_=bounded.policy.evaluate_actions(observations,sampled)
    assert (sampled.abs()<=1).all()
    torch.testing.assert_close(old_log_prob,reevaluated)


def test_bounded_rollouts_real_ppo_update_and_reload(tmp_path):
    torch.set_num_threads(1)
    probe=ActuatorProbe()
    model=PPO(SquashedFlatRoadPolicy,probe,seed=11,n_steps=8,batch_size=8,n_epochs=2)
    before=copy.deepcopy(model.policy.state_dict())
    model.learn(16)
    assert model._n_updates==4
    assert any(not torch.equal(v,model.policy.state_dict()[n]) for n,v in before.items())
    actions=model.rollout_buffer.actions.reshape(-1,2)
    assert (np.abs(actions)<=1).all()
    np.testing.assert_allclose(np.asarray(probe.actions[-8:]),actions,atol=2e-7,rtol=0)
    model.save(tmp_path/'model.zip')
    restored=PPO.load(tmp_path/'model.zip',device='cpu')
    assert isinstance(restored.policy,SquashedFlatRoadPolicy) and restored.policy.squash_output
    assert all(torch.equal(p,restored.policy.state_dict()[n]) for n,p in model.policy.state_dict().items())
    obs=np.zeros((2,4705),dtype=np.float32)
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],restored.predict(obs,deterministic=True)[0])


def test_distribution_protocol_rejects_mixed_or_resumed_interventions():
    job={'name':'bounded','method':'B1','gradient_clip_mode':'global','flat_action_distribution':'tanh_gaussian_v1'}
    protocol={'methods':['bounded'],'jobs':[job],'flat_distribution_experiment':True}
    assert registered_jobs(protocol)==[job]
    for change in ({'gradient_clipping_experiment':True},{'rollout_length_experiment':True},{'recovery':{'bounded':{}}}):
        with pytest.raises(ValueError):registered_jobs({**protocol,**change})
    for change in ({'method':'H'},{'rollout_actions':256},{'flat_action_distribution':'unknown'}):
        with pytest.raises(ValueError):registered_jobs({**protocol,'jobs':[{**job,**change}]})


def test_prefix_preserves_measured_points_and_changes_geometry_identity():
    points=[{'x':float(i),'y':0.,'z':0.,'yaw_degrees_carla':0.} for i in range(10)]
    route={'points':points,'map':'test','opendrive_sha256':'map_hash','geometry_id':'original'}
    prefix=short_route(route)
    assert prefix['points']==points[:7] and prefix['geometry_id']!='original'
    assert prefix['parent_geometry_id']=='original' and prefix['sanity_prefix_length_m']==6.
    assert len(route['points'])==10
    with pytest.raises(ValueError):short_route(route,20.)

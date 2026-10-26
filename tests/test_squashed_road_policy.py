import copy
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from navstack.road_candidate_policy import FlatRoadCandidateActorCriticPolicy
from navstack.squashed_road_policy import SquashedFlatRoadPolicy
from scripts.native_road_learning import registered_jobs, training_config
from navstack.config import VehicleConfig
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.vehicle import VehiclePlant
from navstack.experiments import signature
from scripts.prepare_native_flat_distribution_pair import short_route
from scripts.watch_existing_pair import command_identifies_batch


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


def test_guard_resolves_explicit_batch_argument_without_substring_adoption(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    batch=tmp_path/'runs/batch with spaces'
    assert command_identifies_batch('python native_road_learning.py --batch "runs/batch with spaces"',batch)
    assert command_identifies_batch(f'python native_road_learning.py --batch "{batch}"',batch)
    assert not command_identifies_batch(f'python --output "{batch}" --batch runs/other',batch)
    assert not command_identifies_batch(f'python --batch "{batch}" --batch runs/other',batch)


def test_progress_reward_is_explicit_shared_and_legacy_preserving():
    protocol={'episode_seconds':12.,'feedback_ablation':True}
    for method in ('H0','H','B1'):
        legacy=training_config(protocol,method,VehicleConfig())
        changed=training_config({**protocol,'task_progress_weight':1.},method,VehicleConfig())
        assert legacy.reward.progress_weight==0 and changed.reward.progress_weight==1.
        before=legacy.to_dict();after=changed.to_dict()
        before['reward']['progress_weight']=1.
        assert before==after
    for value in (True,'1',-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):training_config({**protocol,'task_progress_weight':value},'B1',VehicleConfig())


def test_flat_speed_semantics_actual_lightweight_control_and_versioning():
    legacy=EnvConfig(method='B1',scenario='empty',obstacle_num=0)
    speed=EnvConfig(method='B1',scenario='empty',obstacle_num=0,flat_longitudinal_mode='speed_target_v1')
    assert 'flat_longitudinal_mode' not in signature(legacy)
    assert signature(speed)['flat_longitudinal_mode']=='speed_target_v1'
    for cfg in (legacy,speed):
        env=NavigationEnv(cfg)
        env.reset(seed=11)
        _,_,_,_,info=env.step(np.array([0.,.2],dtype=np.float32))
        command=info['trace'][0]['requested_command']
        assert command['acceleration']==(0. if cfg is legacy else cfg.vehicle.max_accel)
        assert abs(command['steering']-.1)<1e-7
        if cfg is speed:
            assert env.state.v>0
            env.plant=VehiclePlant(cfg.vehicle,replace(env.plant.state,v=2.))
            _,_,_,_,braking=env.step(np.array([-1.,0.],dtype=np.float32))
            assert braking['trace'][0]['requested_command']['acceleration']==-2.
            assert braking['plan'] is None and braking['mode']=='flat'
        env.close()
    with pytest.raises(ValueError):EnvConfig(method='H',flat_longitudinal_mode='speed_target_v1')

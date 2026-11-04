import numpy as np
import pytest
import torch
from gymnasium import Env,spaces
from stable_baselines3 import PPO
from navstack.road_candidate_policy import EndpointClearRoadPolicy,SharedEndpointAblatedRoadPolicy,endpoint_policy_class
from scripts.audit_native_feedback_dependence import ablate,probabilities
from scripts.native_road_learning import training_config
from navstack.config import VehicleConfig


class NoPlant(Env):
    observation_space=spaces.Box(-10.,10.,(4705,),dtype=np.float32)
    action_space=spaces.Discrete(31)
    def reset(self,**kwargs):raise AssertionError('No plant execution during policy validation')


def pair():
    torch.set_num_threads(1)
    return [PPO(p,NoPlant(),n_steps=8,batch_size=8,seed=11,device='cpu') for p in (EndpointClearRoadPolicy,SharedEndpointAblatedRoadPolicy)]


def test_same_seed_networks_parameters_and_common_physical_mask_are_identical():
    h,h0=pair()
    assert h.policy.state_dict().keys()==h0.policy.state_dict().keys()
    assert all(torch.equal(v,h0.policy.state_dict()[k]) for k,v in h.policy.state_dict().items())
    x=torch.zeros(2,4705);x[:,351+5]=1;x[:,351+8+5]=1;x[:,351+16+5]=1
    with torch.no_grad():
        first=h.policy.mlp_extractor.forward_actor(x);second=h0.policy.mlp_extractor.forward_actor(x)
        assert torch.equal(first[:,-31:],second[:,-31:]) and torch.all(second[:,-31:].sum(1)==3)
        p=h0.policy.get_distribution(x).distribution.probs
        assert torch.all(p[:,3:]==0) and torch.all(p[:,:3]>0)


def test_internal_ablation_hides_feedback_from_actor_and_critic_but_keeps_mask(tmp_path):
    h,h0=pair();raw=np.random.default_rng(11).normal(size=4705).astype(np.float32)
    raw[351:599].reshape(31,8)[:,5:]=1;raw[599:605]=.5
    x=torch.as_tensor(raw[None]);zero=torch.as_tensor(ablate(raw,'zero_all_feedback')[None])
    with torch.no_grad():
        mask=h0.policy.mlp_extractor.forward_actor(x)[:,-31:]>.5
        torch.testing.assert_close(h0.policy.mlp_extractor.forward_actor_unmasked(x),h.policy.mlp_extractor.forward_actor_unmasked(zero),rtol=0,atol=0)
        torch.testing.assert_close(h0.policy.mlp_extractor.forward_critic(x),h.policy.mlp_extractor.forward_critic(zero),rtol=0,atol=0)
        torch.testing.assert_close(probabilities(h0.policy,raw[None],mask),probabilities(h0.policy,zero,mask),rtol=0,atol=0)
    h0.save(tmp_path/'shared_H0.zip');loaded=PPO.load(tmp_path/'shared_H0.zip',device='cpu')
    np.testing.assert_array_equal(loaded.predict(raw[None],deterministic=True)[0],h0.predict(raw[None],deterministic=True)[0])
    assert not loaded.num_timesteps and not loaded.policy.optimizer.state


def test_runtime_shares_all_environment_information_and_keeps_legacy_default():
    p={'episode_seconds':45,'feedback_ablation':True,'continuation_feedback':True,'road_candidate_policy':True,'candidate_head':'unshared_endpoint_v1','subgoal_mode':'route_frame_terminal'}
    assert training_config(p,'H0',VehicleConfig()).execution_feedback=='continuation_ablated'
    p['shared_endpoint_mask_v1']=True
    assert training_config(p,'H',VehicleConfig()).to_dict()==training_config(p,'H0',VehicleConfig()).to_dict()
    assert endpoint_policy_class(p,'H') is EndpointClearRoadPolicy and endpoint_policy_class(p,'H0') is SharedEndpointAblatedRoadPolicy
    assert endpoint_policy_class({},'H0') is EndpointClearRoadPolicy
    for bad in ('true',1):
        with pytest.raises(ValueError):training_config({**p,'shared_endpoint_mask_v1':bad},'H0',VehicleConfig())
    with pytest.raises(ValueError):training_config({**p,'candidate_head':'unshared_v1'},'H0',VehicleConfig())

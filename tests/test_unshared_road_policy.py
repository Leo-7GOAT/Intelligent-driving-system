import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

from navstack.road_candidate_policy import RoadCandidateActorCriticPolicy,UnsharedRoadCandidatePolicy
from scripts.native_road_learning import registered_jobs


class Probe(gym.Env):
    observation_space=gym.spaces.Box(-np.inf,np.inf,(4705,),np.float32)
    action_space=gym.spaces.Discrete(31)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.steps=0
        return np.zeros(4705,dtype=np.float32),{}

    def step(self,action):
        self.steps+=1
        return np.zeros(4705,dtype=np.float32),float(action)/30.,self.steps==8,False,{}


def test_unshared_head_preserves_seeded_encoders_critic_and_rng_and_uses_common_context(tmp_path):
    torch.set_num_threads(1)
    shared=PPO(RoadCandidateActorCriticPolicy,Probe(),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    rng=torch.get_rng_state().clone()
    unshared=PPO(UnsharedRoadCandidatePolicy,Probe(),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    assert torch.equal(rng,torch.get_rng_state())
    for name,p in shared.policy.state_dict().items():
        if not name.startswith('action_net.'):
            assert torch.equal(p,unshared.policy.state_dict()[name])
    obs=torch.ones((2,4705));obs[:,351:599]=0
    gradients=[]
    for model in (shared,unshared):
        value=obs.clone().requires_grad_()
        log_prob=model.policy.get_distribution(value).log_prob(torch.zeros(2,dtype=torch.long)).sum()
        log_prob.backward();gradients.append(float(value.grad[:,:351].abs().sum()))
    assert gradients[0]<1e-7 and gradients[1]>1e-5
    before={k:v.clone() for k,v in unshared.policy.state_dict().items()}
    unshared.learn(16)
    assert unshared._n_updates==4
    assert any(not torch.equal(v,unshared.policy.state_dict()[k]) for k,v in before.items())
    unshared.save(tmp_path/'model.zip');loaded=PPO.load(tmp_path/'model.zip',device='cpu')
    assert isinstance(loaded.policy,UnsharedRoadCandidatePolicy)
    assert all(torch.equal(v,loaded.policy.state_dict()[k]) for k,v in unshared.policy.state_dict().items())
    np.testing.assert_array_equal(unshared.predict(obs.numpy(),deterministic=True)[0],loaded.predict(obs.numpy(),deterministic=True)[0])


def test_actor_head_pair_requires_its_own_registration():
    import pytest
    job={'name':'H_new','method':'H','gradient_clip_mode':'global','candidate_head':'unshared_v1'}
    p={'actor_head_experiment':True,'jobs':[job],'methods':['H_new']}
    assert registered_jobs(p)==[job]
    for change in ({'flat_distribution_experiment':True},{'recovery':{'H_new':{}}}):
        with pytest.raises(ValueError):registered_jobs({**p,**change})
    with pytest.raises(ValueError):registered_jobs({**p,'jobs':[{**job,'method':'B1'}]})

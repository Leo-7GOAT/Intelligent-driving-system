import numpy as np
import pytest
import torch
from gymnasium import Env,spaces
from stable_baselines3 import PPO
from navstack.critic_rate_policy import CriticRateAdam,promote_fresh_critic_rate
from navstack.independent_clip_ppo import load_ppo
from navstack.road_candidate_policy import OrderedFlatRoadCandidatePolicy


class Plant(Env):
    observation_space=spaces.Box(-1.,1.,(4705,),dtype=np.float32)
    action_space=spaces.Box(-1.,1.,(2,),dtype=np.float32)
    def reset(self,**kwargs):self.steps=0;return np.zeros(4705,dtype=np.float32),{}
    def step(self,action):
        self.steps+=1
        return np.zeros(4705,dtype=np.float32),(-90. if self.steps==8 else .1),False,self.steps==8,{}


def test_actual_adam_updates_use_role_rates_and_restore_schedule_base():
    actor=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));critic=torch.nn.Parameter(actor.detach().clone())
    optimizer=CriticRateAdam([{'params':[actor],'learning_rate_multiplier':1.},{'params':[critic],'learning_rate_multiplier':30.}],lr=3e-5)
    actor.grad=torch.ones_like(actor);critic.grad=torch.ones_like(critic);optimizer.step()
    assert abs(float(((1-critic)/(1-actor)).detach())-30)<1e-8
    assert [g['lr'] for g in optimizer.param_groups]==[3e-5,3e-5]


def test_fresh_promotion_preserves_every_prediction_and_real_sb3_training_roundtrip(tmp_path):
    torch.set_num_threads(1)
    model=PPO(OrderedFlatRoadCandidatePolicy,Plant(),n_steps=8,batch_size=8,n_epochs=2,learning_rate=3e-5,seed=11,device='cpu')
    x=torch.zeros(2,4705);before={k:v.clone() for k,v in model.policy.state_dict().items()}
    with torch.no_grad():a=model.policy.get_distribution(x).distribution.mean.clone();v=model.policy.predict_values(x).clone()
    result=promote_fresh_critic_rate(model)
    assert result['critic_learning_rate']==30*3e-5 and all(torch.equal(t,model.policy.state_dict()[k]) for k,t in before.items())
    with torch.no_grad():
        torch.testing.assert_close(a,model.policy.get_distribution(x).distribution.mean,rtol=0,atol=0)
        torch.testing.assert_close(v,model.policy.predict_values(x),rtol=0,atol=0)
    model.learn(8)
    assert model.num_timesteps==8 and isinstance(model.policy.optimizer,CriticRateAdam)
    assert [g['last_step_effective_lr'] for g in model.policy.optimizer.param_groups]==[3e-5,30*3e-5]
    model.save(tmp_path/'trained.zip');loaded=load_ppo(tmp_path/'trained.zip',device='cpu')
    assert isinstance(loaded.policy.optimizer,CriticRateAdam) and loaded.num_timesteps==8
    assert [g['learning_rate_multiplier'] for g in loaded.policy.optimizer.param_groups]==[1.,30.]
    assert [g['last_step_effective_lr'] for g in loaded.policy.optimizer.param_groups]==[3e-5,30*3e-5]
    assert all(torch.equal(t,loaded.policy.state_dict()[k]) for k,t in model.policy.state_dict().items())
    with pytest.raises(ValueError):promote_fresh_critic_rate(loaded)

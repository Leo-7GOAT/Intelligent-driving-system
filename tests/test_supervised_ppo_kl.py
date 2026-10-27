import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from scripts.prepare_native_supervised_ppo import fresh_target_kl
from test_supervised_ppo import NoExecutionEnv


def test_kl_registration_survives_reload_without_policy_or_optimizer_mutation(tmp_path):
    model=PPO('MlpPolicy',NoExecutionEnv(),n_steps=8,batch_size=8,seed=11,device='cpu')
    before={k:v.clone() for k,v in model.policy.state_dict().items()}
    info=fresh_target_kl(model,.01)
    assert info['previous_target_kl'] is None
    assert all(torch.equal(before[k],v) for k,v in model.policy.state_dict().items())
    assert not model.policy.optimizer.state and model.num_timesteps==model._n_updates==0
    model.save(tmp_path/'initial.zip')
    assert PPO.load(tmp_path/'initial.zip',device='cpu').target_kl==.01


@pytest.mark.parametrize('fault',[True,0.,.051,float('nan'),'trained'])
def test_invalid_or_previously_trained_kl_changes_rejected(fault):
    model=PPO('MlpPolicy',NoExecutionEnv(),n_steps=8,batch_size=8,seed=11,device='cpu')
    if fault=='trained':model.num_timesteps=8;fault=.01
    with pytest.raises(ValueError):fresh_target_kl(model,fault)
    assert model.target_kl is None


def test_actual_sb3_kl_stop_limits_adam_epochs_without_shortening_collection():
    from gymnasium import spaces
    class Plant(NoExecutionEnv):
        def reset(self,**kwargs):
            super(NoExecutionEnv,self).reset(**kwargs)
            self.steps=0
            return np.zeros(3,dtype=np.float32),{}
        def step(self,action):
            self.steps+=1
            return np.array([self.steps/8,0.,0.],dtype=np.float32),float(action[0]),False,self.steps==8,{}
    counts=[]
    for target in (None,.001):
        model=PPO('MlpPolicy',Plant(),n_steps=8,batch_size=8,n_epochs=10,learning_rate=.01,seed=11,device='cpu')
        if target is not None:fresh_target_kl(model,target)
        steps=[]
        handle=model.policy.optimizer.register_step_post_hook(lambda *args:steps.append(1))
        model.learn(8);handle.remove()
        assert model.num_timesteps==8
        counts.append(len(steps))
    assert counts[0]==10 and 0<counts[1]<counts[0]

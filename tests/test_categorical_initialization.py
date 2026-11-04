import numpy as np
import torch
import pytest
from gymnasium import Env,spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from navstack.imitation import fit_categorical_initialization
from navstack.independent_clip_ppo import actor_critic_parameters


class NoExecutionEnv(Env):
    observation_space=spaces.Box(-1e6,1e6,(3,),dtype=np.float32)
    action_space=spaces.Discrete(3)
    def reset(self,**kwargs):raise AssertionError('Supervision must not execute a plant')


def test_categorical_teaching_leaves_critic_ppo_state_and_holdout_statistics_untouched():
    raw=DummyVecEnv([NoExecutionEnv]);norm=VecNormalize(raw,norm_reward=False)
    model=PPO('MlpPolicy',norm,n_steps=8,batch_size=8,seed=11,device='cpu')
    _,critic=actor_critic_parameters(model.policy);before=[p.detach().clone() for p in critic]
    observations=np.array([[0,0,0],[1,1,1],[1e5,1e5,1e5]],dtype=np.float32)
    rows=fit_categorical_initialization(model,norm,observations,np.array([0,1,2]),[True,True,False],steps=3)
    assert rows[-1]['supervised_optimizer_steps']==3
    assert abs(norm.obs_rms.count-2.0001)<1e-8 and np.max(norm.obs_rms.mean)<1.
    assert not model.policy.optimizer.state and model.num_timesteps==model._n_updates==0
    assert all(torch.equal(p,v) for p,v in zip(critic,before))
    with pytest.raises(ValueError):fit_categorical_initialization(model,norm,observations,np.array([0,1,2]),[True]*3,steps=1)
    with pytest.raises(ValueError):fit_categorical_initialization(model,norm,observations,np.array([0,1,3]),[True,True,False],steps=1)
    with pytest.raises(ValueError):fit_categorical_initialization(model,norm,observations,np.array([[0],[1],[2]]),[[True],[True],[False]],steps=1)
    norm.close()

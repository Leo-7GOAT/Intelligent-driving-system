import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import actor_vector
from navstack.imitation import fit_demonstrations


def test_supervision_excludes_holdout_statistics_and_is_not_ppo_update(tmp_path):
    torch.set_num_threads(1)
    cfg=EnvConfig(method='B1',scenario='empty',obstacle_num=0)
    env=NavigationEnv(cfg);observations=[];actions=[]
    for seed in (8101,8102):
        obs,_=env.reset(seed=seed)
        for i in range(8):
            action=np.array([.4,.1 if i<4 else -.1],dtype=np.float32)
            observations.append(obs.copy());actions.append(action)
            obs,_,_,_,_=env.step(action)
    env.close()
    raw=DummyVecEnv([lambda:NavigationEnv(cfg)]);norm=VecNormalize(raw,norm_reward=False)
    model=PPO('MlpPolicy',norm,n_steps=8,batch_size=8,seed=11,device='cpu')
    before=actor_vector(model).clone();obs=np.array(observations);mask=np.arange(16)<8
    # A distinct heldout value makes accidental inclusion in RMS unambiguous.
    obs[~mask,0]=1000
    history=fit_demonstrations(model,norm,obs,np.array(actions),np.zeros(16,dtype=np.float32),mask,epochs=5)
    assert abs(norm.obs_rms.count-(8+1e-4))<1e-8
    assert abs(norm.obs_rms.mean[0]-obs[mask,0].mean())<1e-3
    assert not torch.equal(before,actor_vector(model))
    assert model._n_updates==0 and not model.policy.optimizer.state
    assert history[-1]['supervised_optimizer_steps']==5
    model.save(tmp_path/'supervised')
    loaded=PPO.load(tmp_path/'supervised')
    for key,value in model.policy.state_dict().items():assert torch.equal(value,loaded.policy.state_dict()[key])
    norm.close()

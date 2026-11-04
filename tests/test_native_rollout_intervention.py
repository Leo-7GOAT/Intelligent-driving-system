import copy
import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize

from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import signature,write_json
from navstack.independent_clip_ppo import load_ppo
from scripts.continue_native_learning import _equal
from scripts.native_road_learning import registered_jobs,restore_training,save_training_checkpoint


def test_rollout_change_preserves_actual_policy_adam_rng_then_collects_full_buffer(tmp_path):
    torch.set_num_threads(1)
    cfg=EnvConfig(method='B1',scenario='empty',flat_hold=10)
    raw=DummyVecEnv([lambda:NavigationEnv(cfg)]);raw.seed(11);raw.envs[0].case_index=2
    vec=VecNormalize(raw,norm_reward=False,gamma=cfg.gamma)
    model=PPO('MlpPolicy',vec,n_steps=8,batch_size=8,n_epochs=2,seed=11,device='cpu')
    model.learn(16)
    write_json(tmp_path/'config.json',{'env':cfg.to_dict(),'signature':signature(cfg)})
    save_training_checkpoint(tmp_path,160,model,vec,raw,{'physics_steps':160,'episodes':0})
    checkpoint=tmp_path/'checkpoint_000000160'
    with pytest.raises(ValueError,match='explicit experimental'):
        load_ppo(checkpoint/'model.zip',rollout_actions=16)
    with pytest.raises(ValueError,match='implicit n_steps'):
        load_ppo(checkpoint/'model.zip',n_steps=16)
    with pytest.raises(ValueError,match='multiple'):
        load_ppo(checkpoint/'model.zip',rollout_actions=15,allow_rollout_intervention=True)
    expected_rng=torch.get_rng_state().clone()
    expected_optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
    new_raw=DummyVecEnv([lambda:NavigationEnv(cfg)])
    loaded,norm,_=restore_training(checkpoint,new_raw,cfg,99,rollout_actions=16,allow_rollout_intervention=True)
    try:
        assert loaded.n_steps==loaded.rollout_buffer.buffer_size==16
        assert loaded.batch_size==8 and loaded.n_epochs==2 and loaded.num_timesteps==model.num_timesteps
        assert _equal(model.policy.state_dict(),loaded.policy.state_dict())
        assert _equal(expected_optimizer,loaded.policy.optimizer.state_dict())
        assert torch.equal(expected_rng,torch.get_rng_state())
        np.testing.assert_array_equal(norm.obs_rms.mean,vec.obs_rms.mean)
        steps=[]
        hook=loaded.policy.optimizer.register_step_post_hook(lambda *args:steps.append(1))
        before=loaded.num_timesteps
        loaded.learn(16,reset_num_timesteps=False)
        hook.remove()
        assert loaded.num_timesteps-before==16 and len(steps)==4
        loaded.save(tmp_path/'long.zip')
        assert load_ppo(tmp_path/'long.zip').n_steps==16
    finally:
        norm.close();vec.close()


def test_registered_rollout_pair_cannot_also_change_clipping():
    jobs=[{'name':'H_'+str(n),'method':'H','gradient_clip_mode':'global','rollout_actions':n} for n in (32,256)]
    protocol={'methods':[j['name'] for j in jobs],'jobs':jobs,'rollout_length_experiment':True}
    assert registered_jobs(protocol)==jobs
    with pytest.raises(ValueError,match='exactly one'):
        registered_jobs({**protocol,'gradient_clipping_experiment':True})
    changed=copy.deepcopy(protocol);changed['jobs'][1]['gradient_clip_mode']='independent_actor_critic_v1'
    with pytest.raises(ValueError,match='retain global clipping'):
        registered_jobs(changed)

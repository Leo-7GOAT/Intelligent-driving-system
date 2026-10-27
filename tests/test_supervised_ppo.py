import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from scripts.prepare_native_supervised_ppo import initial_exploration, guarded_initialization, categorical_exploration, fresh_learning_rate, fresh_rollout


class NoExecutionEnv(Env):
    observation_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
    action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
    def reset(self, **kwargs):
        raise AssertionError('Parameter initialization must not reset or execute a plant')


@pytest.mark.parametrize('log_std', [-2.3, -4.])
def test_exploration_initialization_preserves_every_mean_critic_and_optimizer_byte(log_std):
    model = PPO('MlpPolicy', NoExecutionEnv(), n_steps=8, batch_size=8, seed=11, device='cpu')
    before = {name: value.detach().clone() for name,value in model.policy.named_parameters()}
    change = initial_exploration(model, log_std)
    assert change['parameter_changes'] == ['log_std']
    for name, value in model.policy.named_parameters():
        if name != 'log_std': assert torch.equal(before[name], value)
    assert torch.allclose(model.policy.log_std, torch.tensor([log_std,log_std]))
    assert not model.policy.optimizer.state and model.num_timesteps == model._n_updates == 0


@pytest.mark.parametrize('fault', ['interactions', 'updates', 'range'])
def test_supervised_initialization_cannot_reset_a_previously_trained_ppo_distribution(fault):
    model = PPO('MlpPolicy', NoExecutionEnv(), n_steps=8, batch_size=8, seed=11, device='cpu')
    if fault == 'interactions': model.num_timesteps = 1
    elif fault == 'updates': model._n_updates = 1
    with pytest.raises(ValueError): initial_exploration(model, 0. if fault == 'range' else -2.3)


def test_guarded_reward_initialization_preserves_all_nonreward_semantics_and_original_metadata(tmp_path):
    from copy import deepcopy
    from test_native_hazard_learning import parent
    from scripts.prepare_native_hazard_learning import learning_protocol
    from scripts.native_road_learning import training_config
    from navstack.config import VehicleConfig
    from navstack.experiments import signature
    parent(tmp_path)
    protocol=learning_protocol(tmp_path,20480,7200,11,['B1'])
    protocol.update(flat_hold=1,flat_action_catalog='continuous',flat_longitudinal_mode='acceleration')
    config=training_config(protocol,'B1',VehicleConfig())
    metadata={'env':config.to_dict(),'signature':signature(config),'seed':11}
    original=deepcopy(metadata)
    changed=guarded_initialization(protocol,metadata)
    assert metadata==original and changed['signature']!=original['signature']
    before,after=original['env'].copy(),changed['env'].copy()
    old_reward,new_reward=before.pop('reward'),after.pop('reward')
    assert before==after
    assert new_reward=={**old_reward,'progress_weight':.5,'collision':-90.,'out_of_bounds':-90.}
    assert protocol['reward_intervention']['version']=='guarded_progress_v1'
    broken=deepcopy(original);broken['signature']={}
    with pytest.raises(ValueError,match='signature'):guarded_initialization(protocol,broken)


def test_categorical_exploration_calibrates_train_views_without_resetting_mean_order_or_critic():
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    class CategoricalEnv(NoExecutionEnv):
        action_space=spaces.Discrete(3)
    raw=DummyVecEnv([CategoricalEnv]);norm=VecNormalize(raw,norm_reward=False)
    model=PPO('MlpPolicy',norm,n_steps=8,batch_size=8,seed=11,device='cpu')
    with torch.no_grad():model.policy.action_net.bias.copy_(torch.tensor([8.,0.,0.]))
    before={name:p.detach().clone() for name,p in model.policy.named_parameters()}
    rms=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
    changed=categorical_exploration(model,norm,np.zeros((6,3),dtype=np.float32),.9)
    assert changed['temperature']>1 and abs(changed['initialized_mean_max_probability']-.9)<1e-4
    assert changed['deterministic_calibration_actions_unchanged']
    assert set(changed['parameter_changes'])<= {'action_net.weight','action_net.bias'}
    assert all(torch.equal(before[name],p) for name,p in model.policy.named_parameters() if not name.startswith('action_net.'))
    np.testing.assert_array_equal(norm.obs_rms.mean,rms[0]);np.testing.assert_array_equal(norm.obs_rms.var,rms[1])
    assert norm.obs_rms.count==rms[2] and not model.policy.optimizer.state and not model.num_timesteps
    norm.close()


def test_lower_learning_rate_survives_checkpoint_loading_without_changing_policy_or_running_plant(tmp_path):
    model=PPO('MlpPolicy',NoExecutionEnv(),n_steps=8,batch_size=8,seed=11,device='cpu')
    before={name:p.detach().clone() for name,p in model.policy.named_parameters()}
    info=fresh_learning_rate(model,3e-5)
    assert info['initialized_learning_rate']==3e-5
    assert all(torch.equal(before[name],p) for name,p in model.policy.named_parameters())
    assert not model.policy.optimizer.state and model.num_timesteps==model._n_updates==0
    model.save(tmp_path/'model.zip');loaded=PPO.load(tmp_path/'model.zip',device='cpu')
    assert all(loaded.lr_schedule(progress)==3e-5 for progress in (0.,.5,1.))
    assert all(group['lr']==3e-5 for group in loaded.policy.optimizer.param_groups)
    loaded.num_timesteps=1
    with pytest.raises(ValueError):fresh_learning_rate(loaded,3e-5)


def test_longer_initial_rollout_rebuilds_buffer_and_preserves_actor_critic_and_checkpoint(tmp_path):
    import hashlib
    model=PPO('MlpPolicy',NoExecutionEnv(),n_steps=8,batch_size=8,seed=11,device='cpu')
    path=tmp_path/'initial.zip';model.save(path)
    original_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    changed,info=fresh_rollout(path,32)
    assert info['previous_rollout_actions']==8 and info['initialized_rollout_actions']==32
    assert changed.n_steps==changed.rollout_buffer.buffer_size==32
    assert changed.batch_size==model.batch_size and changed.n_epochs==model.n_epochs
    assert changed.gamma==model.gamma and changed.gae_lambda==model.gae_lambda
    assert not changed.rollout_buffer.full and changed.rollout_buffer.pos==0
    assert changed.num_timesteps==changed._n_updates==0 and not changed.policy.optimizer.state
    assert all(torch.equal(value,changed.policy.state_dict()[name]) for name,value in model.policy.state_dict().items())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original_sha
    changed.save(tmp_path/'long.zip');loaded=PPO.load(tmp_path/'long.zip',device='cpu')
    assert loaded.n_steps==loaded.rollout_buffer.buffer_size==32
    for invalid in (8,33,True,4097):
        with pytest.raises(ValueError):fresh_rollout(path,invalid)
    model._n_updates=1;model.save(tmp_path/'updated.zip')
    with pytest.raises(ValueError):fresh_rollout(tmp_path/'updated.zip',32)

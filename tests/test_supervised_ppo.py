import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from scripts.prepare_native_supervised_ppo import initial_exploration, guarded_initialization


class NoExecutionEnv(Env):
    observation_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
    action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
    def reset(self, **kwargs):
        raise AssertionError('Parameter initialization must not reset or execute a plant')


def test_exploration_initialization_preserves_every_mean_critic_and_optimizer_byte():
    model = PPO('MlpPolicy', NoExecutionEnv(), n_steps=8, batch_size=8, seed=11, device='cpu')
    before = {name: value.detach().clone() for name,value in model.policy.named_parameters()}
    change = initial_exploration(model, -2.3)
    assert change['parameter_changes'] == ['log_std']
    for name, value in model.policy.named_parameters():
        if name != 'log_std': assert torch.equal(before[name], value)
    assert torch.allclose(model.policy.log_std, torch.tensor([-2.3,-2.3]))
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

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from scripts.prepare_native_supervised_ppo import initial_exploration


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

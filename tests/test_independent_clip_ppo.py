import copy
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import actor_vector
from navstack.independent_clip_ppo import IndependentClipPPO, actor_critic_parameters, load_ppo, INDEPENDENT_CLIP


def test_actual_ppo_actor_update_independent_of_value_target_scale_and_reload(tmp_path):
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: NavigationEnv(EnvConfig(method='B1', scenario='empty', flat_hold=10, deadline=4))])
    try:
        model = IndependentClipPPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=2, seed=11)
        initial = actor_vector(model).clone()
        model.learn(16)
        assert torch.linalg.vector_norm(actor_vector(model)-initial) > 0
        assert model.max_grad_norm == .5
        assert not model.policy.optimizer._optimizer_step_pre_hooks
        model.save(tmp_path/'model.zip')
        restored = load_ppo(tmp_path/'model.zip', device='cpu')
        assert isinstance(restored, IndependentClipPPO)
        assert restored.gradient_clip_mode == INDEPENDENT_CLIP
        assert all(torch.equal(v, restored.policy.state_dict()[k]) for k, v in model.policy.state_dict().items())
        actors, critics = [], []
        for offset in (0., 1000.):
            branch = load_ppo(tmp_path/'model.zip', device='cpu')
            branch.rollout_buffer = copy.deepcopy(model.rollout_buffer)
            branch.rollout_buffer.returns += offset
            branch.set_logger(configure(None, []))
            np.random.seed(123)
            branch.train()
            actors.append(actor_vector(branch).clone())
            critics.append(torch.cat([p.flatten() for p in actor_critic_parameters(branch.policy)[1]]))
            assert branch.max_grad_norm == .5
            assert not branch.policy.optimizer._optimizer_step_pre_hooks
        torch.testing.assert_close(actors[0], actors[1], rtol=0, atol=0)
        assert not torch.equal(critics[0], critics[1])
        # Legacy checkpoints remain ordinary PPO; loading does not opt in.
        standard = PPO('MlpPolicy', env, n_steps=8, batch_size=8, seed=11)
        standard.save(tmp_path/'standard.zip')
        assert type(load_ppo(tmp_path/'standard.zip')) is PPO
        with pytest.raises(ValueError, match='explicit experimental intervention'):
            load_ppo(tmp_path/'standard.zip', gradient_clip_mode=INDEPENDENT_CLIP)
        changed = load_ppo(tmp_path/'standard.zip', gradient_clip_mode=INDEPENDENT_CLIP, allow_intervention=True)
        assert type(changed) is IndependentClipPPO
        assert all(torch.equal(v, changed.policy.state_dict()[k]) for k, v in standard.policy.state_dict().items())
    finally:
        env.close()


def test_clipping_hook_cleanup_on_failed_training_and_shared_parameter_rejection(tmp_path, monkeypatch):
    env = DummyVecEnv([lambda: NavigationEnv(EnvConfig(method='B1', scenario='empty'))])
    try:
        model = IndependentClipPPO('MlpPolicy', env, n_steps=8, batch_size=8, seed=11)
        model.set_logger(configure(None, []))
        monkeypatch.setattr(PPO, 'train', lambda self: (_ for _ in ()).throw(RuntimeError('injected')))
        with pytest.raises(RuntimeError, match='injected'):
            model.train()
        assert model.max_grad_norm == .5
        assert not model.policy.optimizer._optimizer_step_pre_hooks
        model.policy.value_net.weight = model.policy.action_net.weight
        with pytest.raises(ValueError, match='disjoint'):
            model.train()
        model.policy.register_parameter('unclassified_shared_weight', torch.nn.Parameter(torch.ones(1)))
        with pytest.raises(ValueError, match='disjoint'):
            model.train()
    finally:
        env.close()

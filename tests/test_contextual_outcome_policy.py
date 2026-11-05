import pytest
import torch

from navstack.contextual_outcome_policy import ConditionalOutcomeBias, migrate_learner


def test_conditional_feedback_can_learn_opposite_preferences_in_different_states():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    head = ConditionalOutcomeBias(33, scale=1.)
    latent = torch.zeros(2, 33)
    latent[:, 0] = torch.tensor([-1., 1.])
    latent[:, -29] = 1.
    assert not torch.count_nonzero(head(latent))
    optimizer = torch.optim.Adam(head.parameters(), lr=.02)
    for _ in range(50):
        optimizer.zero_grad()
        loss = ((head(latent)[:, 2] - torch.tensor([-1., 1.])) ** 2).mean()
        loss.backward()
        optimizer.step()
    assert head(latent)[0, 2] < -.7 and head(latent)[1, 2] > .7
    latent[:, -31:] = 0.
    assert not torch.count_nonzero(head(latent))


@pytest.mark.parametrize('ablated', [False, True])
def test_migration_keeps_old_tensors_adam_rng_and_roundtrips_real_ppo(tmp_path, ablated):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from navstack.config import EnvConfig
    from navstack.outcome_context_policy import OutcomeContextPolicy, OutcomeContextObservationMixin
    from navstack.reference_context_policy import ReferenceContextVecNormalize
    from navstack.independent_clip_ppo import actor_critic_parameters
    from test_route_catalog_environment import RouteFixture, config
    torch.set_num_threads(1)

    class Fixture(OutcomeContextObservationMixin, RouteFixture):
        def reset(self, **kwargs):
            _, info = super().reset(**kwargs)
            # Deliberately expose a genuine reusable reference at collection
            # start. Cold-start four-step rollouts otherwise contain no feedback.
            mode, _ = self._prepare_reference(26)
            assert mode == 'reference'
            observation = self._observation()
            assert torch.count_nonzero(torch.as_tensor(observation[-31:]))
            return observation, info

    cfg = EnvConfig.from_dict({**config(feedback='continuation').to_dict(),
                              'route_goal_candidate': True, 'reference_reuse': True})
    norm = ReferenceContextVecNormalize(DummyVecEnv([lambda: Fixture(cfg)]), norm_reward=False)
    old = PPO(OutcomeContextPolicy, norm, n_steps=2, batch_size=2, n_epochs=1,
              device='cpu', seed=11)
    old.learn(4)
    norm.training = False
    observation = torch.as_tensor(norm.reset()).repeat(3, 1)
    observation[:, -31:] = 0.
    observation[0, -29], observation[1, -28] = 1., -1.
    rng = torch.get_rng_state().clone()
    new = migrate_learner(old, ablated=ablated)
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        assert torch.equal(old.policy.get_distribution(observation).distribution.probs,
                           new.policy.get_distribution(observation).distribution.probs)
        assert torch.equal(old.policy.predict_values(observation), new.policy.predict_values(observation))
    before = dict(old.policy.named_parameters())
    after = dict(new.policy.named_parameters())
    assert len(before) == 34 and len(after) == 42
    for name, parameter in after.items():
        state = new.policy.optimizer.state[parameter]
        if name in before:
            assert torch.equal(parameter, before[name])
            for key, value in old.policy.optimizer.state[before[name]].items():
                assert torch.equal(value, state[key]) if torch.is_tensor(value) else value == state[key]
        else:
            assert state['step'] == 0 and not torch.count_nonzero(state['exp_avg'])
    actor, critic = actor_critic_parameters(new.policy)
    assert not {id(p) for p in actor} & {id(p) for p in critic}
    new.save(tmp_path / 'model.zip')
    restored = PPO.load(tmp_path / 'model.zip', env=norm, device='cpu')
    with torch.no_grad():
        assert torch.equal(new.policy.get_distribution(observation).distribution.probs,
                           restored.policy.get_distribution(observation).distribution.probs)
    original = {k: v.clone() for k, v in old.policy.state_dict().items()}
    branch_before = {k: v.clone() for k, v in restored.policy.state_dict().items()
                     if 'contextual_' in k}
    contexts = []
    def observe(local, global_):
        contexts.append(local['self']._last_obs[:, -31:].copy())
        return True
    restored.learn(4, reset_num_timesteps=False, callback=observe)
    assert all(torch.equal(v, original[k]) for k, v in old.policy.state_dict().items())
    changes = [not torch.equal(v, restored.policy.state_dict()[k]) for k, v in branch_before.items()]
    assert any((context != 0).any() for context in contexts)
    assert any(changes) if not ablated else not any(changes)
    assert restored.policy.contextual_ablated == ablated
    # This is four lightweight fixture interactions, not native training evidence.
    assert restored.num_timesteps == old.num_timesteps + 4
    from navstack.contextual_outcome_policy import (
        VERSION, registered, validate_native_registration,
    )
    profile = dict(version=VERSION, hidden=32, actor_scale=100., parameter_states=42,
                   new_parameter_states=8, new_supervision=0, ablated=ablated)
    protocol = dict(contextual_feedback_registration=profile)
    assert registered(protocol)
    validate_native_registration(protocol, restored)
    with pytest.raises(ValueError):
        validate_native_registration({}, restored)
    with pytest.raises(ValueError):
        validate_native_registration(protocol, old)
    with pytest.raises(ValueError):
        validate_native_registration(dict(contextual_feedback_registration={**profile, 'ablated':not ablated}), restored)
    norm.close()


def test_ablation_removes_new_branch_even_after_nonzero_parameters_and_masks_survive():
    from types import SimpleNamespace
    from torch import nn
    from navstack.contextual_outcome_policy import ContextualActionHead
    from navstack.outcome_context_policy import LearnedOutcomeBias
    original = SimpleNamespace(backbone=nn.Linear(2, 31), reference_bias=LearnedOutcomeBias(1., 1.))
    with torch.no_grad():
        original.backbone.weight.zero_()
        original.backbone.bias.zero_()
        original.backbone.bias[3] = -1e9
    head = ContextualActionHead(original, 33, scale=100., ablated=False)
    latent = torch.zeros(2, 33)
    latent[:, -28] = 1.
    with torch.no_grad():
        head.contextual_bias.net[-1].bias.fill_(100.)
    assert torch.all(head(latent)[:, 3] == -1e9)
    head.contextual_bias.ablated = True
    assert torch.equal(head(latent), original.backbone(latent[:, :-31]))
    latent[:, -31] = 1.
    with pytest.raises(ValueError):
        head(latent)

"""Opt-in independent actor/critic clipping using the unchanged SB3 PPO loss.

The native structured policies have disjoint actor and critic parameters. A
large value gradient should not impose its clipping factor on the smaller actor
gradient. During train(), SB3's global bound is temporarily disabled, then an
optimizer pre-step hook clips the two disjoint sets at the registered bound.
This uses no process-global monkeypatch and preserves Adam state and PPO's
surrogate, epochs, batches, learning rate, entropy and KL early stopping.
"""
import math

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file

INDEPENDENT_CLIP = 'independent_actor_critic_v1'
GLOBAL_CLIP = 'global'


def actor_critic_parameters(policy):
    actor, critic, unknown = [], [], []
    # Keep aliases while checking ownership: default named_parameters deduplicates
    # a tensor shared by actor and critic and would hide that unsafe coupling.
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        is_actor = 'policy_net' in name or 'action_net' in name or name == 'log_std'
        is_critic = 'value_net' in name
        if is_actor == is_critic:
            unknown.append(name)
        elif is_actor:
            actor.append(parameter)
        else:
            critic.append(parameter)
    if unknown or not actor or not critic or {id(p) for p in actor} & {id(p) for p in critic}:
        raise ValueError('Independent clipping requires entirely disjoint actor/critic parameters: ' + ', '.join(unknown))
    actor = list({id(p): p for p in actor}.values())
    critic = list({id(p): p for p in critic}.values())
    return actor, critic


class IndependentClipPPO(PPO):
    def __init__(self, *args, **kwargs):
        self.gradient_clip_mode = INDEPENDENT_CLIP
        super().__init__(*args, **kwargs)

    def train(self):
        bound = self.max_grad_norm
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError('Independent clipping requires a finite positive bound')
        actor, critic = actor_critic_parameters(self.policy)
        norms = []

        def clip_before_step(optimizer, args, kwargs):
            a = torch.nn.utils.clip_grad_norm_(actor, bound, error_if_nonfinite=True)
            c = torch.nn.utils.clip_grad_norm_(critic, bound, error_if_nonfinite=True)
            norms.append((float(a), float(c)))

        hook = self.policy.optimizer.register_step_pre_hook(clip_before_step)
        self.max_grad_norm = float('inf')
        try:
            super().train()
        finally:
            self.max_grad_norm = bound
            hook.remove()
        if norms:
            values = np.asarray(norms)
            self.logger.record('train/actor_gradient_norm_mean', float(values[:, 0].mean()))
            self.logger.record('train/critic_gradient_norm_mean', float(values[:, 1].mean()))
            self.logger.record('train/actor_clip_fraction', float((values[:, 0] > bound).mean()))
            self.logger.record('train/critic_clip_fraction', float((values[:, 1] > bound).mean()))
        self.logger.record('train/independent_actor_critic_clip', 1.)


def load_ppo(path, *, gradient_clip_mode=None, allow_intervention=False, **kwargs):
    """Read the saved algorithm marker before constructing a resumable learner."""
    data, _, _ = load_from_zip_file(path, device=kwargs.get('device', 'auto'))
    saved_mode = data.get('gradient_clip_mode', GLOBAL_CLIP)
    mode = saved_mode if gradient_clip_mode is None else gradient_clip_mode
    if saved_mode not in (GLOBAL_CLIP, INDEPENDENT_CLIP) or mode not in (GLOBAL_CLIP, INDEPENDENT_CLIP):
        raise ValueError('Unknown PPO gradient clipping mode')
    if mode != saved_mode and not allow_intervention:
        raise ValueError('Changing saved PPO clipping mode requires an explicit experimental intervention')
    cls = IndependentClipPPO if mode == INDEPENDENT_CLIP else PPO
    model = cls.load(path, **kwargs)
    model.gradient_clip_mode = mode
    return model

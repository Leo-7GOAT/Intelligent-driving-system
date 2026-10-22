"""Shared candidate scoring for standard PPO; no masks or rule decisions.

The first 351 observation entries keep the existing common information. The
next 31 x 8 entries hold scaled geometry and optional feasibility witnesses;
the last six contain previous execution outcomes. H0 zeros only witnesses and
outcomes, retaining exactly the same network and candidate geometry.
"""
from functools import partial

import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize


class CandidateVecNormalize(VecNormalize):
    """Normalize global state; retain common fixed scales for candidate slots.

    Independent running statistics per slot would destroy permutation
    equivariance and make identical targets have different representations.
    Running statistics are still saved; candidate dimensions bypass them.
    """
    def _normalize_obs(self, obs, obs_rms):
        normalized = super()._normalize_obs(obs, obs_rms)
        normalized[..., 351:599] = obs[..., 351:599]
        return normalized

    def _unnormalize_obs(self, obs, obs_rms):
        original = super()._unnormalize_obs(obs, obs_rms)
        original[..., 351:599] = obs[..., 351:599]
        return original


class CandidateExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim_pi = 31 * 32
        self.latent_dim_vf = 64
        # Names retain actor_vector's policy_net accounting convention.
        self.policy_net_context = nn.Sequential(nn.Linear(357, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.policy_net_candidate = nn.Sequential(nn.Linear(72, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())
        self.value_net = nn.Sequential(nn.Linear(605, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())

    def forward_actor(self, features):
        context = self.policy_net_context(torch.cat((features[:, :351], features[:, 599:]), dim=1))
        candidates = features[:, 351:599].reshape(-1, 31, 8)
        joined = torch.cat((context[:, None, :].expand(-1, 31, -1), candidates), dim=2)
        return self.policy_net_candidate(joined).flatten(1)

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class SharedActionScores(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Linear(32, 1)

    def forward(self, latent):
        return self.score(latent.reshape(-1, 31, 32)).squeeze(-1)


class CandidateActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        if self.observation_space.shape != (605,) or not isinstance(self.action_space, spaces.Discrete) or self.action_space.n != 31:
            raise ValueError("Candidate policy requires 605 inputs and 31 categorical actions")
        self.mlp_extractor = CandidateExtractor().to(self.device)
        self.action_net = SharedActionScores().to(self.device)
        self.value_net = nn.Linear(64, 1).to(self.device)
        if self.ortho_init:
            for module, gain in ((self.mlp_extractor, np.sqrt(2)), (self.action_net, .01), (self.value_net, 1.)):
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

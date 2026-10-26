"""Flat PPO with categorical direct speed/steering, no planner or MPC."""
from functools import partial
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from .road_candidate_policy import FlatRoadCandidateActorCriticPolicy,RoadCandidateExtractor


class DiscreteFlatRoadPolicy(FlatRoadCandidateActorCriticPolicy):
    def _build_mlp_extractor(self):
        if (self.observation_space.shape not in ((4701,),(4705,)) or
                not isinstance(self.action_space,spaces.MultiDiscrete) or
                not np.array_equal(self.action_space.nvec,[3,7])):
            raise ValueError('Require common road inputs and the3x7direct control catalog')
        self.mlp_extractor=RoadCandidateExtractor(flat=True,
            actuator_inputs=self.observation_space.shape[0]-4701).to(self.device)

    def _build(self,lr_schedule):
        distribution=self.action_dist
        # Reproduce the Gaussian control's encoder/critic initialization and
        # global RNG exactly; only replace its output distribution/head.
        self.action_dist=DiagGaussianDistribution(2)
        super()._build(lr_schedule)
        with torch.random.fork_rng(devices=[]):
            self.action_net=nn.Linear(64,10).to(self.device)
            if self.ortho_init:self.action_net.apply(partial(self.init_weights,gain=.01))
        self.action_dist=distribution
        del self.log_std
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

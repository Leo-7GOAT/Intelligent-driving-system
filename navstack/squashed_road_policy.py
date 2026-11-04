"""Opt-in bounded flat PPO actions with the tanh change-of-variables density.

This changes the sampled distribution, not just its mean. There is no planner,
teacher, rule action, safety override or inherited policy. The existing flat
road encoder and critic remain unchanged.
"""
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from .road_candidate_policy import FlatRoadCandidateActorCriticPolicy


class SquashedFlatRoadPolicy(FlatRoadCandidateActorCriticPolicy):
    def __init__(self, *args, squash_output=True, use_sde=False, **kwargs):
        if use_sde or not squash_output:
            raise ValueError('This policy requires tanh Gaussian actions without gSDE')
        # The base constructor only permits its squash flag for gSDE. Our
        # explicit distribution below supplies that behavior for a diagonal
        # Gaussian; expose the flag after construction for SB3 unscaling.
        super().__init__(*args, squash_output=False, use_sde=False, **kwargs)
        self._squash_output = True

    def _build(self, lr_schedule):
        if self.action_space.shape != (2,):
            raise ValueError('Expected two normalized direct actuator commands')
        if (self.action_space.low != -1).any() or (self.action_space.high != 1).any():
            raise ValueError('Tanh actuator policy requires the existing [-1,1] action space')
        self.action_dist = SquashedDiagGaussianDistribution(2)
        super()._build(lr_schedule)

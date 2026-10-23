"""Continuous PPO with a bounded Gaussian mean, not a bounded random variable.

The Gaussian/gSDE likelihood and PPO objective remain SB3 implementations.
Only the mean parametrization changes. Samples may exceed [-1, 1] and retain
SB3's standard actuator clipping; there is no post-hoc probability modification.
"""
import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BoundedMeanActorCriticPolicy(ActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi):
        # Bounding the mean prevents an arbitrary distance between the Gaussian
        # centre and the actuator range. This does not bound learned variance.
        mean = torch.tanh(self.action_net(latent_pi))
        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean, self.log_std, latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean, self.log_std)
        raise ValueError('Bounded-mean policy requires continuous Gaussian actions')

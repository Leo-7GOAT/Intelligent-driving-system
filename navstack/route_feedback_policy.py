"""Learned direct-control residual on observable route errors; no runtime teacher."""
import torch
from torch import nn
from torch.nn import functional as F
from .road_candidate_policy import OrderedFlatRoadCandidatePolicy,OrderedFlatRoadExtractor
from .fixed_reward_units import FixedRewardUnitsNormalize

VERSION='learned_known_route_error_residual_v1'
MOTOR=(0,1,3,4,16,17,18,19,20,4701,4702,4703,4704)
EXTRA=4+len(MOTOR)


class RouteFeedbackNormalize(FixedRewardUnitsNormalize):
    def _normalize_obs(self,obs,obs_rms):
        normalized=super()._normalize_obs(obs,obs_rms)
        normalized[...,599:603]=obs[...,599:603]
        return normalized

    def _unnormalize_obs(self,obs,obs_rms):
        result=super()._unnormalize_obs(obs,obs_rms)
        result[...,599:603]=obs[...,599:603]
        return result


class RouteFeedbackExtractor(OrderedFlatRoadExtractor):
    def original_features(self,features):
        result=features.clone();result[:,599:603]=self.route_base_constants
        return result

    def forward_actor(self,features):
        base=super().forward_actor(self.original_features(features))
        return torch.cat((base,features[:,599:603],features[:,MOTOR]),dim=1)

    def forward_critic(self,features):
        return super().forward_critic(self.original_features(features))


class RouteFeedbackHead(nn.Linear):
    def base_action(self,latent):
        return F.linear(latent[:,:64].contiguous(),self.weight,self.bias)

    def residual(self,latent,base):
        return self.policy_net_route_feedback(torch.cat((latent,base),dim=1))+self.policy_net_route_cancel(base)

    def forward(self,latent):
        base=self.base_action(latent)
        return base+self.residual(latent,base)


def install(policy,constants=None):
    if type(policy.mlp_extractor) is not OrderedFlatRoadExtractor or type(policy.action_net) is not nn.Linear:
        raise ValueError('Require the unchanged ordered direct actor')
    policy.mlp_extractor.__class__=RouteFeedbackExtractor;policy.mlp_extractor.latent_dim_pi=64+EXTRA
    value=policy.action_net.weight.new_zeros(4) if constants is None else policy.action_net.weight.new_tensor(constants)
    policy.mlp_extractor.register_buffer('route_base_constants',value)
    policy.action_net.__class__=RouteFeedbackHead
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(22)
        feedback=nn.Sequential(nn.Linear(64+EXTRA+2,32),nn.Tanh(),nn.Linear(32,2)).to(policy.device)
        cancel=nn.Linear(2,2,bias=False).to(policy.device)
        nn.init.zeros_(feedback[-1].weight);nn.init.zeros_(feedback[-1].bias);nn.init.zeros_(cancel.weight)
    policy.action_net.add_module('policy_net_route_feedback',feedback)
    policy.action_net.add_module('policy_net_route_cancel',cancel)
    params=list(feedback.parameters())+list(cancel.parameters())
    policy.optimizer.add_param_group({'params':params})
    return params


class RouteFeedbackFlatPolicy(OrderedFlatRoadCandidatePolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule);install(self)


def promote(model,norm):
    if type(model.policy) is not OrderedFlatRoadCandidatePolicy or type(norm) is not FixedRewardUnitsNormalize:
        raise ValueError('Require the original ordered actor and frozen reward-unit normalizer')
    import numpy as np
    constants=norm.normalize_obs(np.zeros((1,4705),dtype=np.float32))[0,599:603]
    params=install(model.policy,constants)
    model.policy.__class__=RouteFeedbackFlatPolicy;model.policy_class=RouteFeedbackFlatPolicy
    norm.__class__=RouteFeedbackNormalize
    return dict(version=VERSION,new_learned_parameters=sum(p.numel() for p in params),
        initial_residual_zero=True,original_PPO_optimizer_groups_retained=True,
        old_RMS_values_retained=True,new_raw_observation_slots=[599,600,601,602]),params

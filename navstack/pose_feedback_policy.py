"""Opt-in learned direct steering residual on already visible route geometry.

No planner, controller, rule selection or runtime teacher is added to B1.
Ten zero-initialized coefficients preserve the original policy exactly until
explicitly fitted or updated by PPO. Acceleration and critic paths are intact.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .road_candidate_policy import OrderedFlatRoadCandidatePolicy,OrderedFlatRoadExtractor

PROFILE='learned_route_pose_residual_v1'
POSE_FEATURES=10


def pose_features(features):
    candidates=features[:,351:599].reshape(-1,31,8)
    geometry=candidates[:,(5,15,25),:3].flatten(1)
    return torch.cat((torch.ones_like(geometry[:,:1]),geometry),dim=1)


class PoseFeedbackExtractor(OrderedFlatRoadExtractor):
    def forward_actor(self,features):
        return torch.cat((super().forward_actor(features),pose_features(features)),dim=1)


class PoseFeedbackHead(nn.Linear):
    def forward(self,latent):
        # Retain the original contiguous GEMM layout on every BLAS backend.
        base=F.linear(latent[:,:64].contiguous(),self.weight,self.bias)
        residual=latent[:,64:]@self.policy_net_pose
        return base+torch.stack((torch.zeros_like(residual),residual),dim=1)


def install(policy):
    if type(policy.mlp_extractor) is not OrderedFlatRoadExtractor or type(policy.action_net) is not nn.Linear:
        raise ValueError('Require the unchanged ordered flat actor')
    policy.mlp_extractor.__class__=PoseFeedbackExtractor
    policy.mlp_extractor.latent_dim_pi=64+POSE_FEATURES
    policy.action_net.__class__=PoseFeedbackHead
    policy.action_net.register_parameter('policy_net_pose',nn.Parameter(policy.action_net.weight.new_zeros(POSE_FEATURES)))
    # Original group and parameter ordering,including trained Adam moments,
    # remain intact. Only the new zero-initialized head gets a separate group.
    policy.optimizer.add_param_group({'params':[policy.action_net.policy_net_pose]})


class PoseFeedbackFlatPolicy(OrderedFlatRoadCandidatePolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        install(self)


def promote(model):
    if type(model.policy) is not OrderedFlatRoadCandidatePolicy:
        raise ValueError('Require an explicitly versioned original ordered policy')
    install(model.policy)
    model.policy.__class__=PoseFeedbackFlatPolicy
    model.policy_class=PoseFeedbackFlatPolicy
    return {'version':PROFILE,'new_learned_parameters':POSE_FEATURES,
        'zero_initial_residual':True,'old_optimizer_groups_retained':True,
        'inputs':'Existing candidate5/15/25 x/15,y/15,sin(heading) plus learned intercept.',
        'scope':'Direct learned steering;no runtime MPC,planner,teacher or action selector.'}

"""Explicit learned acceleration residual with an unchanged direct steering path."""
import torch
from torch import nn
from torch.nn import functional as F
from .road_candidate_policy import OrderedFlatRoadCandidatePolicy,OrderedFlatRoadExtractor

INDICES=(0,16,17,20,3,4701,4702,4703,4704)
PROFILE='learned_terminal_acceleration_residual_v1'


class TerminalAccelerationExtractor(OrderedFlatRoadExtractor):
    def forward_actor(self,features):
        return torch.cat((super().forward_actor(features),features[:,INDICES]),dim=1)


class TerminalAccelerationHead(nn.Linear):
    def forward(self,latent):
        base=F.linear(latent[:,:64],self.weight,self.bias)
        residual=self.policy_net_terminal(latent[:,64:]).squeeze(1)
        return base+torch.stack((residual,torch.zeros_like(residual)),dim=1)


def install(policy):
    if (policy.observation_space.shape!=(4705,) or type(policy.mlp_extractor) is not OrderedFlatRoadExtractor
            or type(policy.action_net) is not nn.Linear):raise ValueError('Require the original4705input ordered direct policy')
    policy.mlp_extractor.__class__=TerminalAccelerationExtractor;policy.mlp_extractor.latent_dim_pi=73
    policy.action_net.__class__=TerminalAccelerationHead
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(22)
        head=nn.Sequential(nn.Linear(9,16),nn.Tanh(),nn.Linear(16,1)).to(policy.action_net.weight.device)
        nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    policy.action_net.add_module('policy_net_terminal',head)
    policy.optimizer.add_param_group({'params':list(head.parameters())})


class TerminalAccelerationFlatPolicy(OrderedFlatRoadCandidatePolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule);install(self)


def promote(model):
    if type(model.policy) is not OrderedFlatRoadCandidatePolicy:raise ValueError('Require explicit original direct-policy promotion')
    install(model.policy);model.policy.__class__=TerminalAccelerationFlatPolicy;model.policy_class=TerminalAccelerationFlatPolicy
    return {'version':PROFILE,'new_parameters':177,'initial_outputs_exact':True,
        'inputs':'Existing normalized speed,local goalx/goaly/distance,previous acceleration and4actuator states.',
        'runtime':'Learned NNacceleration only;no goal-distance rule,teacher or trajectory controller. Original steering function unchanged.'}

"""Opt-in B1 Adam rates: unchanged PPO loss/actor step,30x critic step."""
import math
import torch
from .independent_clip_ppo import actor_critic_parameters
from .road_candidate_policy import OrderedFlatRoadCandidatePolicy

PROFILE='actor1_critic30_v1'


class CriticRateAdam(torch.optim.Adam):
    def step(self,closure=None):
        original=[group['lr'] for group in self.param_groups]
        try:
            for group,rate in zip(self.param_groups,original):
                multiplier=group.get('learning_rate_multiplier',1.)
                effective=float(rate)*multiplier
                if not math.isfinite(effective) or not 0<effective<=1e-3:raise ValueError('Effective Adam rate outside registered bounds')
                group['lr']=effective;group['last_step_effective_lr']=effective
            return super().step(closure)
        finally:
            for group,rate in zip(self.param_groups,original):group['lr']=rate


def install_optimizer(policy,rate):
    if not math.isfinite(rate) or not 1e-6<=rate<=1e-3/30:raise ValueError('Fast critic requires a bounded small actor rate')
    if policy.optimizer.state:raise ValueError('Never discard trained Adam state during promotion')
    actor,critic=actor_critic_parameters(policy)
    policy.optimizer_class=CriticRateAdam
    policy.optimizer=CriticRateAdam([
        {'params':actor,'learning_rate_multiplier':1.,'role':'actor','last_step_effective_lr':None},
        {'params':critic,'learning_rate_multiplier':30.,'role':'critic','last_step_effective_lr':None}],
        lr=rate,**policy.optimizer_kwargs)


class OrderedFlatCriticRatePolicy(OrderedFlatRoadCandidatePolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        install_optimizer(self,float(lr_schedule(1)))


def promote_fresh_critic_rate(model):
    if (type(model.policy) is not OrderedFlatRoadCandidatePolicy or model.num_timesteps or model._n_updates
            or model.policy.optimizer.state):raise ValueError('Require untouched ordered B1actor-only initialization')
    before={name:value.clone() for name,value in model.policy.state_dict().items()}
    rate=float(model.lr_schedule(1))
    if not 1e-6<=rate<=1e-3/30:raise ValueError('Requested critic rate exceeds1e-3')
    model.policy.__class__=OrderedFlatCriticRatePolicy
    model.policy_class=OrderedFlatCriticRatePolicy
    install_optimizer(model.policy,rate)
    if not all(torch.equal(value,model.policy.state_dict()[name]) for name,value in before.items()):raise ValueError('Optimizer promotion changed network tensors')
    return {'version':PROFILE,'actor_learning_rate':rate,'critic_learning_rate':30*rate,
            'actor_multiplier':1,'critic_multiplier':30,'network_tensors_unchanged':True,
            'PPO_optimizer_state_initially_empty':True,'scope':'Same standard PPO objective,gradient clipping,rollouts and actor rate. Only disjoint critic Adamstep rate differs; no extra updates or data.'}

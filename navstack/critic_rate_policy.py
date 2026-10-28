"""Opt-in B1 Adam rates: unchanged PPO loss/actor step,30x critic step."""
import math
import torch
from .independent_clip_ppo import actor_critic_parameters
from .road_candidate_policy import OrderedFlatRoadCandidatePolicy

PROFILE='actor1_critic30_v1'


class CriticRateAdam(torch.optim.Adam):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Inheriting Adam.step avoids a nested PyTorch profile wrapper calling
        # every external optimizer hook twice. Rate hooks survive model reload
        # because the policy reconstructs this optimizer before loading state.
        self.register_step_pre_hook(self._apply_rates)
        self.register_step_post_hook(self._restore_rates)

    def _apply_rates(self,optimizer,args,kwargs):
        original=[group['lr'] for group in self.param_groups]
        effective=[float(rate)*group.get('learning_rate_multiplier',1.) for group,rate in zip(self.param_groups,original)]
        if any(not math.isfinite(rate) or not 0<rate<=1e-3 for rate in effective):
            raise ValueError('Effective Adam rate outside registered bounds')
        self._scheduled_rates=original
        for group,rate in zip(self.param_groups,effective):
            group['lr']=rate;group['last_step_effective_lr']=rate

    def _restore_rates(self,optimizer,args,kwargs):
        for group,rate in zip(self.param_groups,self._scheduled_rates):group['lr']=rate


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

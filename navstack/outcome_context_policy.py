"""Versioned action-aligned outcome feedback; no action mask or override.

The existing31context slots encode +1 for a reusable reference, -1 for the
failed planner proposals since the last accepted reference, and0otherwise. This structures information
already present in goal history/outcome fields; it adds no physical oracle.
Both new actor/critic terms start at zero and must learn from executed rewards.
"""
from copy import copy,deepcopy
import math
import numpy as np
import torch
from torch import nn
from .reference_context import reference_candidate_context,LearnedReferenceBias
from .reference_context_policy import (ReferenceContextObservationMixin,ReferenceContextPolicy,
    AblatedReferenceContextPolicy,ReferenceValueHead)

VERSION='action_aligned_outcome_context_v1'
FAILURE_GAIN_SCALE=1000.


def outcome_context(env,last_action=None,failed_actions=None):
    result=reference_candidate_context(env)
    action=getattr(env,'_outcome_last_action',None) if last_action is None else last_action
    outcome=getattr(env,'execution_outcome',np.zeros(6))
    failed=set(getattr(env,'_outcome_failed_actions',()) if failed_actions is None else failed_actions)
    if outcome[0]>.5:failed.clear()
    if action is not None and outcome[1]>.5:
        if not isinstance(action,(int,np.integer)) or not 0<=action<31:raise ValueError('Invalid previous policy action')
        if action:failed.add(int(action))
    for action in failed:
        if not isinstance(action,(int,np.integer)) or not 1<=action<31:raise ValueError('Invalid failed action history')
        if result[action]==0:result[action]=-1.
    return result


class OutcomeContextObservationMixin(ReferenceContextObservationMixin):
    def reset(self,*args,**kwargs):
        self._outcome_last_action=None;self._outcome_failed_actions=set()
        return super().reset(*args,**kwargs)

    def step(self,action):
        self._outcome_last_action=int(action)
        result=super().step(action)
        mode=result[-1]['mode']
        if mode=='reference':self._outcome_failed_actions.clear()
        elif mode=='plan_failed' and int(action):self._outcome_failed_actions.add(int(action))
        return result

    def _observation(self):
        result=super()._observation();result[-31:]=outcome_context(self)
        return result


class LearnedOutcomeBias(LearnedReferenceBias):
    def __init__(self,scale,failure_scale):
        super().__init__(scale)
        if not math.isfinite(failure_scale) or not 0<failure_scale<=1000:raise ValueError('Invalid failure learning-coordinate scale')
        self.failure_scale=float(failure_scale);self.failure_gain=nn.Parameter(torch.zeros(()))

    def forward(self,logits,context,*,ablated=False):
        if logits.shape!=context.shape or logits.shape[-1]!=31:raise ValueError('Outcome context shape mismatch')
        if not torch.isfinite(context).all() or torch.any((context!=0)&(context!=1)&(context!=-1)) or torch.any(context[...,0]!=0):
            raise ValueError('Measured signed context required;wait must stay zero')
        visible=torch.zeros_like(context) if ablated else context
        return logits+self.scale*self.gain*visible.clamp(min=0)+self.failure_scale*self.failure_gain*visible.clamp(max=0)


class OutcomeValueHead(ReferenceValueHead):
    def __init__(self,backbone):
        super().__init__(backbone);self.failure_weight=nn.Parameter(torch.zeros(31,1))

    def forward(self,latent):
        context=latent[:,-31:]
        return (self.backbone(latent[:,:-31])+context.clamp(min=0)@self.context_weight
            +context.clamp(max=0)@self.failure_weight)


class OutcomePolicyBuild:
    reference_context_version=VERSION

    def __init__(self,*args,failure_gain_scale=FAILURE_GAIN_SCALE,**kwargs):
        self.failure_gain_scale=float(failure_gain_scale)
        super().__init__(*args,**kwargs)

    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters();result['failure_gain_scale']=self.failure_gain_scale
        return result

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.action_net.reference_bias=LearnedOutcomeBias(self.reference_gain_scale,self.failure_gain_scale)
        self.value_net=OutcomeValueHead(self.value_net.backbone)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class OutcomeContextPolicy(OutcomePolicyBuild,ReferenceContextPolicy):pass


class AblatedOutcomeContextPolicy(OutcomePolicyBuild,AblatedReferenceContextPolicy):pass


def migrate_learner(model):
    """Preserve every original tensor/Adam group/RNG;two new states are zero.

    The separate failure coordinate uses scale1000to permit order-one logit
    learning at the existing3e-5learning rate. Its coefficient is free-signed;
    no fixed penalty or failure-triggered action replacement is installed.
    """
    if type(model.policy) not in (ReferenceContextPolicy,AblatedReferenceContextPolicy):
        raise ValueError('Require a versioned reusable-reference initializer')
    if len(model.policy.optimizer.param_groups)!=1:raise ValueError('Require one retained Adam group')
    new=copy(model);new.env=None;new._last_obs=None
    new.policy_class=AblatedOutcomeContextPolicy if model.policy.reference_ablated else OutcomeContextPolicy
    new.policy_kwargs={**model.policy_kwargs,'failure_gain_scale':FAILURE_GAIN_SCALE}
    with torch.random.fork_rng(devices=[]):
        new.policy=new.policy_class(new.observation_space,model.action_space,model.lr_schedule,**new.policy_kwargs)
    old=model.policy.state_dict();values=new.policy.state_dict()
    for name in values:
        if name in old:values[name]=old[name].clone()
        elif torch.count_nonzero(values[name]):raise ValueError('New outcome parameters must be zero')
    new.policy.load_state_dict(values);previous=dict(model.policy.named_parameters())
    for name,parameter in new.policy.named_parameters():
        if name in previous:
            new.policy.optimizer.state[parameter]=deepcopy(model.policy.optimizer.state.get(previous[name],{}))
        else:
            state=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.zeros_like(parameter))
            if new.policy.optimizer.param_groups[0].get('amsgrad'):state['max_exp_avg_sq']=torch.zeros_like(parameter)
            new.policy.optimizer.state[parameter]=state
    for key,value in model.policy.optimizer.param_groups[0].items():
        if key!='params':new.policy.optimizer.param_groups[0][key]=deepcopy(value)
    return new

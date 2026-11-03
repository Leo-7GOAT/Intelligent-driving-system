"""Versioned appended reference context, with exact zero-effect initialization."""
from copy import copy,deepcopy
import inspect
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from .reference_context import reference_candidate_context,LearnedReferenceBias
from .candidate_policy import CandidateVecNormalize
from .road_candidate_policy import EndpointClearRoadPolicy,SharedEndpointAblatedRoadPolicy

VERSION='appended_reference_context_v1'


class ReferenceContextObservationMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.observation_space.shape not in ((4701,),(4705,)) or self.config.method!='H' or not self.config.reference_reuse:
            raise ValueError('Reference context requires the explicit hierarchical reference-reuse interface')
        self.observation_space=spaces.Box(-np.inf,np.inf,(self.observation_space.shape[0]+31,),dtype=np.float32)

    def _observation(self):
        return np.r_[super()._observation(),reference_candidate_context(self)].astype(np.float32)


class ReferenceContextVecNormalize(CandidateVecNormalize):
    def _normalize_obs(self,obs,obs_rms):
        if obs.shape[-1] not in (4732,4736):raise ValueError('Reference context observation schema differs')
        result=super()._normalize_obs(obs,obs_rms)
        result[...,605:4701]=obs[...,605:4701];result[...,-31:]=obs[...,-31:]
        return result

    def _unnormalize_obs(self,obs,obs_rms):
        result=super()._unnormalize_obs(obs,obs_rms)
        result[...,605:4701]=obs[...,605:4701];result[...,-31:]=obs[...,-31:]
        return result


class ReferenceExtractor(nn.Module):
    def __init__(self,backbone,ablated):
        super().__init__();self.backbone=backbone;self.ablated=ablated
        self.latent_dim_pi=backbone.latent_dim_pi+31;self.latent_dim_vf=backbone.latent_dim_vf+31

    def context(self,features):
        return torch.zeros_like(features[:,-31:]) if self.ablated else features[:,-31:]

    def forward_actor(self,features):return torch.cat((self.backbone.forward_actor(features[:,:-31]),self.context(features)),dim=1)
    def forward_critic(self,features):return torch.cat((self.backbone.forward_critic(features[:,:-31]),self.context(features)),dim=1)
    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)


class ReferenceActionHead(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone;self.reference_bias=LearnedReferenceBias()

    def forward(self,latent):
        logits=self.backbone(latent[:,:-31]);result=self.reference_bias(logits,latent[:,-31:])
        # Preserve the original endpoint mask exactly,including for an
        # arbitrarily large learned coefficient.
        return torch.where(logits<-1e7,logits,result)


class ReferenceValueHead(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone
        self.context_weight=nn.Parameter(torch.zeros(31,1))

    def forward(self,latent):return self.backbone(latent[:,:-31])+latent[:,-31:]@self.context_weight


class ReferencePolicyBuild:
    reference_context_version=VERSION
    reference_ablated=False

    def _build(self,lr_schedule):
        full=self.observation_space
        if full.shape not in ((4732,),(4736,)):raise ValueError('Require31appended reference bits')
        self.observation_space=spaces.Box(-np.inf,np.inf,(full.shape[0]-31,),dtype=np.float32)
        try:super()._build(lr_schedule)
        finally:self.observation_space=full
        self.mlp_extractor=ReferenceExtractor(self.mlp_extractor,self.reference_ablated)
        self.action_net=ReferenceActionHead(self.action_net)
        self.value_net=ReferenceValueHead(self.value_net)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class ReferenceContextPolicy(ReferencePolicyBuild,EndpointClearRoadPolicy):pass


class AblatedReferenceContextPolicy(ReferencePolicyBuild,SharedEndpointAblatedRoadPolicy):
    reference_ablated=True


def original_name(name):
    for prefix in ('mlp_extractor.','action_net.','value_net.'):
        if name.startswith(prefix+'backbone.'):return prefix+name[len(prefix+'backbone.'):]
    return None


def migrate_learner(model,normalizer,*,ablated=False):
    """Preserve all original tensors/Adam;create two zero-initialized states.

    The returned learner must be saved/reloaded before native training so SB3
    builds a rollout buffer for the new explicit observation schema.
    """
    expected=SharedEndpointAblatedRoadPolicy if ablated else EndpointClearRoadPolicy
    if type(model.policy) is not expected or model.observation_space.shape not in ((4701,),(4705,)):
        raise ValueError('Migration requires the declared endpoint-policy initializer')
    if len(model.policy.optimizer.param_groups)!=1:raise ValueError('Require one retained Adam group')
    new=copy(model);new.env=None;new._last_obs=None
    new.observation_space=spaces.Box(-np.inf,np.inf,(model.observation_space.shape[0]+31,),dtype=np.float32)
    new.policy_class=AblatedReferenceContextPolicy if ablated else ReferenceContextPolicy
    with torch.random.fork_rng(devices=[]):
        new.policy=new.policy_class(new.observation_space,model.action_space,model.lr_schedule,**model.policy_kwargs)
    before=model.policy.state_dict();values=new.policy.state_dict()
    for name in values:
        old=original_name(name)
        if old is not None:
            if old not in before:raise ValueError('Unmapped original learned tensor:'+old)
            values[name]=before[old].clone()
        elif torch.count_nonzero(values[name]):raise ValueError('New learned parameters must start at zero')
    new.policy.load_state_dict(values)
    old_parameters=dict(model.policy.named_parameters())
    for name,parameter in new.policy.named_parameters():
        old=original_name(name)
        if old is not None and old_parameters[old] in model.policy.optimizer.state:
            new.policy.optimizer.state[parameter]=deepcopy(model.policy.optimizer.state[old_parameters[old]])
    for key,value in model.policy.optimizer.param_groups[0].items():
        if key!='params':new.policy.optimizer.param_groups[0][key]=deepcopy(value)
    # Detached VecNormalize intentionally lacks class_attributes;generic
    # deepcopy probes __deepcopy__ through its recursive env delegation.
    norm=ReferenceContextVecNormalize.__new__(ReferenceContextVecNormalize)
    norm.__setstate__(deepcopy({key:value for key,value in normalizer.__dict__.items()
                              if key not in ('venv','class_attributes')}))
    norm.class_attributes=dict(inspect.getmembers(ReferenceContextVecNormalize))
    norm.returns=np.zeros(norm.num_envs)
    norm.observation_space=new.observation_space
    norm.obs_rms.mean=np.r_[normalizer.obs_rms.mean,np.zeros(31)]
    norm.obs_rms.var=np.r_[normalizer.obs_rms.var,np.ones(31)]
    if norm.__dict__.get('old_obs') is not None:
        norm.old_obs=np.concatenate((norm.old_obs,np.zeros((*norm.old_obs.shape[:-1],31))),axis=-1)
    return new,norm

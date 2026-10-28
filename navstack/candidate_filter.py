"""Explicit candidate filters on observed endpoint and connector evidence."""
import numpy as np
import torch
from torch import nn


def local_witness_mask(candidates):
    """Keep wait and witnessed local connectors; fall back if none is known.

    The last (goal-continuation) witness is deliberately ignored. False means
    unknown, so an all-unknown state retains the complete planner action set.
    """
    values=np.asarray(candidates)
    if values.shape != (31,8) or not np.isfinite(values).all():
        raise ValueError('Require finite fixed-scale31x8candidate inputs')
    mask=(values[:,5]>.5)&(values[:,6]>.5)
    mask[0]=True
    if not mask[1:].any():mask[:]=True
    return mask


def endpoint_clear_mask(candidates):
    values=np.asarray(candidates)
    if values.shape!=(31,8) or not np.isfinite(values).all():raise ValueError('Require finite fixed-scale31x8candidate inputs')
    # Wait's clear bit is the availability sentinel. Ablated/unknown feedback
    # keeps all actions; known blocked endpoints can never admit a full path.
    if values[0,5]<=.5:return np.ones(31,dtype=bool)
    mask=values[:,5]>.5;mask[0]=True
    return mask


class LocalWitnessExtractor(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone
        self.latent_dim_pi=backbone.latent_dim_pi+31
        self.latent_dim_vf=backbone.latent_dim_vf

    def forward_actor(self,features):
        candidates=features[:,351:599].reshape(-1,31,8)
        mask=(candidates[:,:,5]>.5)&(candidates[:,:,6]>.5)
        mask=mask.clone();mask[:,0]=True
        mask=torch.where(mask[:,1:].any(dim=1,keepdim=True),mask,torch.ones_like(mask))
        return torch.cat((self.backbone.forward_actor(features),mask.to(features.dtype)),dim=1)

    def forward_critic(self,features):return self.backbone.forward_critic(features)

    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)


class LocalWitnessHead(nn.Module):
    def __init__(self,head):
        super().__init__();self.head=head

    def forward(self,latent):
        return self.head(latent[:,:-31]).masked_fill(latent[:,-31:]<.5,-1e8)


class EndpointClearExtractor(LocalWitnessExtractor):
    def forward_actor_unmasked(self,features):return self.backbone.forward_actor(features)

    def forward_actor(self,features):
        candidates=features[:,351:599].reshape(-1,31,8)
        mask=(candidates[:,:,5]>.5).clone();mask[:,0]=True
        mask=torch.where(candidates[:,0,5:6]>.5,mask,torch.ones_like(mask))
        return torch.cat((self.forward_actor_unmasked(features),mask.to(features.dtype)),dim=1)


class SharedEndpointFeedbackAblationExtractor(EndpointClearExtractor):
    """Keep the common physical mask; hide feedback from actor AND critic."""
    @staticmethod
    def learned_inputs(features):
        if features.ndim!=2 or features.shape[1] not in (4701,4705):raise ValueError('Require road candidate observations')
        result=features.clone()
        result[:,351:599].reshape(-1,31,8)[:,:,5:]=0
        result[:,599:605]=0
        return result

    def forward_actor_unmasked(self,features):
        return self.backbone.forward_actor(self.learned_inputs(features))

    def forward_critic(self,features):
        return self.backbone.forward_critic(self.learned_inputs(features))

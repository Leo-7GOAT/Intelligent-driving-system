"""Explicit reference availability for a future versioned learned policy.

This module does not select, mask or replace an action. Existing environments
and published policies do not call it; integration requires a new experiment.
"""
import numpy as np
import torch
from torch import nn


def reference_candidate_context(env,catalog=None):
    """One observed bit per action: same still-reusable timed reference."""
    result=np.zeros(31,dtype=np.float32)
    selection=getattr(env,'_reference_selection',None)
    if selection is None or not env.reusable_reference(*selection):return result
    catalog=[env.decode_action(i) for i in range(31)] if catalog is None else catalog
    if len(catalog)!=31:raise ValueError('Require the registered31-action catalog')
    for index in range(1,31):
        result[index]=float(catalog[index]==selection)
    return result


class LearnedReferenceBias(nn.Module):
    """Zero-initialized trainable logit term; no forced commitment or fallback."""
    def __init__(self):
        super().__init__()
        self.gain=nn.Parameter(torch.zeros(()))

    def forward(self,logits,context,*,ablated=False):
        if logits.shape!=context.shape or logits.shape[-1]!=31:
            raise ValueError('Reference context must match the31categorical logits')
        if not torch.isfinite(context).all() or torch.any((context!=0)&(context!=1)) or torch.any(context[...,0]!=0):
            raise ValueError('Require binary measured context;wait has no reference')
        visible=torch.zeros_like(context) if ablated else context
        return logits+self.gain*visible

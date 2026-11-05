"""State-conditioned use of existing signed execution feedback.

The v1 observation codec and physical pipeline are unchanged. New actor and
critic residuals have exactly zero output at migration. This module supplies
no action mask, held-option override, planning oracle or extra supervision.
Native admission requires a separately versioned experiment registration.
"""
from copy import copy, deepcopy
import math

import torch
from torch import nn

from .outcome_context_policy import OutcomeContextPolicy, OutcomeValueHead
from .contextual_feedback_registration import (
    VERSION, HIDDEN, ACTOR_SCALE, NEW_PARAMETER_STATES, TOTAL_PARAMETER_STATES,
    TRAINING_REGISTRATION, registered,
)


def validate_native_registration(protocol, model):
    tag = getattr(model.policy, 'contextual_feedback_version', None)
    if tag is None and protocol.get('contextual_feedback_registration') is None:
        return
    if (not registered(protocol) or type(model.policy) is not ContextualOutcomePolicy
            or tag != VERSION or model.policy.contextual_scale != ACTOR_SCALE
            or model.policy.contextual_ablated != protocol['contextual_feedback_registration']['ablated']
            or len(list(model.policy.named_parameters())) != TOTAL_PARAMETER_STATES):
        raise ValueError('Saved conditional policy differs from explicit native registration')


def _network(inputs, outputs):
    net = nn.Sequential(nn.Linear(inputs, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, outputs))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class ConditionalOutcomeBias(nn.Module):
    def __init__(self, inputs, *, scale=ACTOR_SCALE, ablated=False):
        super().__init__()
        if not math.isfinite(scale) or not 0 < scale <= 100:
            raise ValueError('Conditional actor scale must be finite in (0,100]')
        if type(ablated) is not bool:
            raise ValueError('Explicit conditional-feedback ablation required')
        self.scale, self.ablated = float(scale), ablated
        self.net = _network(inputs, 2)

    def forward(self, latent):
        context = latent[:, -31:]
        if (not torch.isfinite(context).all()
                or torch.any((context != 0) & (context != 1) & (context != -1))
                or torch.any(context[:, 0] != 0)):
            raise ValueError('Require the unchanged signed31 execution context')
        visible = torch.zeros_like(context) if self.ablated else context
        gains = self.scale * self.net(torch.cat((latent[:, :-31], visible), dim=1))
        return visible.clamp(min=0) * gains[:, :1] + visible.clamp(max=0) * gains[:, 1:]


class ContextualActionHead(nn.Module):
    def __init__(self, original, inputs, *, scale, ablated):
        super().__init__()
        # Original parameter names and objects survive wrapping; migration can
        # restore every original Adam state by name without index remapping.
        self.backbone, self.reference_bias = original.backbone, original.reference_bias
        self.contextual_bias = ConditionalOutcomeBias(inputs, scale=scale, ablated=ablated)

    def forward(self, latent):
        logits = self.backbone(latent[:, :-31])
        result = self.reference_bias(logits, latent[:, -31:]) + self.contextual_bias(latent)
        return torch.where(logits < -1e7, logits, result)


class ContextualValueHead(OutcomeValueHead):
    def __init__(self, original, inputs, *, ablated):
        super().__init__(original.backbone, original.failure_ablated)
        self.context_weight, self.failure_weight = original.context_weight, original.failure_weight
        self.contextual_ablated = ablated
        self.contextual_net = _network(inputs, 1)

    def forward(self, latent):
        context = latent[:, -31:]
        visible = torch.zeros_like(context) if self.contextual_ablated else context
        active = torch.any(visible != 0, dim=1, keepdim=True).to(latent.dtype)
        residual = self.contextual_net(torch.cat((latent[:, :-31], visible), dim=1)) * active
        return super().forward(latent) + residual


class ContextualOutcomePolicy(OutcomeContextPolicy):
    contextual_feedback_version = VERSION

    def __init__(self, *args, contextual_scale=ACTOR_SCALE, contextual_ablated=False, **kwargs):
        if type(contextual_ablated) is not bool:
            raise ValueError('Explicit conditional-feedback ablation required')
        self.contextual_scale, self.contextual_ablated = float(contextual_scale), contextual_ablated
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result.update(contextual_scale=self.contextual_scale, contextual_ablated=self.contextual_ablated)
        return result

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Extra initialization cannot perturb the original seeded construction.
        with torch.random.fork_rng(devices=[]):
            self.action_net = ContextualActionHead(self.action_net, self.mlp_extractor.latent_dim_pi,
                scale=self.contextual_scale, ablated=self.contextual_ablated).to(self.device)
            self.value_net = ContextualValueHead(self.value_net, self.mlp_extractor.latent_dim_vf,
                ablated=self.contextual_ablated).to(self.device)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def migrate_learner(model, *, ablated=False):
    """Exact old tensors/Adam/RNG; only eight new parameter states start at zero.

    New hidden weights are ordinary random features; their output layers are
    zero, so both branches have exactly zero initial effect for every context.
    Ablation removes only these new conditional branches, retaining all v1
    feedback and all physical execution capabilities.
    """
    if type(model.policy) is not OutcomeContextPolicy:
        raise ValueError('Require an unmodified v1 outcome-context learner')
    if len(model.policy.optimizer.param_groups) != 1:
        raise ValueError('Require one original Adam group')
    if type(ablated) is not bool:
        raise ValueError('Explicit conditional-feedback ablation required')
    new = copy(model)
    new.env, new._last_obs = None, None
    new.policy_class = ContextualOutcomePolicy
    new.policy_kwargs = {**model.policy_kwargs, 'contextual_scale': ACTOR_SCALE,
                         'contextual_ablated': ablated}
    with torch.random.fork_rng(devices=[]):
        new.policy = ContextualOutcomePolicy(new.observation_space, model.action_space,
                                            model.lr_schedule, **new.policy_kwargs)
    old, values = model.policy.state_dict(), new.policy.state_dict()
    if not set(old).issubset(values):
        raise ValueError('Migration lost an original state tensor')
    for name in old:
        values[name] = old[name].clone()
    new.policy.load_state_dict(values)
    previous = dict(model.policy.named_parameters())
    added = []
    for name, parameter in new.policy.named_parameters():
        if name in previous:
            new.policy.optimizer.state[parameter] = deepcopy(model.policy.optimizer.state.get(previous[name], {}))
        else:
            if not name.startswith(('action_net.contextual_bias.net.', 'value_net.contextual_net.')):
                raise ValueError('Unexpected new policy parameter: ' + name)
            added.append(name)
            state = dict(step=torch.tensor(0.), exp_avg=torch.zeros_like(parameter),
                         exp_avg_sq=torch.zeros_like(parameter))
            if new.policy.optimizer.param_groups[0].get('amsgrad'):
                state['max_exp_avg_sq'] = torch.zeros_like(parameter)
            new.policy.optimizer.state[parameter] = state
    if len(added) != NEW_PARAMETER_STATES:
        raise ValueError('Unexpected conditional parameter-state count')
    for key, value in model.policy.optimizer.param_groups[0].items():
        if key != 'params':
            new.policy.optimizer.param_groups[0][key] = deepcopy(value)
    return new

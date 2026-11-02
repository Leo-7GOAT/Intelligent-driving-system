from copy import deepcopy
from types import SimpleNamespace
import torch
import pytest
from scripts.queue_route_feedback_ppo import change_rate
from scripts.fit_broader_measured_actor import exact_state


def test_rate_contrast_keeps_weights_exploration_and_existing_adam_history():
    policy=torch.nn.Linear(4,2);policy.log_std=torch.nn.Parameter(torch.zeros(2))
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-6)
    (policy(torch.ones(1,4)).square().sum()+policy.log_std.square().sum()).backward();optimizer.step();optimizer.zero_grad()
    policy.optimizer=optimizer
    model=SimpleNamespace(policy=policy,learning_rate=3e-6)
    before=deepcopy(optimizer.state_dict());weights=deepcopy(policy.state_dict())
    proof=change_rate(model,1e-4)
    assert proof['old_group_rates']==[3e-6] and proof['new_group_rates']==[1e-4]
    assert model.learning_rate==model.lr_schedule(.4)==1e-4
    assert exact_state(before['state'],optimizer.state_dict()['state'])
    assert all(torch.equal(v,policy.state_dict()[k]) for k,v in weights.items())
    with pytest.raises(ValueError):change_rate(model,.01)

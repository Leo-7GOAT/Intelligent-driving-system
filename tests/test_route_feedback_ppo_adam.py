from copy import deepcopy
import pytest
import torch
from scripts.audit_route_feedback_ppo import verify_adam


def trained_adam():
    old=torch.nn.Parameter(torch.ones(2));new=[torch.nn.Parameter(torch.ones(2)) for _ in range(5)]
    optimizer=torch.optim.Adam([old],lr=.0003)
    old.sum().backward();optimizer.step();optimizer.zero_grad()
    optimizer.add_param_group({'params':new});before=deepcopy(optimizer.state_dict())
    for _ in range(4):
        (old.square().sum()+sum(p.square().sum() for p in new)).backward()
        optimizer.step();optimizer.zero_grad()
    return before,deepcopy(optimizer.state_dict())


def test_actual_adam_retains_old_history_and_initializes_exactly_the_five_new_policy_parameters():
    before,after=trained_adam()
    assert verify_adam(before,after,4)==dict(original_states=1,new_route_states=5,step_increment_each=4)


def test_empty_new_group_adopts_the_existing_ppo_schedule_without_rewriting_old_adam():
    before,after=trained_adam();before['param_groups'][0]['lr']=3e-6
    for group in after['param_groups']:group['lr']=3e-6
    assert verify_adam(before,after,4)['new_route_states']==5


@pytest.mark.parametrize('fault',['missing','wrong_step','wrong_group','unregistered_empty'])
def test_dropped_or_undertrained_parameters_cannot_count_as_complete_ppo(fault):
    before,after=trained_adam()
    if fault=='missing':after['state'].pop(3)
    elif fault=='wrong_step':after['state'][2]['step']-=1
    elif fault=='wrong_group':after['param_groups'][-1]['lr']=.03
    else:before['state'].clear()
    with pytest.raises(ValueError):verify_adam(before,after,4)

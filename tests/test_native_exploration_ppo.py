from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
import torch

from scripts.prepare_native_exploration_ppo import set_exploration,equal


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.log_std=torch.nn.Parameter(torch.full((2,),-4.))
        self.mean=torch.nn.Linear(3,2);self.value=torch.nn.Linear(3,1)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=3e-6)


def trained():
    policy=Policy();sum(p.square().sum() for p in policy.parameters()).backward();policy.optimizer.step()
    return SimpleNamespace(policy=policy,num_timesteps=20480,_n_updates=160)


def test_std_intervention_preserves_mean_critic_optimizer_and_counters():
    model=trained();before={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    optimizer=deepcopy(model.policy.optimizer.state_dict());x=torch.tensor([[.2,.4,.9]])
    mean=model.policy.mean(x).detach().clone();value=model.policy.value(x).detach().clone()
    result=set_exploration(model)
    np.testing.assert_allclose(result['new_std'],[.1,.1],rtol=1e-6)
    assert torch.equal(mean,model.policy.mean(x)) and torch.equal(value,model.policy.value(x))
    assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items() if k!='log_std')
    assert equal(optimizer,model.policy.optimizer.state_dict())
    assert (model.num_timesteps,model._n_updates)==(20480,160)
    assert result['optimizer_updates']==0 and result['changed_scalar_parameters']==2


@pytest.mark.parametrize('std',[.01,.2,0.,float('nan'),True])
def test_only_locked_exploration_amplitude_is_allowed(std):
    with pytest.raises(ValueError):set_exploration(trained(),std)


def test_untrained_or_opaque_action_policy_is_rejected():
    model=trained();model.num_timesteps=0
    with pytest.raises(ValueError):set_exploration(model)
    model=trained();model.policy.log_std=torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError):set_exploration(model)

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from navstack.bounded_policy import BoundedMeanActorCriticPolicy
from navstack.config import EnvConfig
from navstack.env import NavigationEnv


@pytest.mark.parametrize('use_sde',[False,True])
def test_bounded_mean_likelihood_update_and_reload(tmp_path,use_sde):
    torch.set_num_threads(1)
    env=NavigationEnv(EnvConfig(method='B1',scenario='empty',flat_hold=10))
    model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,n_epochs=2,seed=11,use_sde=use_sde,device='cpu')
    model.learn(16)
    with torch.no_grad():model.policy.action_net.bias[0]=4.
    model.save(tmp_path/'original')
    bounded=PPO.load(tmp_path/'original',env=env,custom_objects={'policy_class':BoundedMeanActorCriticPolicy})
    for key,value in model.policy.state_dict().items():
        assert torch.equal(value,bounded.policy.state_dict()[key])
    for key,state in model.policy.optimizer.state_dict()['state'].items():
        for name,value in state.items():
            other=bounded.policy.optimizer.state_dict()['state'][key][name]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other
    obs,_=env.reset(seed=11);tensor=torch.tensor(obs[None])
    with torch.no_grad():
        raw=model.policy.get_distribution(tensor).distribution.mean.clone()
        dist=bounded.policy.get_distribution(tensor)
        mean=dist.distribution.mean.clone()
        torch.testing.assert_close(mean,torch.tanh(raw))
        assert mean.abs().max()<=1 and raw[0,0]>1
        # Likelihood at out-of-range Gaussian samples still uses the actual
        # Gaussian, not a fake mask or a clipped-sample likelihood.
        actions=torch.tensor([[1.4,-1.2]])
        expected=dist.log_prob(actions)
        _,actual,_=bounded.policy.evaluate_actions(tensor,actions)
        torch.testing.assert_close(expected,actual)
    before=bounded.policy.action_net.weight.detach().clone();updates=bounded._n_updates
    bounded.learn(16)
    assert bounded._n_updates>=updates+4
    assert not torch.equal(before,bounded.policy.action_net.weight)
    action,_=bounded.predict(obs,deterministic=True)
    bounded.save(tmp_path/'bounded')
    restored=PPO.load(tmp_path/'bounded')
    assert isinstance(restored.policy,BoundedMeanActorCriticPolicy)
    np.testing.assert_array_equal(restored.predict(obs,deterministic=True)[0],action)
    env.close()

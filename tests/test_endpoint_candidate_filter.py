import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from navstack.candidate_filter import endpoint_clear_mask
from navstack.road_candidate_policy import EndpointClearRoadPolicy,UnsharedRoadCandidatePolicy
from navstack.config import EnvConfig
from navstack.experiments import signature
from test_local_candidate_filter import Probe,inputs


def endpoint_inputs():
    obs=inputs();c=obs[351:599].reshape(31,8)
    c[0,5]=1.;c[26,5]=1.  # Known free endpoint, unknown local/global connection.
    return obs


class EndpointProbe(Probe):
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed,options=options)
        return endpoint_inputs(),{}
    def step(self,action):
        assert action in (0,3,9,26)
        self.count+=1
        return endpoint_inputs(),float(action)/30.,self.count==8,False,{}


def test_endpoint_filter_preserves_unknown_connections_exact_likelihood_and_seeded_network(tmp_path):
    torch.set_num_threads(1)
    baseline=PPO(UnsharedRoadCandidatePolicy,EndpointProbe(),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    rng=torch.get_rng_state().clone()
    model=PPO(EndpointClearRoadPolicy,EndpointProbe(),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in model.policy.state_dict().items():
        old=name.replace('mlp_extractor.backbone.','mlp_extractor.').replace('action_net.head.','action_net.')
        assert torch.equal(value,baseline.policy.state_dict()[old])
    obs=torch.as_tensor(np.stack([endpoint_inputs()]*32))
    mask=endpoint_clear_mask(obs[0,351:599].numpy().reshape(31,8))
    assert np.flatnonzero(mask).tolist()==[0,3,9,26]
    probabilities=model.policy.get_distribution(obs).distribution.probs.detach().numpy()
    assert np.flatnonzero(probabilities[0]>0).tolist()==np.flatnonzero(mask).tolist()
    with torch.no_grad():
        actions,_,before=model.policy(obs);_,after,_=model.policy.evaluate_actions(obs,actions)
    torch.testing.assert_close(before,after)
    unknown=obs.clone();unknown[:,351:599]=0
    assert (model.policy.get_distribution(unknown).distribution.probs>0).all()
    blocked=unknown.clone();blocked[:,351+5]=1
    assert (model.policy.get_distribution(blocked).distribution.probs[:,0]==1).all()
    assert endpoint_clear_mask(blocked[0,351:599].numpy().reshape(31,8)).sum()==1
    model.learn(16);assert model._n_updates==4
    model.save(tmp_path/'model.zip');loaded=PPO.load(tmp_path/'model.zip',device='cpu')
    assert isinstance(loaded.policy,EndpointClearRoadPolicy)
    np.testing.assert_array_equal(model.predict(obs.numpy(),deterministic=True)[0],loaded.predict(obs.numpy(),deterministic=True)[0])


def test_endpoint_rule_shares_filter_and_disabled_config_remains_portable():
    from test_route_catalog_environment import RouteFixture,config
    from navstack.env import rule_action
    settings=config();settings.endpoint_candidate_filter=True;settings.__post_init__()
    env=RouteFixture(settings);obs,_=env.reset(seed=11)
    mask=endpoint_clear_mask(obs[351:599].reshape(31,8))
    np.testing.assert_array_equal(env._local_candidate_mask,mask)
    assert mask[rule_action(env)]
    assert 'endpoint_candidate_filter' not in EnvConfig().to_dict()
    assert 'endpoint_candidate_filter' not in signature(EnvConfig())
    with pytest.raises(ValueError):EnvConfig(endpoint_candidate_filter=True)

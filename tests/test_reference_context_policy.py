"""Synthetic learner migration/roundtrip;not CARLA performance evidence."""
import pickle
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.config import EnvConfig
from navstack.road_candidate_policy import RoadCandidateVecNormalize,EndpointClearRoadPolicy,SharedEndpointAblatedRoadPolicy
from navstack.reference_context_policy import (migrate_learner,original_name,ReferenceContextObservationMixin,
    ReferenceContextVecNormalize)
from navstack.independent_clip_ppo import actor_critic_parameters
from test_route_catalog_environment import RouteFixture,config


class ContextFixture(ReferenceContextObservationMixin,RouteFixture):pass


@pytest.mark.parametrize('ablated',[False,True])
def test_migration_retains_original_probabilities_values_adam_rms_and_survives_training_reload(tmp_path,ablated):
    torch.set_num_threads(1)
    cfg=EnvConfig.from_dict({**config(feedback='continuation').to_dict(),'route_goal_candidate':True,'reference_reuse':True})
    raw=DummyVecEnv([lambda:RouteFixture(cfg)])
    norm=RoadCandidateVecNormalize(raw,norm_reward=False)
    model=PPO(SharedEndpointAblatedRoadPolicy if ablated else EndpointClearRoadPolicy,norm,
              n_steps=2,batch_size=2,n_epochs=1,seed=11,device='cpu')
    model.learn(2)
    norm.training=False
    obs=raw.reset();normalized=norm.normalize_obs(obs)
    detached=pickle.loads(pickle.dumps(norm))
    rng=torch.get_rng_state().clone()
    new,newnorm=migrate_learner(model,detached,ablated=ablated)
    assert torch.equal(rng,torch.get_rng_state())
    context=np.zeros((1,31),dtype=np.float32);context[:,26]=1
    extended=np.concatenate((obs,context),axis=1)
    n=newnorm.normalize_obs(extended)
    np.testing.assert_array_equal(n[:,:-31],normalized)
    np.testing.assert_array_equal(n[:,-31:],context)
    with torch.no_grad():
        oldtensor=torch.as_tensor(normalized);newtensor=torch.as_tensor(n)
        assert torch.equal(model.policy.get_distribution(oldtensor).distribution.probs,
                           new.policy.get_distribution(newtensor).distribution.probs)
        assert torch.equal(model.policy.predict_values(oldtensor),new.policy.predict_values(newtensor))
    actor_critic_parameters(new.policy)
    old_parameters=dict(model.policy.named_parameters())
    for name,parameter in new.policy.named_parameters():
        old=original_name(name)
        if old is not None:
            for key,value in model.policy.optimizer.state[old_parameters[old]].items():
                actual=new.policy.optimizer.state[parameter][key]
                assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual
        else:assert parameter not in new.policy.optimizer.state
    np.testing.assert_array_equal(newnorm.obs_rms.mean[:-31],norm.obs_rms.mean)
    assert newnorm.obs_rms.count==norm.obs_rms.count
    new.save(tmp_path/'model.zip');newnorm.save(tmp_path/'normalization.pkl')
    wrapped=DummyVecEnv([lambda:ContextFixture(cfg)])
    restored=ReferenceContextVecNormalize.load(tmp_path/'normalization.pkl',wrapped)
    restored.training=False
    loaded=PPO.load(tmp_path/'model.zip',env=restored,device='cpu')
    action=loaded.predict(n,deterministic=True)[0]
    np.testing.assert_array_equal(new.predict(n,deterministic=True)[0],action)
    before=loaded._n_updates;loaded.learn(2,reset_num_timesteps=False)
    assert loaded._n_updates>before
    loaded.save(tmp_path/'learned.zip');roundtrip=PPO.load(tmp_path/'learned.zip',device='cpu')
    assert all(torch.equal(value,roundtrip.policy.state_dict()[name]) for name,value in loaded.policy.state_dict().items())
    restored.close();norm.close()

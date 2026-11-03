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
from scripts.prepare_native_reference_context import set_training_entropy


class ContextFixture(ReferenceContextObservationMixin,RouteFixture):pass


@pytest.mark.parametrize('ablated',[False,True])
@pytest.mark.parametrize('gain_scale',[1.,10.,100.])
@pytest.mark.parametrize('entropy_coef',[None,.02])
def test_migration_retains_original_probabilities_values_adam_rms_and_survives_training_reload(tmp_path,ablated,gain_scale,entropy_coef):
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
    new,newnorm=migrate_learner(model,detached,ablated=ablated,gain_scale=gain_scale)
    set_training_entropy(new,entropy_coef)
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
    assert loaded.ent_coef==(model.ent_coef if entropy_coef is None else entropy_coef)
    assert loaded.policy.reference_gain_scale==gain_scale
    assert loaded.policy.action_net.reference_bias.scale==gain_scale
    action=loaded.predict(n,deterministic=True)[0]
    np.testing.assert_array_equal(new.predict(n,deterministic=True)[0],action)
    before=loaded._n_updates;loaded.learn(2,reset_num_timesteps=False)
    assert loaded._n_updates>before
    loaded.save(tmp_path/'learned.zip');roundtrip=PPO.load(tmp_path/'learned.zip',device='cpu')
    assert roundtrip.ent_coef==loaded.ent_coef
    assert all(torch.equal(value,roundtrip.policy.state_dict()[name]) for name,value in loaded.policy.state_dict().items())
    if gain_scale==1.:
        # The first frozen native implementation predates the explicit scale
        # attribute/kwarg. Its next checkpoint must remain writable and loadable.
        from navstack.experiments import write_json,signature
        from scripts.native_road_learning import save_training_checkpoint
        del loaded.policy.reference_gain_scale
        loaded.policy_kwargs.pop('reference_gain_scale',None)
        wrapped.envs[0].case_index=3
        write_json(tmp_path/'config.json',{'env':cfg.to_dict(),'signature':signature(cfg)})
        save_training_checkpoint(tmp_path,320,loaded,restored,wrapped,{'physics_steps':320})
        checkpoint=tmp_path/'checkpoint_000000320'
        import json
        assert json.loads((checkpoint/'metadata.json').read_text())['reference_gain_scale']==1.
        compatible=PPO.load(checkpoint/'model.zip',device='cpu')
        assert compatible.policy.reference_gain_scale==1.
        assert all(torch.equal(value,compatible.policy.state_dict()[name]) for name,value in loaded.policy.state_dict().items())
    restored.close();norm.close()

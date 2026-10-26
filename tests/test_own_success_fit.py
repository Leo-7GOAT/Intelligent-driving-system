import numpy as np
import torch
import pytest
from stable_baselines3 import PPO
from test_discrete_flat_policy import Probe
from navstack.discrete_flat_policy import DiscreteFlatRoadPolicy
from scripts.fit_own_success_policy import fit_actor


def test_auxiliary_success_replay_changes_actor_without_ppo_or_critic_updates():
    torch.set_num_threads(1)
    model=PPO(DiscreteFlatRoadPolicy,Probe(True),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    model.learn(16)  # Populate both actor and critic Adam state before the check.
    actor=model.policy.action_net.weight.detach().clone()
    obs=np.zeros((8,4705),dtype=np.float32)
    actions=np.tile([1,3],(8,1));groups=np.repeat([0,2],4)
    updates=model._n_updates
    result=fit_actor(model,obs,actions,groups,10)
    assert result['actor_optimizer_steps']==10 and result['ppo_updates_added']==0
    assert result['critic_unchanged'] and result['critic_adam_unchanged']
    assert model._n_updates==updates
    assert not torch.equal(actor,model.policy.action_net.weight)
    assert result['last_loss']<result['first_loss']


def test_frozen_statistics_still_allow_actual_ppo_updates():
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    from scripts.native_road_learning import configure_continuation_normalization
    env=VecNormalize(DummyVecEnv([lambda:Probe(True)]),norm_reward=False)
    model=PPO(DiscreteFlatRoadPolicy,env,seed=11,n_steps=8,batch_size=8,n_epochs=2)
    model.learn(8)
    before=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count)
    actor=model.policy.action_net.weight.detach().clone();updates=model._n_updates
    with pytest.raises(ValueError):configure_continuation_normalization(env,{'freeze_observation_statistics':True},None)
    configure_continuation_normalization(env,{'freeze_observation_statistics':True},{'checkpoint':'saved'})
    model.learn(8,reset_num_timesteps=False)
    np.testing.assert_array_equal(before[0],env.obs_rms.mean);np.testing.assert_array_equal(before[1],env.obs_rms.var)
    assert before[2]==env.obs_rms.count and model._n_updates>updates
    assert not torch.equal(actor,model.policy.action_net.weight)
    env.close()

import numpy as np
import pytest
from gymnasium import Env,spaces
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from navstack.road_candidate_policy import RoadCandidateVecNormalize
from navstack.fixed_reward_units import initialize_fixed_reward_units


class PhysicalReward(Env):
    observation_space=spaces.Box(-1,1,(4705,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(2,),dtype=np.float32)
    def reset(self,**kwargs):return np.zeros(4705,dtype=np.float32),{}
    def step(self,action):return np.zeros(4705,dtype=np.float32),-90.,False,False,{'physical_reward':-90.}


def test_frozen_observations_and_raw_physical_reward_survive_fixed_units_and_reload(tmp_path):
    raw=DummyVecEnv([PhysicalReward]);norm=RoadCandidateVecNormalize(raw,norm_reward=False)
    norm.obs_rms.update(np.zeros((2,4705),dtype=np.float32));norm.training=False
    before=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
    initialize_fixed_reward_units(norm,.05);x=norm.reset()
    obs,reward,done,info=norm.step(np.zeros((1,2),dtype=np.float32))
    np.testing.assert_array_equal(reward,[-4.5]);np.testing.assert_array_equal(norm.get_original_reward(),[-90.])
    assert info[0]['physical_reward']==-90.
    np.testing.assert_array_equal(norm.unnormalize_reward(reward),[-90.])
    np.testing.assert_array_equal(before[0],norm.obs_rms.mean);np.testing.assert_array_equal(before[1],norm.obs_rms.var)
    assert before[2]==norm.obs_rms.count
    norm.save(tmp_path/'norm.pkl');loaded=VecNormalize.load(tmp_path/'norm.pkl',DummyVecEnv([PhysicalReward]))
    assert not loaded.norm_reward and not loaded.training
    np.testing.assert_array_equal(loaded.reset(),x)
    np.testing.assert_array_equal(loaded.step(np.zeros((1,2),dtype=np.float32))[1],[-4.5])


@pytest.mark.parametrize('fault',['adaptive','unregistered_scale'])
def test_no_silent_adaptive_reward_change_or_unregistered_scale(fault):
    norm=RoadCandidateVecNormalize(DummyVecEnv([PhysicalReward]),norm_reward=fault=='adaptive')
    with pytest.raises(ValueError):initialize_fixed_reward_units(norm,.1 if fault=='unregistered_scale' else .05)


def test_real_sb3_rollout_receives_scaled_rewards_while_raw_environment_info_remains_physical():
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    norm=RoadCandidateVecNormalize(DummyVecEnv([PhysicalReward]),norm_reward=False)
    norm.obs_rms.update(np.zeros((2,4705),dtype=np.float32));norm.training=False
    initialize_fixed_reward_units(norm,.05)
    model=PPO('MlpPolicy',norm,n_steps=8,batch_size=8,n_epochs=1,seed=11,device='cpu')
    model.learn(8)
    np.testing.assert_array_equal(model.rollout_buffer.rewards,np.full((8,1),-4.5,dtype=np.float32))
    np.testing.assert_array_equal(norm.get_original_reward(),[-90.])


def test_detached_checkpoint_can_be_promoted_and_saved_without_constructing_or_resetting_a_plant(tmp_path):
    import pickle
    norm=RoadCandidateVecNormalize(DummyVecEnv([PhysicalReward]),norm_reward=False)
    norm.training=False;norm.save(tmp_path/'original.pkl')
    with (tmp_path/'original.pkl').open('rb') as stream:detached=pickle.load(stream)
    assert detached.venv is None
    initialize_fixed_reward_units(detached,.05)
    with (tmp_path/'scaled.pkl').open('wb') as stream:pickle.dump(detached,stream)
    loaded=VecNormalize.load(tmp_path/'scaled.pkl',DummyVecEnv([PhysicalReward]))
    np.testing.assert_array_equal(loaded.normalize_reward(np.array([-90.])),[-4.5])

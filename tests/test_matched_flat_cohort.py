"""Synthetic learner/dependency checks,not CARLA performance evidence."""
import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.road_candidate_policy import OrderedFlatRoadCandidatePolicy,RoadCandidateVecNormalize
from scripts.run_matched_flat_cohort import standardize,dependency_complete,protocol_for
from navstack.research import atomic_json


class ObservationFixture(gym.Env):
    observation_space=gym.spaces.Box(-np.inf,np.inf,(4705,),dtype=np.float32)
    action_space=gym.spaces.Box(-1.,1.,(2,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        return np.zeros(4705,dtype=np.float32),{}
    def step(self,action):return np.zeros(4705,dtype=np.float32),0.,False,False,{}


def test_recipe_standardization_rebuilds_rollout_without_changing_policy_means_or_values(tmp_path):
    torch.set_num_threads(1)
    raw=DummyVecEnv([ObservationFixture]);norm=RoadCandidateVecNormalize(raw,norm_reward=False)
    old=PPO(OrderedFlatRoadCandidatePolicy,norm,n_steps=320,batch_size=320,n_epochs=10,seed=33,device='cpu')
    old.save(tmp_path/'old.zip');obs=norm.normalize_obs(raw.reset());count=norm.obs_rms.count
    try:
        new,changed,proof=standardize(tmp_path/'old.zip',norm)
        assert new.n_steps==new.rollout_buffer.buffer_size==1280
        assert new.batch_size==320 and new.n_epochs==10 and not new.policy.optimizer.state
        assert new.learning_rate==3e-6 and new.target_kl==.01
        assert new.policy.log_std.detach().tolist()==[-4.,-4.]
        np.testing.assert_array_equal(old.predict(obs,deterministic=True)[0],new.predict(obs,deterministic=True)[0])
        with torch.no_grad():
            torch.testing.assert_close(old.policy.predict_values(torch.as_tensor(obs)),
                new.policy.predict_values(torch.as_tensor(obs)),rtol=0,atol=0)
        assert changed.obs_rms.count==count
        np.testing.assert_array_equal(changed.normalize_reward(np.array([-90.,20.])),[-4.5,1.])
        assert proof['all_weights_exact_except_log_std'] and proof['physical_reward_unchanged']
        old.num_timesteps=1;old.save(tmp_path/'used.zip')
        with pytest.raises(ValueError):standardize(tmp_path/'used.zip',changed)
    finally:raw.close()


def test_dependency_requires_every_closed_job_and_rejects_recorded_failure(tmp_path):
    (tmp_path/'execution').mkdir()
    assert dependency_complete(tmp_path) is False
    atomic_json(tmp_path/'execution/complete.json',dict(status='ALL_REGISTERED_JOBS_CLOSED',seeds=[11,22,33,44,55]))
    with pytest.raises(FileNotFoundError):dependency_complete(tmp_path)
    atomic_json(tmp_path/'execution/failure.json',dict(error='retained native failure'))
    with pytest.raises(RuntimeError):dependency_complete(tmp_path)


def test_unknown_flat_phase_is_rejected_before_any_staging(tmp_path):
    with pytest.raises(ValueError):protocol_for(tmp_path,{},'best_checkpoint',tmp_path)
    assert not list(tmp_path.iterdir())

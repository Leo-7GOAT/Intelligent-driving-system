import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from navstack.candidate_policy import CandidateActorCriticPolicy, CandidateVecNormalize
from navstack.config import EnvConfig, PlannerConfig
from navstack.env import NavigationEnv
from navstack.experiments import load_run, signature
from navstack.research import cumulative_train


def config(feedback="enabled", method="H"):
    return EnvConfig(method=method, scenario="empty", deadline=45, obstacle_num=0,
                     execution_feedback=feedback, candidate_observation=True,
                     subgoal_mode="goal_frame_terminal",
                     planner=PlannerConfig(max_seconds=None, analytic_expansion=True, vectorized_primitives=True))


def test_candidate_geometry_common_and_feedback_ablated():
    observations = []
    for method in ("H", "B1", "B2", "B3"):
        env = NavigationEnv(config(method=method))
        obs, _ = env.reset(seed=11)
        assert obs.shape == (605,) and env.observation_space.contains(obs)
        observations.append(obs)
        env.close()
    for obs in observations[1:]:
        np.testing.assert_array_equal(obs, observations[0])
    env = NavigationEnv(config("ablated"))
    ablated, _ = env.reset(seed=11)
    np.testing.assert_array_equal(ablated[351:599].reshape(31, 8)[:, :5], observations[0][351:599].reshape(31, 8)[:, :5])
    assert not ablated[351:599].reshape(31, 8)[:, 5:].any()
    assert not ablated[599:].any()
    assert signature(config())["observation_shape"] == [605]
    env.close()


def test_candidate_policy_scores_permute_without_masking():
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: NavigationEnv(config())])
    vec = CandidateVecNormalize(env, norm_reward=False)
    obs = vec.reset()
    original = vec.get_original_obs()
    np.testing.assert_array_equal(obs[:, 351:599], original[:, 351:599])
    np.testing.assert_allclose(vec.unnormalize_obs(obs)[:, 351:599], original[:, 351:599])
    model = PPO(CandidateActorCriticPolicy, vec, n_steps=8, batch_size=8, seed=11)
    permutation = np.random.default_rng(2).permutation(31)
    changed = obs.copy()
    changed[:, 351:599] = obs[:, 351:599].reshape(-1, 31, 8)[:, permutation].reshape(-1, 248)
    with torch.no_grad():
        a = model.policy.get_distribution(torch.as_tensor(obs)).distribution.probs
        b = model.policy.get_distribution(torch.as_tensor(changed)).distribution.probs
    torch.testing.assert_close(b, a[:, permutation])
    assert torch.isfinite(a).all() and (a > 0).all()
    vec.close()


def test_candidate_ppo_updates_resumes_and_reloads(tmp_path):
    options = {"policy": "candidate_shared", "target_kl": .03}
    first = cumulative_train(tmp_path, config(), 11, 640, time.time()+120,
                             rollout_physics=320, epochs=2, ppo_options=options)
    second = cumulative_train(tmp_path, config(), 11, 960, time.time()+120,
                              rollout_physics=320, epochs=2, ppo_options=options)
    assert first["optimization_phases"] >= 2
    assert second["optimization_phases"] > first["optimization_phases"]
    assert second["physics_steps"] >= 960
    assert second["actor_l2_change"] > 0 and second["reload_exact"]
    model, normalizer = load_run(second["checkpoint"])
    assert isinstance(model.policy, CandidateActorCriticPolicy)
    assert isinstance(normalizer, CandidateVecNormalize)
    normalizer.close()

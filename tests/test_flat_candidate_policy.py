import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.candidate_policy import CandidateVecNormalize, FlatCandidateActorCriticPolicy
from navstack.config import EnvConfig, PlannerConfig
from navstack.env import NavigationEnv
from navstack.experiments import load_run
from navstack.research import cumulative_train


def config():
    return EnvConfig(method="B1", flat_hold=10, scenario="empty", deadline=45, obstacle_num=0,
                     candidate_observation=True, execution_feedback="enabled", subgoal_mode="goal_frame_terminal",
                     planner=PlannerConfig(max_seconds=None, analytic_expansion=True, vectorized_primitives=True))


def test_flat_actor_uses_candidate_set_without_order_dependence():
    torch.set_num_threads(1)
    vec = CandidateVecNormalize(DummyVecEnv([lambda: NavigationEnv(config())]), norm_reward=False)
    try:
        obs = vec.reset()
        model = PPO(FlatCandidateActorCriticPolicy, vec, n_steps=8, batch_size=8, use_sde=True, seed=11)
        changed = obs.copy()
        permutation = np.random.default_rng(3).permutation(31)
        changed[:, 351:599] = obs[:, 351:599].reshape(-1, 31, 8)[:, permutation].reshape(-1, 248)
        with torch.no_grad():
            original = model.policy.get_distribution(torch.as_tensor(obs)).distribution
            reordered = model.policy.get_distribution(torch.as_tensor(changed)).distribution
        torch.testing.assert_close(original.mean, reordered.mean)
        torch.testing.assert_close(original.stddev, reordered.stddev)
        # The supplied candidate information actually influences the actor.
        tensor = torch.as_tensor(obs).requires_grad_(True)
        model.policy.get_distribution(tensor).distribution.mean.sum().backward()
        assert tensor.grad[:, 351:599].abs().sum() > 0
        assert original.mean.shape == (1, 2) and torch.isfinite(original.stddev).all()
    finally:
        vec.close()


def test_pooled_flat_sde_updates_all_actor_branches_and_resumes(tmp_path):
    options = {"policy": "flat_candidate_pool", "use_sde": True, "sde_sample_freq": 1,
               "target_kl": .03, "log_std_init": -2}
    first = cumulative_train(tmp_path, config(), 11, 640, time.time()+120,
                             rollout_physics=320, epochs=2, ppo_options=options)
    old, old_normalizer = load_run(first["checkpoint"])
    saved = {k: v.clone() for k, v in old.policy.state_dict().items()}
    old_normalizer.close()
    second = cumulative_train(tmp_path, config(), 11, 1280, time.time()+120,
                              rollout_physics=320, epochs=2, ppo_options=options)
    model, normalizer = load_run(second["checkpoint"])
    try:
        assert isinstance(model.policy, FlatCandidateActorCriticPolicy)
        assert first["optimization_phases"] >= 2 and second["optimization_phases"] > first["optimization_phases"]
        assert second["reload_exact"] and second["actor_l2_change"] > 0
        current = model.policy.state_dict()
        for branch in ("policy_net_context", "policy_net_candidate", "policy_net_fusion"):
            assert any(not torch.equal(v, current[k]) for k, v in saved.items() if branch in k)
        assert normalizer.training is False
    finally:
        normalizer.close()

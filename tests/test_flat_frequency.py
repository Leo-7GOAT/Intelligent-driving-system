"""A held flat action must preserve the actual per-physics-step dynamics/reward."""
from dataclasses import asdict
import numpy as np
import pytest
from navstack.config import EnvConfig
from navstack.env import NavigationEnv


@pytest.mark.parametrize("action", ([.4, .1], [-.2, -.1]))
def test_flat_hold_matches_ten_direct_physics_steps(action):
    common = dict(method="B1", scenario="empty", split="validation", deadline=45,
                  obstacle_num=0, subgoal_mode="goal_frame_terminal",
                  candidate_observation=True, execution_feedback="enabled")
    fast = NavigationEnv(EnvConfig(flat_hold=1, **common))
    held = NavigationEnv(EnvConfig(flat_hold=10, **common))
    try:
        fast.reset(seed=7301)
        held.reset(seed=7301)
        expected_reward = 0.
        for index in range(10):
            _, reward, terminated, truncated, info = fast.step(action)
            assert not terminated and not truncated
            expected_reward += fast.config.gamma_low**index * reward
            assert info["trace"][0]["controller_status"] == "direct"
        _, actual_reward, terminated, truncated, info = held.step(action)
        assert not terminated and not truncated
        np.testing.assert_allclose(list(asdict(fast.state).values()), list(asdict(held.state).values()), atol=1e-12)
        assert actual_reward == pytest.approx(expected_reward)
        assert held.config.gamma == pytest.approx(fast.config.gamma**10)
        assert held.state.time == pytest.approx(.5)
        assert held.reset_count == fast.reset_count == 1
        assert held.total_physics_steps == fast.total_physics_steps == 10
        assert held.total_actions == 1 and fast.total_actions == 10
        assert len(info["trace"]) == 10
        assert all(row["controller_status"] == "direct" for row in info["trace"])
        assert held.reference is None and fast.reference is None
    finally:
        fast.close()
        held.close()

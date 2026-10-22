import math
import pytest

from navstack.config import EnvConfig, RewardConfig
from navstack.experiments import signature
from navstack.reward import RewardCalculator
from navstack.vehicle import VehicleState


def test_terminal_pose_potential_bounded_periodic_and_not_a_base_reward():
    original = RewardCalculator(RewardConfig(beta=40, progress_weight=.25), .05)
    shaped = RewardCalculator(RewardConfig(beta=40, progress_weight=.25, heading_potential_weight=.1), .05)
    goal = (27, 12, 0)
    aligned = VehicleState(25, 12, 0)
    wrong = VehicleState(25, 12, .8)
    assert shaped.potential(aligned, goal) == original.potential(aligned, goal)
    assert original.potential(wrong, goal)-.2 <= shaped.potential(wrong, goal) < original.potential(wrong, goal)
    assert shaped.potential(VehicleState(25, 12, .8+2*math.pi), goal) == pytest.approx(shaped.potential(wrong, goal))
    _, old_parts, _ = original.calculate(wrong, aligned, goal, 0)
    _, new_parts, _ = shaped.calculate(wrong, aligned, goal, 0)
    assert new_parts["potential"] > old_parts["potential"]
    assert {k:v for k,v in new_parts.items() if k != "potential"} == {k:v for k,v in old_parts.items() if k != "potential"}


def test_heading_shaping_telescopes_for_terminal_and_external_truncation():
    config = RewardConfig(beta=40, heading_potential_weight=.1)
    reward = RewardCalculator(config, .05)
    states = [VehicleState(4, 12, .1), VehicleState(10, 12, .5), VehicleState(20, 12, -.3), VehicleState(26.6, 12, .8)]
    goal = (27, 12, 0)
    for terminal in (False, True):
        terms = [reward.calculate(a, b, goal, 0, terminated=terminal and i==len(states)-2)[1]["potential"]
                 for i, (a, b) in enumerate(zip(states, states[1:]))]
        actual = sum(reward.gamma**i * value for i, value in enumerate(terms))
        expected = config.beta * (-reward.potential(states[0], goal)
                    + (0 if terminal else reward.gamma**len(terms)*reward.potential(states[-1], goal)))
        assert actual == pytest.approx(expected, abs=1e-10)


def test_disabled_shaping_keeps_old_signature_and_active_version_is_explicit():
    config = EnvConfig()
    assert "heading_potential_weight" not in signature(config)["reward"]
    config.reward.heading_potential_weight = .1
    assert signature(config)["reward"]["heading_potential_weight"] == .1
    for kwargs in ({"heading_potential_weight": -1}, {"heading_potential_radius": 0}, {"heading_potential_radius": float("nan")}):
        with pytest.raises(ValueError):
            RewardConfig(**kwargs)

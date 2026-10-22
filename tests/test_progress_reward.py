import pytest
from navstack.config import RewardConfig
from navstack.reward import RewardCalculator
from navstack.vehicle import VehicleState


def test_progress_reward_retains_measured_progress_at_failed_terminal():
    calculator = RewardCalculator(RewardConfig(beta=0, progress_weight=1), .05)
    before, after = VehicleState(4, 12), VehicleState(5, 12)
    _, parts, _ = calculator.calculate(before, after, (10, 12, 0), 0, "deadline", True)
    assert parts["task_progress"] == pytest.approx(1)
    assert parts["event"] == -10
    _, overshoot, _ = calculator.calculate(VehicleState(10, 12), VehicleState(11, 12), (10, 12, 0), 0)
    assert overshoot["task_progress"] == pytest.approx(-1)


def test_progress_is_physical_distance_additive_and_not_wait_bonus():
    calculator = RewardCalculator(RewardConfig(beta=0, progress_weight=1), .05)
    states = [VehicleState(x, 12) for x in (4, 4.4, 5)]
    def progress(a, b): return calculator.calculate(a, b, (10, 12, 0), 0)[1]["task_progress"]
    assert progress(states[0], states[2]) == pytest.approx(progress(states[0], states[1]) + progress(states[1], states[2]))
    assert progress(states[0], states[0]) == 0

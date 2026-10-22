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


def test_quarter_progress_reward_full_route_ranking_and_failure_margin():
    # Reward-only mathematical checks, not simulated vehicle performance.
    import numpy as np
    calculator = RewardCalculator(RewardConfig(beta=40, progress_weight=.25), .05)
    goal = (27, 12, 0)
    def score(xs, event):
        states = [VehicleState(float(x), 12) for x in xs]
        total = 0.
        for index, (before, after) in enumerate(zip(states, states[1:])):
            terminal = index == len(states)-2
            reward, _, _ = calculator.calculate(before, after, goal, 0, event if terminal else None, terminal)
            total += calculator.gamma**index * reward
        return total
    idle = score(np.full(901, 4.), "deadline")
    partial = score(np.r_[np.linspace(4, 14, 401), np.full(500, 14.)], "deadline")
    success = score(np.linspace(4, 27, 601), "success")
    wait_success = score(np.r_[np.full(100, 4.), np.linspace(4, 27, 601)], "success")
    loops = score(4+10*np.sin(np.linspace(0, 4*np.pi, 901))**2, "deadline")
    assert success > wait_success > max(partial, idle, loops)
    assert partial > idle
    for ticks in (100, 400, 900):
        for event in ("collision", "out_of_bounds"):
            assert score(np.linspace(4, 26.9, ticks+1), event) < idle
    # Nonnegative remaining distance bounds discounted progress return by
    # weight * initial distance, even with revisits; here this is below the
    # discounted bounds-vs-deadline event margin at the45s task deadline.
    assert .25 * 23 < 10 * .99**45

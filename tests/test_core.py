import math
from dataclasses import replace
import numpy as np
import pytest
from navstack.config import EnvConfig, VehicleConfig, PlannerConfig
from navstack.vehicle import VehiclePlant, VehicleState, Command, wrap
from navstack.world import World, MovingObstacle
from navstack.planning import (
    HybridAStar,
    GridAStarPlanner,
    grid_search,
    primitive,
    PlanResult,
)
from navstack.trajectory import parameterize, tracking_error, dynamic_safe
from navstack.mpc import MPC, ControlResult
from navstack.env import NavigationEnv, ACTIONS
from navstack.reward import RewardCalculator


def empty(**kwargs):
    return NavigationEnv(EnvConfig(scenario="empty", **kwargs))


def test_no_teleport_or_reset_and_continuity():
    env = empty()
    obs, _ = env.reset(seed=2)
    state = env.state
    count = env.reset_count
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    for _ in range(2):
        _, _, terminated, truncated, info = env.step(16)
        assert not terminated and not truncated
        assert info["executed_substeps"] == 10
        assert info["plan"]["expanded_nodes"] > 0
        assert any(t["controller_status"] == "solved" for t in info["trace"])
    assert env.reset_count == count and env.state.time == pytest.approx(1.0)
    assert 0 < env.state.x - state.x < 1.1
    assert env.state.v > 0 and env.plant.previous_command.acceleration != 0


def test_no_reference_and_failure_advances_time(monkeypatch):
    env = empty()
    env.reset(seed=0)
    monkeypatch.setattr(
        env.planner,
        "plan",
        lambda *a: PlanResult("no_path", np.empty((0, 3)), None, 2, 0.001, "injected"),
    )
    _, _, done, _, info = env.step(16)
    assert not done and not info["success"] and env.state.time == pytest.approx(0.5)
    assert info["trace"][0]["tracking"]["lateral"] is None
    assert env.episode["interventions"] == 10
    assert tracking_error(env.state, None)["missing_reason"] == "no_valid_reference"


def test_qp_failure_not_success(monkeypatch):
    env = empty()
    env.reset(seed=0)
    monkeypatch.setattr(
        env.controller,
        "step",
        lambda *a: ControlResult(
            Command(-2.5, 0),
            "failed",
            {"qp_seconds": 0.0, "control_seconds": 0.0, "reason": "injected"},
        ),
    )
    _, _, _, _, info = env.step(16)
    assert not info["success"] and env.state.x == 4
    assert env.episode["mpc_failures"] == 10 and env.state.time > 0.49


def test_actual_collision_and_bounds_failure_priority():
    env = empty(method="B1")
    env.reset(seed=1)
    env.world.static = [(4, 10, 5, 15)]
    env.goal = (4, env.state.y, 0)
    _, _, done, _, info = env.step(np.array([1.0, 0.0]))
    assert done and info["termination_reason"] == "collision" and not info["success"]
    env.reset(seed=1)
    env.plant = VehiclePlant(
        env.config.vehicle, VehicleState(env.world.width - 0.1, 12, 0, 2)
    )
    _, _, done, _, info = env.step(np.array([1.0, 0.0]))
    assert done and info["termination_reason"] == "out_of_bounds"
    assert env.state.x > env.world.width - 0.1


def test_empty_path_stationary_never_success():
    env = empty(deadline=1.0)
    env.reset(seed=1)
    env.goal = (env.state.x, env.state.y, 0)
    env.step(0)
    _, _, term, _, info = env.step(0)
    assert term and info["termination_reason"] == "deadline"


def test_random_obstacles_and_map_invariant_semantics():
    a = NavigationEnv(EnvConfig(scenario="static", obstacle_num=5))
    b = NavigationEnv(EnvConfig(scenario="static", obstacle_num=5))
    a.reset(seed=12)
    b.reset(seed=12)
    assert len(a.world.static) == 6 and a.world.static == b.world.static
    b.reset(seed=13)
    assert a.world.static != b.world.static
    large = NavigationEnv(EnvConfig(width=44, height=30))
    large.reset(seed=12)
    assert (
        a.action_space == large.action_space
        and a.observation_space.shape == large.observation_space.shape
    )
    for i in range(len(ACTIONS)):
        ga, _ = a.decode_action(i)
        gb, _ = large.decode_action(i)
        np.testing.assert_allclose(
            np.array(ga[:2]) - [a.state.x, a.state.y],
            np.array(gb[:2]) - [large.state.x, large.state.y],
        )


def test_astar_dijkstra_cost_and_corner_cutting():
    rng = np.random.default_rng(42)
    for _ in range(30):
        obs = {tuple(p) for p in rng.integers(0, 12, size=(22, 2))} - {(0, 0), (11, 11)}
        a, ca, _ = grid_search((0, 0), (11, 11), obs, 12)
        d, cd, _ = grid_search((0, 0), (11, 11), obs, 12, True)
        assert ca == pytest.approx(cd) and bool(a) == bool(d)
    assert not grid_search((0, 0), (1, 1), {(1, 0), (0, 1)}, 2)[0]


def test_hybrid_actual_bicycle_and_full_body():
    v = VehicleConfig()
    c = PlannerConfig(max_seconds=1.0)
    world = World(34, 24)
    planner = HybridAStar(v, c)
    end = primitive((4, 12, 0), 0.25, 6.4, v.wheelbase, 0.1)[-1]
    result = planner.plan((4, 12, 0), end, world)
    assert result.status == "success" and result.expanded_nodes > 0
    ds = np.linalg.norm(np.diff(result.path[:, :2], axis=0), axis=1)
    assert (
        np.max(abs(np.diff(np.unwrap(result.path[:, 2])) / ds))
        <= 1 / v.min_turn_radius + 0.002
    )
    assert np.all(ds > 0) and np.all(
        np.sum(
            np.diff(result.path[:, :2], axis=0)
            * np.column_stack(
                [np.cos(result.path[:-1, 2]), np.sin(result.path[:-1, 2])]
            ),
            axis=1,
        )
        > 0
    )
    world.static = [
        (6.5, 11, 6.6, 13)
    ]  # obstacle touches nose, although rear axle is clear
    assert planner.plan((4, 12, 0), (12, 12, 0), world).status == "invalid"
    wall = World(34, 24, [(12, 0, 12.02, 24)])
    result = planner.plan((4, 12, 0), (20, 12, 0), wall)
    assert result.status in ("timeout", "no_path")


def test_grid_postprocess_real_chain():
    env = empty(method="B2")
    env.reset(seed=3)
    _, _, _, _, info = env.step(16)
    assert info["plan"]["status"] == "success"
    assert env.state.x > 4 and any(
        t["controller_status"] == "solved" for t in info["trace"]
    )


def test_arc_time_yaw_and_first_steering_rate():
    v = VehicleConfig()
    path = np.vstack(
        [
            [5, 10, math.pi - 0.05],
            primitive((5, 10, math.pi - 0.05), 0.2, 2.0, v.wheelbase, 0.1),
        ]
    )
    reference = parameterize(path, 1.0, 0.5, v)
    assert np.all(np.diff(reference.s) > 0) and np.all(np.diff(reference.t) > 0)
    assert max(abs(np.diff(reference.yaw))) < 0.1
    assert reference.v[-1] > 0  # local waypoint is not a parking command
    state = VehicleState(5, 10, -math.pi - 0.05, 0.5)
    mpc = MPC(v)
    prev = Command(0.2, 0.2)
    before = state
    result = mpc.step(state, reference, prev)
    assert result.status == "solved" and state == before
    assert abs(result.command.steering - prev.steering) <= v.steer_rate * 0.05 + 1e-8


def test_projection_not_vertex_distance_and_no_snap():
    v = VehicleConfig()
    ref = parameterize(np.array([[4.0, 10, 0], [14, 10, 0]]), 1.0, 1.0, v, spacing=10.0)
    err = tracking_error(VehicleState(7, 11, 0, 1), ref)
    assert err["lateral"] == pytest.approx(1.0)
    plant = VehiclePlant(v, VehicleState(4, 10, 0, 1))
    state = plant.step(Command(), 0.05)
    assert state.x == pytest.approx(4.05)


def test_dynamic_time_collision_and_wait():
    v = VehicleConfig()
    world = World(34, 24, dynamic=[MovingObstacle(8, 8, 0, 1)])
    ref = parameterize(np.array([[4, 10, 0], [14, 10, 0]]), 1.0, 1.0, v)
    assert not dynamic_safe(ref, world, v, 3)
    env = empty()
    env.reset(seed=3)
    env.world.dynamic = [MovingObstacle(8, env.state.y - 2, 0, 1)]
    _, _, _, _, info = env.step(16)
    assert info["mode"] == "dynamic_brake" and env.state.time > 0.49
    assert env.world.dynamic[0].y == pytest.approx(env.state.y - 1.5)


def test_delay_and_mismatch_really_affect_plant():
    nominal = VehiclePlant(VehicleConfig(steer_tau=0), VehicleState(4, 10, 0, 1))
    delayed = VehiclePlant(
        VehicleConfig(
            steer_tau=0.5,
            input_delay_steps=3,
            wheelbase_scale=1.2,
            acceleration_scale=0.8,
        ),
        VehicleState(4, 10, 0, 1),
    )
    for _ in range(20):
        nominal.step(Command(1, 0.3), 0.05)
        delayed.step(Command(1, 0.3), 0.05)
    assert nominal.state.yaw != pytest.approx(
        delayed.state.yaw
    ) and nominal.state.v != pytest.approx(delayed.state.v)


def test_discount_reward_terminal_and_external_truncation():
    c = EnvConfig()
    r = RewardCalculator(c.reward, c.control_dt)
    s0 = VehicleState(4, 12)
    s1 = VehicleState(4.1, 12)
    _, parts, _ = r.calculate(s0, s1, (25, 12, 0), 0)
    assert parts["potential"] == pytest.approx(
        c.gamma_low * r.potential(s1, (25, 12, 0)) - r.potential(s0, (25, 12, 0))
    )
    _, terminal, _ = r.calculate(s0, s1, (25, 12, 0), 0, "success", True)
    assert terminal["potential"] == -r.potential(s0, (25, 12, 0))
    assert c.gamma == pytest.approx(c.gamma_low**10)
    env = gym_time_limit(empty(), 1)
    env.reset(seed=5)
    _, _, terminated, truncated, info = env.step(0)
    assert truncated and not terminated and info["termination_reason"] is None
    assert info["trace"][-1]["reward_components"]["potential"] < 0.01
    with pytest.raises(ValueError):
        EnvConfig(decision_dt=0.27)


def gym_time_limit(env, steps):
    from gymnasium.wrappers import TimeLimit

    return TimeLimit(env, max_episode_steps=steps)


def test_complete_episode_reward_ranking():
    c = EnvConfig()
    calc = RewardCalculator(c.reward, 0.05)
    goal = (25, 12, 0)

    def total(xs, event, wait=0):
        states = [
            VehicleState(float(x), 12, v=0, acceleration=0, time=i * 0.05)
            for i, x in enumerate([4.0] * wait + list(xs))
        ]
        ret = 0.0
        for i, (a, b) in enumerate(zip(states, states[1:])):
            end = i == len(states) - 2
            reward, _, _ = calc.calculate(a, b, goal, 0, event if end else None, end)
            ret += calc.gamma**i * reward
        return ret

    safe = total(np.linspace(4, 25, 301), "success")
    wait_then_success = total(np.linspace(4, 25, 301), "success", 100)
    idle = total(np.full(601, 4), "deadline")
    invalid = total(np.full(601, 4), "deadline")
    loops = total(4 + np.sin(np.linspace(0, 12 * math.pi, 601)), "deadline")
    collision = total(np.linspace(4, 24, 80), "collision")
    assert safe > wait_then_success > max(idle, invalid, loops, collision)


def test_multienv_statistics_and_bootstrap():
    from stable_baselines3.common.vec_env import DummyVecEnv

    a = empty(deadline=0.5)
    b = empty(deadline=1.0)
    vec = DummyVecEnv([lambda: a, lambda: b])
    vec.reset()
    _, _, done, infos = vec.step([0, 0])
    assert (
        done.tolist() == [True, False]
        and "episode_metrics" in infos[0]
        and "episode_metrics" not in infos[1]
    )
    assert not infos[0]["TimeLimit.truncated"] and "terminal_observation" in infos[0]
    vec.close()
    vec = DummyVecEnv([lambda: gym_time_limit(empty(), 1)])
    vec.reset()
    _, _, done, info = vec.step([0])
    assert info[0]["TimeLimit.truncated"] and "terminal_observation" in info[0]
    vec.close()


def test_legacy_model_rejected(tmp_path):
    from navstack.experiments import load_run

    with pytest.raises(ValueError, match="legacy checkpoint"):
        load_run(tmp_path)


def test_gymnasium_interface():
    from stable_baselines3.common.env_checker import check_env

    check_env(empty(), warn=True)
    check_env(empty(method="B1"), warn=True)


def test_actual_success_after_wait_event_once():
    from navstack.env import rule_action

    returns = []
    for waits in (0, 10):
        env = empty(deadline=30.0)
        env.reset(seed=4)
        for _ in range(waits):
            env.step(0)
        while not env.done:
            _, _, _, _, info = env.step(rule_action(env))
        assert info["termination_reason"] == "success"
        assert env.state.v <= env.config.goal_speed and not env.collision_history
        assert env.episode["reward_components"]["event"] == 20
        assert 0 <= info["episode_metrics"]["path_efficiency_success_only"] <= 1
        with pytest.raises(RuntimeError):
            env.step(0)
        returns.append(env.episode["discounted_return"])
    assert returns[0] > returns[1]


def test_real_solver_exception_is_explicit_braking(monkeypatch):
    import osqp

    v = VehicleConfig()
    state = VehicleState(4, 12, 0, 1)
    reference = parameterize(np.array([[4, 12, 0], [10, 12, 0]]), 1.0, 1.0, v)

    def fail(*args, **kwargs):
        raise RuntimeError("fault injection")

    monkeypatch.setattr(osqp.OSQP, "setup", fail)
    result = MPC(v).step(state, reference, Command(0.1, 0))
    assert result.status == "failed" and result.command.acceleration < 0
    assert "solver_exception" in result.diagnostics["reason"]


def test_action_reward_equals_discounted_physical_rewards():
    env = empty()
    env.reset(seed=9)
    _, reward, _, _, info = env.step(16)
    expected = sum(
        env.config.gamma_low**i * tick["reward"] for i, tick in enumerate(info["trace"])
    )
    assert reward == pytest.approx(expected)


def test_exact_delay_queue_in_observation():
    env = NavigationEnv(
        EnvConfig(
            method="B1", scenario="empty", vehicle=VehicleConfig(input_delay_steps=2)
        )
    )
    env.reset(seed=2)
    obs, *_ = env.step(np.array([1.0, 0.5], np.float32))
    assert obs[5] == 2 and np.any(obs[6:10] != 0)
    assert env.state.v == 0


def test_real_static_detour_planner_qp_and_plant():
    v = VehicleConfig()
    world = World(40, 30, [(14, 13, 16, 17)])
    planner = HybridAStar(v, PlannerConfig(max_seconds=1.0, max_nodes=10000))
    result = planner.plan((4, 15, 0), (26, 15, 0), world)
    assert result.status == "success" and result.expanded_nodes > 0
    assert max(abs(result.path[:, 1] - 15)) > 3  # genuine obstacle detour
    ref = parameterize(result.path, 1.5, 0.0, v, stop=True)
    plant = VehiclePlant(v, VehicleState(4, 15))
    controller = MPC(v)
    for _ in range(450):
        command = controller.step(plant.state, ref, plant.previous_command)
        assert command.status == "solved"
        s = plant.step(command.command, 0.05)
        assert not world.collision((s.x, s.y, s.yaw), v)
        assert not world.out_of_bounds((s.x, s.y, s.yaw), v)
    assert math.hypot(s.x - 26, s.y - 15) < 0.8 and s.v < 0.25

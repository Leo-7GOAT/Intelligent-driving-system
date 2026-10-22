import numpy as np
import pytest
from navstack.config import VehicleConfig, PlannerConfig, EnvConfig
from navstack.world import World
from navstack.planning import HybridAStar, analytic_connection
from navstack.trajectory import parameterize
from navstack.env import NavigationEnv, rule_action


def test_connector_has_correct_heading_and_rejects_body_collision_and_tight_turn():
    vehicle = VehicleConfig()
    world = World(34,24)
    start, goal = (4,12,0), (16,15,0)
    path = analytic_connection(start, goal, world, vehicle)
    assert path is not None
    np.testing.assert_allclose(path[[0,-1]], [start,goal], atol=1e-8)
    ref = parameterize(path,1,0,vehicle)
    assert np.max(np.abs(ref.kappa)) < 1/vehicle.min_turn_radius
    world.static = [(8,10,13,17)]
    assert analytic_connection(start,goal,world,vehicle) is None
    assert analytic_connection(start,(4.5,15,0),World(34,24),vehicle) is None


def test_goal_frame_reference_executes_and_does_not_move_measured_state_during_planning():
    config = EnvConfig(subgoal_mode="goal_frame", scenario="empty", planner=PlannerConfig(max_seconds=None,analytic_expansion=True))
    env = NavigationEnv(config)
    env.reset(seed=7301)
    state = env.state
    mode, detail = env._prepare_reference(26)
    assert mode == "reference" and detail["expanded_nodes"] == 1
    assert env.state == state
    _, _, _, _, info = env.step(26)
    assert env.state.time == pytest.approx(.5)
    assert env.state.x > state.x
    assert all(t["controller_status"] == "solved" for t in info["trace"])

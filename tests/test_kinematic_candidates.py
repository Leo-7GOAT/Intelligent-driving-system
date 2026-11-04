from dataclasses import asdict
import math

import numpy as np
import pytest

from navstack.config import EnvConfig, PlannerConfig, VehicleConfig
from navstack.experiments import signature
from navstack.planning import HybridAStar, primitive
from navstack.subgoals import kinematic_arc_goal_candidates
from navstack.vehicle import VehicleState
from navstack.world import World
from test_route_catalog_environment import RouteFixture, config


def test_arc_proposals_match_vehicle_primitives_without_moving_measured_state():
    vehicle = VehicleConfig()
    state = VehicleState(20., 20., yaw=2.9)
    before = asdict(state)
    goal = (33., 23., -.2)
    candidates = kinematic_arc_goal_candidates(state, goal, vehicle)
    assert len(candidates) == 31 and candidates[0] == (None, 0.)
    assert candidates[25:27] == ((goal, .75), (goal, 1.5))
    index = 1
    for length in (3., 6., 9.):
        for fraction in (-1., -.5, 0., .5, 1.):
            expected = primitive((state.x, state.y, state.yaw),
                                 fraction * vehicle.max_steer, length,
                                 vehicle.wheelbase, .12)[-1]
            expected[2] = math.atan2(math.sin(expected[2]), math.cos(expected[2]))
            if index != 25:
                for offset, speed in enumerate((.75, 1.5)):
                    np.testing.assert_allclose(candidates[index + offset][0], expected, atol=1e-12)
                    assert candidates[index + offset][1] == speed
            index += 2
    assert asdict(state) == before


def test_local_arc_poses_connect_through_existing_body_checked_planner():
    vehicle = VehicleConfig()
    state = VehicleState(20., 20.)
    world = World(50., 50.)
    planner = HybridAStar(vehicle, PlannerConfig(max_nodes=7200, max_seconds=None,
                                               analytic_expansion=True))
    candidates = kinematic_arc_goal_candidates(state, (38., 20., 0.), vehicle)
    for index in range(1, 31, 2):
        if index == 25:
            continue
        result = planner.plan((state.x, state.y, state.yaw), candidates[index][0], world)
        assert result.status == 'success', (index, result.status)
        assert all(world.free(pose, vehicle) for pose in result.path)


def test_geometry_versions_preserve_old_serialization_and_share_information_with_flat():
    old = config()
    assert 'route_candidate_geometry' not in old.to_dict()
    new = EnvConfig.from_dict({**old.to_dict(), 'route_goal_candidate': True,
                              'route_candidate_geometry': 'kinematic_arcs_goal_v1'})
    flat = EnvConfig.from_dict({**new.to_dict(), 'method': 'B1'})
    assert signature(old) != signature(new)
    assert signature(flat)['candidate_catalog'] == signature(new)['action_catalog']
    with pytest.raises(ValueError):
        EnvConfig.from_dict({**old.to_dict(), 'route_candidate_geometry': 'kinematic_arcs_goal_v1'})
    h, b = RouteFixture(new), RouteFixture(flat)
    try:
        ho, _ = h.reset(seed=11)
        bo, _ = b.reset(seed=11)
        np.testing.assert_array_equal(ho, bo)
        assert h.decode_action(25)[0] == h.goal
        def forbidden(*args):
            raise AssertionError('Flat control must not invoke planning')
        b.planner.plan = forbidden
        b.step(np.array([.1, 0.], dtype=np.float32))
    finally:
        h.close()
        b.close()


def test_offline_inventory_deduplicates_speed_pairs_and_never_executes(monkeypatch):
    from scripts.probe_native_candidate_coverage import candidate_plan_inventory
    cfg = EnvConfig.from_dict({**config().to_dict(), 'route_goal_candidate': True,
                              'route_candidate_geometry': 'kinematic_arcs_goal_v1'})
    env = RouteFixture(cfg)
    try:
        env.reset(seed=11)
        before = asdict(env.state)
        original = env.planner.plan
        calls = []
        def plan(*args):
            calls.append(args[1])
            return original(*args)
        def forbidden(*args):
            raise AssertionError('Offline coverage must not execute the plant')
        monkeypatch.setattr(env.planner, 'plan', plan)
        monkeypatch.setattr(env.plant, 'step', forbidden)
        result = candidate_plan_inventory(env)
        assert result['unique_targets'] == 15
        assert len(calls) == len(set(calls)) == result['endpoint_clear_actions'] // 2
        assert len(result['rows']) == 30 and asdict(env.state) == before
    finally:
        env.close()

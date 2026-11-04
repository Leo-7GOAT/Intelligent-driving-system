from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from navstack.config import PlannerConfig, VehicleConfig
from navstack.planning import HybridAStar
from navstack.planner_memo import ExactStaticFailureMemo, VERSION, configure_planner_memo
from navstack.road_grid import RoadGrid, road_digest
from navstack.world import World, MovingObstacle


def fixture(max_nodes=1):
    vehicle = VehicleConfig()
    config = PlannerConfig(max_nodes=max_nodes, max_seconds=None, analytic_expansion=False)
    world = World(40., 30.)
    planner = HybridAStar(vehicle, config)
    return planner, world, (6., 8., 0.), (29., 20., .5)


def test_exact_repeat_preserves_failure_but_charges_zero_new_nodes_and_separates_original_work():
    planner, world, start, goal = fixture()
    memo = ExactStaticFailureMemo(planner)
    first = memo.plan(start, goal, world)
    second = memo.plan(start, goal, world)
    assert first.status == second.status == 'timeout'
    assert first.failure_reason == second.failure_reason == 'search_budget'
    assert first.cost == second.cost is None
    np.testing.assert_array_equal(first.path, second.path)
    assert first.expanded_nodes == second.cached_expanded_nodes == 1
    assert second.expanded_nodes == 0 and second.cache_hit and second.planning_time >= 0
    assert memo.hits == memo.misses == 1


@pytest.mark.parametrize('change', ['pose', 'goal', 'static', 'width', 'vehicle', 'budget', 'backend', 'world', 'replace_config'])
def test_every_relevant_input_change_requires_another_real_search(change):
    planner, world, start, goal = fixture()
    memo = ExactStaticFailureMemo(planner)
    memo.plan(start, goal, world)
    if change == 'pose': start = (start[0] + 1e-12, *start[1:])
    if change == 'goal': goal = (goal[0] + 1e-12, *goal[1:])
    if change == 'static': world.static.append((18., 18., 19., 19.))
    if change == 'width': world.width += .01
    if change == 'vehicle': planner.vehicle.width += .01
    if change == 'budget': planner.config.max_nodes += 1
    if change == 'backend': world.road_collision_backend = 'batch_sat_v1'
    if change == 'world': world = World(world.width, world.height)
    if change == 'replace_config': planner.config = PlannerConfig(max_nodes=2, max_seconds=None)
    result = memo.plan(start, goal, world)
    assert not getattr(result, 'cache_hit', False)
    assert memo.hits == 0 and memo.misses == 2


def test_road_content_mutation_invalidates_without_changing_original_digest():
    planner, world, start, goal = fixture()
    payload = {'version': 'carla_driving_64_v1', 'width': 40., 'height': 30.,
               'drivable': np.ones((64, 64), dtype=int).tolist()}
    payload['sha256'] = road_digest(payload)
    world.road = RoadGrid(payload)
    memo = ExactStaticFailureMemo(planner)
    memo.plan(start, goal, world)
    world.road.drivable[0, 0] = False
    memo.plan(start, goal, world)
    assert memo.hits == 0 and memo.misses == 2


def test_measured_offroad_truth_invalidates_a_previous_geometric_failure():
    from navstack.carla_bridge import CarlaWorld
    planner, _, start, goal = fixture()
    world = CarlaWorld(40., 30.)
    memo = ExactStaticFailureMemo(planner)
    assert memo.plan(start, goal, world).status == 'timeout'
    world.measured_offroad = True
    result = memo.plan(start, goal, world)
    assert result.status == 'invalid' and not getattr(result, 'cache_hit', False)
    assert memo.hits == 0 and memo.misses == 2


@pytest.mark.parametrize('kind', ['dynamic', 'clock', 'custom_world'])
def test_dynamic_wall_clock_and_unknown_collision_implementations_bypass(kind):
    planner, world, start, goal = fixture()
    if kind == 'dynamic': world.dynamic.append(MovingObstacle(20., 10., 0., 1.))
    if kind == 'clock': planner.config.max_seconds = .1
    if kind == 'custom_world':
        class CustomWorld(World): pass
        world = CustomWorld(40., 30.)
    memo = ExactStaticFailureMemo(planner)
    memo.plan(start, goal, world)
    memo.plan(start, goal, world)
    assert memo.hits == memo.misses == 0 and memo.bypasses == 2


def test_success_is_never_cached_and_opt_in_does_not_change_configuration():
    planner, world, start, _ = fixture()
    memo = ExactStaticFailureMemo(planner)
    for _ in range(2): assert memo.plan(start, start, world).status == 'success'
    assert memo.hits == 0
    env = SimpleNamespace(planner=planner)
    config = asdict(planner.config)
    assert configure_planner_memo(env, {}) is env and env.planner is planner
    configure_planner_memo(env, {'planner_failure_memo': VERSION})
    assert isinstance(env.planner, ExactStaticFailureMemo) and asdict(env.planner.config) == config
    with pytest.raises(ValueError): configure_planner_memo(env, {'planner_failure_memo': True})


def test_cache_capacity_is_bounded_and_evicted_pose_is_searched_again():
    planner, world, start, goal = fixture()
    memo = ExactStaticFailureMemo(planner, max_entries=2)
    for dx in (0., .01, .02, 0.):
        memo.plan((start[0] + dx, *start[1:]), goal, world)
    assert memo.hits == 0 and memo.misses == 4 and len(memo._failures) == 2


def test_fixed_actions_keep_observations_rewards_feedback_and_physics_identical():
    # Synthetic plant equivalence only; real CARLA evidence is separate.
    from test_route_catalog_environment import RouteFixture, config
    settings = config()
    settings.planner.max_nodes = 1
    settings.planner.analytic_expansion = False
    ordinary, cached = RouteFixture(settings), RouteFixture(settings)
    try:
        ordinary.reset(seed=11)
        cached.reset(seed=11)
        configure_planner_memo(cached, {'planner_failure_memo': VERSION})
        for decision in range(4):
            left, right = ordinary.step(5), cached.step(5)
            np.testing.assert_array_equal(left[0], right[0])
            assert left[1:4] == right[1:4]
            assert asdict(ordinary.state) == asdict(cached.state)
            np.testing.assert_array_equal(ordinary.execution_outcome, cached.execution_outcome)
            assert left[4]['mode'] == right[4]['mode'] == 'plan_failed'
            assert [(row['state'], row['command']) for row in left[4]['trace']] == [
                (row['state'], row['command']) for row in right[4]['trace']]
            if decision:
                assert right[4]['plan']['cache_hit'] and right[4]['plan']['expanded_nodes'] == 0
                assert right[4]['plan']['cached_expanded_nodes'] == left[4]['plan']['expanded_nodes']
    finally:
        ordinary.close()
        cached.close()

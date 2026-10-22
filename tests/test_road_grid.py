import copy
import math
import numpy as np
import pytest
from navstack.native_route import sample_road_grid
from navstack.road_grid import RoadGrid
from navstack.world import World, body_corners, rectangles_overlap
from navstack.config import EnvConfig, VehicleConfig
from navstack.experiments import signature
from navstack.carla_bridge import CarlaNavigationEnv
from navstack.vehicle import VehiclePlant, VehicleState
from navstack.planning import GridAStarPlanner
from navstack.config import PlannerConfig


def road():
    return RoadGrid(sample_road_grid({"width": 32., "height": 32., "geometry_id": "fixture"},
        lambda x, y: 4 <= y <= 20 and not (14 <= x <= 15 and 10 <= y <= 11)))


def test_body_overlap_interior_hole_and_batched_scalar_agreement():
    grid = road()
    vehicle = VehicleConfig()
    world = World(32., 32., road=grid)
    # Hole is inside body, away from all four corners: corners alone miss it.
    pose = (13., 10.5, 0.)
    assert all(not grid.occupied(*point) for point in body_corners(pose, vehicle))
    assert world.out_of_bounds(pose, vehicle)
    blocked = grid.rectangles(np.array([[0, 0], [32, 32]]))
    rng = np.random.default_rng(75)
    poses = np.c_[rng.uniform(3, 26, 100), rng.uniform(5, 19, 100), rng.uniform(-math.pi, math.pi, 100)]
    expected = []
    for pose in poses:
        body = body_corners(pose, vehicle)
        hit = any(rectangles_overlap(body, np.array([[a,b],[c,b],[c,d],[a,d]])) for a,b,c,d in blocked)
        expected.append(not hit)
    assert world.free_many(poses, vehicle).tolist() == expected
    assert [world.free(p, vehicle) for p in poses] == expected
    assert grid.clearance(body_corners((5, 7, 0), vehicle)) > 0
    assert grid.clearance(body_corners((13, 10.5, 0), vehicle)) == 0
    bad = copy.deepcopy(grid.payload); bad["drivable"][0][0] = 1
    with pytest.raises(ValueError, match="hash"):
        RoadGrid(bad)


class FixtureNativeEnv(CarlaNavigationEnv):
    # Only isolates observation and terminal gates. Actual CARLA is run separately.
    def _create_episode(self, options=None):
        world = World(32., 32., road=road())
        return world, VehiclePlant(self.config.vehicle, VehicleState(5., 7.)), (25., 7., 0.), "fixture"


def test_shared_road_observation_signature_and_failure_before_success():
    observations = []
    for method in ("H", "B1", "B2", "B3"):
        config = EnvConfig(method=method, road_observation=True)
        env = FixtureNativeEnv(config, None)
        obs, _ = env.reset(seed=11)
        assert obs.shape == (4447,)
        assert signature(config)["observation_shape"] == [4447]
        np.testing.assert_array_equal(obs[-4096:], (~env.world.road.drivable).ravel())
        observations.append(obs)
        env.goal = (5., 7., 0.)
        env.ever_controlled = True
        env.episode["distance"] = 1.
        assert env._event() == "success"
        env.plant._state = VehicleState(5., 7., v=-1.)
        assert env._event() is None
        env.plant._state = VehicleState(13., 10.5)
        env.goal = (13., 10.5, 0.)
        assert env._event() == "out_of_bounds"
        env.close()
    for obs in observations[1:]:
        np.testing.assert_array_equal(obs, observations[0])
    assert signature(EnvConfig())["observation_shape"] == [351]


def test_b2_search_itself_respects_road_not_only_postprocessing():
    grid = RoadGrid(sample_road_grid({"width":32.,"height":32.,"geometry_id":"barrier"},
        lambda x,y: x < 14 or x > 17))
    assert grid.disc_intersects(13., 10., 1.)
    assert not grid.disc_intersects(12., 10., 1.)
    result = GridAStarPlanner(VehicleConfig(), PlannerConfig()).plan((5.,10.,0.), (25.,10.,0.), World(32.,32.,road=grid))
    assert result.status == "no_path"
    assert result.failure_reason == "grid_search_empty"

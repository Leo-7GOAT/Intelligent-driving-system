"""The scalar road checker remains the independent contact oracle."""
import math
import numpy as np
import pytest

from navstack.config import EnvConfig, VehicleConfig
from navstack.env import NavigationEnv
from navstack.experiments import signature
from navstack.native_route import sample_road_grid
from navstack.road_grid import RoadGrid,road_digest
from navstack.road_grid_batch import ExactRoadBatch, _body_corners_many
from navstack.world import World, body_corners
from navstack.vehicle import VehiclePlant, VehicleState


def grid():
    return RoadGrid(sample_road_grid({'width':32.,'height':28.,'geometry_id':'batch-fixture'},
        lambda x,y:4<y<23 and not (12<x<15 and 8<y<11)))


def oracle(grid,poses,vehicle,margin):
    return np.array([grid.intersects(body_corners(p,vehicle,margin)) for p in poses])


def test_batched_corner_coordinates_are_bit_identical_to_original_scalar_libm():
    rng = np.random.default_rng(20261031)
    poses = np.c_[rng.uniform(-1e6, 1e6, (10000, 2)), rng.uniform(-100*np.pi, 100*np.pi, 10000)]
    poses = np.r_[poses, [[0., -0., angle] for angle in (0., -0., np.pi/2, -np.pi/2, np.pi)]]
    for vehicle in (VehicleConfig(), VehicleConfig(front=3.811, rear=.9803, width=2.1634, wheelbase=3.0046)):
        for margin in (0., .1, .25):
            expected = np.asarray([body_corners(p, vehicle, margin) for p in poses])
            actual = _body_corners_many(poses, vehicle, margin)
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert _body_corners_many(np.empty((0, 3)), VehicleConfig()).shape == (0, 4, 2)


def test_batched_closed_cell_sat_matches_scalar_across_full_body_and_margins():
    road=grid();batch=ExactRoadBatch(road);rng=np.random.default_rng(438)
    poses=np.c_[rng.uniform(-4,36,1800),rng.uniform(-4,32,1800),rng.uniform(-4*math.pi,4*math.pi,1800)]
    for vehicle in (VehicleConfig(),VehicleConfig(front=3.811, rear=.9803,width=2.1634,wheelbase=3.0046)):
        for margin in (0.,.1,.25):
            np.testing.assert_array_equal(batch.intersects_many(poses,vehicle,margin),oracle(road,poses,vehicle,margin))
    assert batch.intersects_many(np.empty((0,3)),VehicleConfig()).shape==(0,)
    with pytest.raises(ValueError):batch.intersects_many([[1,2,float('nan')]],VehicleConfig())


def test_near_contact_uses_identical_scalar_tolerance_without_corner_only_shortcut():
    payload=sample_road_grid({'width':32.,'height':32.,'geometry_id':'contact'},lambda x,y:True)
    payload['drivable'][20][20]=0
    payload['sha256']=road_digest({k:v for k,v in payload.items() if k!='sha256'})
    road=RoadGrid(payload);vehicle=VehicleConfig();batch=ExactRoadBatch(road,chunk_size=16)
    poses=[]
    for epsilon in (0.,1e-14,-1e-14,1e-12,-1e-12,1e-10,-1e-10,1e-8,-1e-8,1e-6,-1e-6):
        poses.extend([(10-vehicle.front+epsilon,10.25,0.),
            (10.25,10-vehicle.width/2+epsilon,0.),(10.25,10-vehicle.front+epsilon,math.pi/2)])
    poses=np.array(poses)
    np.testing.assert_array_equal(batch.intersects_many(poses,vehicle),oracle(road,poses,vehicle,0.))
    assert batch.last_scalar_fallbacks>0
    interior=(9.5,10.25,0.)
    assert all(not road.occupied(*p) for p in body_corners(interior,vehicle))
    assert batch.intersects_many([interior],vehicle)[0]


def test_cached_cells_follow_public_raster_changes_and_free_road():
    payload=sample_road_grid({'width':32.,'height':32.,'geometry_id':'mutable'},lambda x,y:True)
    road=RoadGrid(payload);batch=ExactRoadBatch(road);vehicle=VehicleConfig();poses=[(9.5,10.25,0.)]
    assert not batch.intersects_many(poses,vehicle)[0]
    road.drivable[20,20]=False
    assert batch.intersects_many(poses,vehicle)[0]==oracle(road,poses,vehicle,0.)[0]


def test_world_backend_preserves_bounds_static_road_and_invalid_pose_decisions():
    scalar=World(32.,28.,[(6.,14.,8.,17.)],road=grid())
    batched=World(32.,28.,scalar.static,road=scalar.road,road_collision_backend='batch_sat_v1')
    rng=np.random.default_rng(24)
    poses=np.c_[rng.uniform(-4,36,1200),rng.uniform(-4,32,1200),rng.uniform(-math.pi,math.pi,1200)]
    poses=np.r_[poses,[[float('nan'),0.,0.],[float('inf'),0.,0.]]]
    for margin in (0.,.1,.25):
        np.testing.assert_array_equal(batched.free_many(poses,VehicleConfig(),margin),
                                      scalar.free_many(poses,VehicleConfig(),margin))
    replacement=RoadGrid(sample_road_grid({'width':32.,'height':28.,'geometry_id':'replacement'},lambda x,y:True))
    batched.road=scalar.road=replacement
    np.testing.assert_array_equal(batched.free_many(poses,VehicleConfig()),scalar.free_many(poses,VehicleConfig()))
    assert batched._road_batch.grid is replacement


class RoadFixture(NavigationEnv):
    """Real lightweight plant and MPC; not CARLA evidence."""
    def _create_episode(self, options=None):
        return World(32.,28.,road=grid()),VehiclePlant(self.config.vehicle,VehicleState(5.,17.)),(25.,17.,0.),'fixture'


def test_backend_is_explicit_versioned_and_shared_by_all_environment_methods():
    old=EnvConfig(road_observation=True)
    assert 'road_collision_backend' not in old.to_dict()
    assert 'road_collision_backend' not in signature(old)
    for method in ('H','B1','B2','B3'):
        config=EnvConfig(method=method,road_observation=True,road_collision_backend='batch_sat_v1')
        assert EnvConfig.from_dict(config.to_dict()).road_collision_backend=='batch_sat_v1'
        assert signature(config)['road_collision_backend']=='batch_sat_v1'
        env=RoadFixture(config)
        try:
            obs,_=env.reset(seed=11)
            assert env.world.road_collision_backend=='batch_sat_v1'
            assert obs.shape==(4447,)
        finally:env.close()
    with pytest.raises(ValueError,match='Unknown road collision'):EnvConfig(road_collision_backend='other')
    with pytest.raises(ValueError,match='shared measured-road'):EnvConfig(road_collision_backend='batch_sat_v1')


def test_backend_keeps_real_planning_mpc_commands_physics_and_rewards_equal():
    records=[]
    for backend in ('scalar','batch_sat_v1'):
        config=EnvConfig(road_observation=True,road_collision_backend=backend)
        config.planner.max_seconds=None
        config.planner.vectorized_primitives=True
        env=RoadFixture(config)
        try:
            obs,_=env.reset(seed=11)
            after,reward,terminated,truncated,info=env.step(5)
            assert info['executed_substeps']==10 and env.state.time==pytest.approx(.5)
            assert env.state.x>5
            records.append((obs,after,reward,terminated,truncated,env.state,env.plant.previous_command))
        finally:env.close()
    np.testing.assert_array_equal(records[0][0],records[1][0])
    np.testing.assert_array_equal(records[0][1],records[1][1])
    assert records[0][2:]==records[1][2:]

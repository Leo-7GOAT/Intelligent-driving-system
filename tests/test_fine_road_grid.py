"""Shared v4 body checks: remove cell inflation while retaining holes/bounds."""
from copy import deepcopy
import numpy as np
import pytest
from navstack.config import EnvConfig, VehicleConfig
from navstack.experiments import signature
from navstack.native_route import sample_road_grid, REFINED_GRID_VERSION
from navstack.road_grid import RoadGrid, road_digest
from navstack.world import World, body_corners, rectangles_overlap
from navstack.env import NavigationEnv


def test_resolution_removes_coarse_cell_false_positive_without_road_dilation():
    layout={'width':32.,'height':32.,'geometry_id':'strip'}
    query=lambda x,y:4.01<=y<=20.49
    old=sample_road_grid(layout,query,'carla_driving_indexed_union_64_v3')
    new=sample_road_grid(layout,query,REFINED_GRID_VERSION)
    assert old['drivable']==new['drivable']
    coarse,fine=RoadGrid(old),RoadGrid(new);vehicle=VehicleConfig()
    y=4.4+vehicle.width/2
    assert coarse.intersects(body_corners((10,y,0),vehicle))
    assert not fine.intersects(body_corners((10,y,0),vehicle))
    assert fine.intersects(body_corners((10,4+vehicle.width/2,0),vehicle))
    assert World(32,32,road=fine).out_of_bounds((-.1,y,0),vehicle)
    assert fine.clearance(body_corners((10,y,0),vehicle))>0


def test_fine_hole_inside_body_and_batch_matches_independent_rectangle_oracle():
    payload=sample_road_grid({'width':32.,'height':32.,'geometry_id':'hole'},
        lambda x,y:4<=y<=20 and not (11<=x<=11.125 and 6<=y<=6.125),REFINED_GRID_VERSION)
    grid=RoadGrid(payload);vehicle=VehicleConfig();pose=(10,6,0)
    assert all(not grid.occupied(*point) for point in body_corners(pose,vehicle))
    assert grid.intersects(body_corners(pose,vehicle))
    rng=np.random.default_rng(9421)
    poses=np.c_[rng.uniform(6,18,100),rng.uniform(5,19,100),rng.uniform(-3,3,100)]
    expected=[]
    for p in poses:
        body=body_corners(p,vehicle)
        rectangles=grid.rectangles(body)
        hit=any(rectangles_overlap(body,np.array([[a,b],[c,b],[c,d],[a,d]])) for a,b,c,d in rectangles)
        expected.append(not (np.any(body<0) or np.any(body>32) or hit))
    for backend in ('scalar','batch_sat_v1'):
        np.testing.assert_array_equal(World(32,32,road=grid,road_collision_backend=backend).free_many(poses,vehicle),expected)
    world=World(32,32,road=grid,road_collision_backend='batch_sat_v1')
    world.free_many([[20,10,0]],vehicle)
    grid.collision_drivable[:]=False
    assert not world.free_many([[20,10,0]],vehicle)[0]


def test_explicit_version_hash_and_same_policy_shape_for_every_method():
    payload=sample_road_grid({'width':32.,'height':32.,'geometry_id':'shape'},lambda x,y:True,REFINED_GRID_VERSION)
    for method in ('H','B1','B2','B3'):
        c=EnvConfig(method=method,road_observation=True,road_geometry_version=REFINED_GRID_VERSION)
        assert NavigationEnv(c).observation_space.shape==(4447,)
        assert signature(c)['road_geometry'].startswith(REFINED_GRID_VERSION)
    with pytest.raises(ValueError,match='shared road observation'):
        EnvConfig(road_geometry_version=REFINED_GRID_VERSION)
    corrupt=deepcopy(payload);corrupt['collision_drivable'][0][0]=0
    with pytest.raises(ValueError,match='hash'):RoadGrid(corrupt)
    for version in ('carla_driving_indexed_union_64_v3',REFINED_GRID_VERSION):
        corrupt=deepcopy(payload);corrupt['version']=version
        if version==REFINED_GRID_VERSION:corrupt['collision_drivable']=[[1]]
        corrupt.pop('sha256');corrupt['sha256']=road_digest(corrupt)
        with pytest.raises(ValueError,match='collision raster'):RoadGrid(corrupt)

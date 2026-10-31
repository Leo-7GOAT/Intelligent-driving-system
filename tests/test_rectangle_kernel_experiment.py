import math
import numpy as np
import pytest
from navstack.config import VehicleConfig
from navstack.native_route import sample_road_grid,REFINED_GRID_VERSION
from navstack.road_grid import RoadGrid
from navstack.road_grid_batch import ExactRoadBatch
from navstack.world import body_corners
from scripts.experimental_rectangle_kernel import occupied_partition,RectangleRoadBatch


def test_partition_exactly_tiles_occupied_cells_without_bridging_holes_or_gaps():
    rng=np.random.default_rng(20261031)
    masks=[np.ones((8,11),dtype=bool),np.zeros((8,11),dtype=bool),
        rng.random((17,23))>.5,np.indices((16,16)).sum(0)%2==0]
    for drivable in masks:
        coverage=np.zeros(drivable.shape,dtype=int)
        for left,low,right,high in occupied_partition(drivable):
            assert 0<=left<right<=drivable.shape[1] and 0<=low<high<=drivable.shape[0]
            coverage[low:high,left:right]+=1
        np.testing.assert_array_equal(coverage,(~drivable).astype(int))
    with pytest.raises(ValueError):occupied_partition(np.ones((2,2),dtype=int))


def test_fine_grid_full_body_flags_match_original_cells_and_scalar_contact_oracle():
    road=RoadGrid(sample_road_grid({'width':32.,'height':28.,'geometry_id':'rectangle-prototype'},
        lambda x,y:4<y<23 and not (12<x<15 and 8<y<11),version=REFINED_GRID_VERSION))
    original=ExactRoadBatch(road);candidate=RectangleRoadBatch(road)
    assert len(candidate.rectangles)<len(original.rectangles)/10
    rng=np.random.default_rng(619)
    poses=np.c_[rng.uniform(-4,36,1600),rng.uniform(-4,32,1600),rng.uniform(-4*math.pi,4*math.pi,1600)]
    for vehicle in (VehicleConfig(),VehicleConfig(front=3.811,rear=.9803,width=2.1634,wheelbase=3.0046)):
        contacts=np.array([[12-vehicle.front+delta,9.,0.] for delta in (-1e-8,-1e-12,0.,1e-12,1e-8)])
        all_poses=np.r_[poses,contacts]
        for margin in (0.,.1,.25):
            expected=original.intersects_many(all_poses,vehicle,margin)
            np.testing.assert_array_equal(candidate.intersects_many(all_poses,vehicle,margin),expected)
            oracle=np.array([road.intersects(body_corners(p,vehicle,margin)) for p in contacts])
            np.testing.assert_array_equal(candidate.intersects_many(contacts,vehicle,margin),oracle)


def test_candidate_refreshes_after_road_mask_mutation_and_keeps_empty_batch_shape():
    road=RoadGrid(sample_road_grid({'width':32.,'height':28.,'geometry_id':'rectangle-refresh'},lambda x,y:True))
    candidate=RectangleRoadBatch(road);vehicle=VehicleConfig()
    assert not candidate.intersects_many([[16.,14.,0.]],vehicle)[0]
    road.collision_drivable[:]=False
    assert candidate.intersects_many([[16.,14.,0.]],vehicle)[0]
    assert candidate.intersects_many(np.empty((0,3)),vehicle).shape==(0,)

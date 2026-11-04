import math
import numpy as np
import pytest
from navstack.config import VehicleConfig
from navstack.reference_road_refinement import footprint_points
from navstack.reference_point_constraints import allowed,local_points


def test_actual_float32_rejected_corner_is_blocked_after_world_local_roundtrip():
    v=VehicleConfig();layout=dict(angle=-89.6,origin=[115.8,-3.3],spawn_z=.6)
    pose=[20.,12.,.4];lat=-.05;turn=-.05
    shifted=[pose[0]-lat*math.sin(pose[2]),pose[1]+lat*math.cos(pose[2]),pose[2]+turn]
    from dataclasses import asdict
    point=np.float32(footprint_points(shifted,asdict(v),layout)[-1]).astype(float).tolist()
    points=local_points([point],layout)
    assert not allowed([pose],v,points)[0]
    assert allowed([[5.,5.,0.]],v,points)[0]
    assert local_points([point,point],layout)==points


def test_only_counterexamples_add_constraints_and_touching_counts_as_overlap():
    v=VehicleConfig();poses=np.array([[10.,10.,0.],[20.,20.,0.]])
    np.testing.assert_array_equal(allowed(poses,v,[]),[True,True])
    assert not allowed(poses,v,[(10+v.front,10.)])[0]
    np.testing.assert_array_equal(allowed(poses,v,[(100.,100.)]),[True,True])
    assert not allowed([[10.,10.,0.]],v,[(10.,10.+v.width/2+.04)])[0]


def test_invalid_or_unbounded_point_inputs_rejected():
    v=VehicleConfig()
    for points in [[[float('nan'),0]],[[0,0,0]],[[0,0]]*8193]:
        with pytest.raises(ValueError):allowed([[0,0,0]],v,points)


def test_private_point_view_and_exact_cache_invalidate_without_changing_physical_world():
    from navstack.world import World
    from navstack.config import PlannerConfig
    from navstack.planning import HybridAStar
    from navstack.planner_memo import ExactSteeringFailureMemo
    from navstack.reference_road_refinement import planning_view
    owner=World(30,30);vehicle=VehicleConfig()
    memo=ExactSteeringFailureMemo(HybridAStar(vehicle,PlannerConfig(max_seconds=None,max_nodes=200)),cache_success=True)
    def call(points):
        return memo.plan((6.,10.,0.),(10.,10.,0.),planning_view(owner,set(),points),initial_steering=0.,reference_speed=1.5)
    assert call({(100.,100.)}).status=='success'
    assert call({(100.,100.)}).cache_hit
    assert not getattr(call({(100.+1e-12,100.)}),'cache_hit',False)
    blocked=call({(12.,10.)})
    assert blocked.status!='success' and not getattr(blocked,'cache_hit',False)
    assert owner.free_many(np.array([[10.,10.,0.]]),vehicle)[0]
    assert not hasattr(owner,'_reference_point_constraints')


def test_registration_requires_frozen_joint_speed_and_known_model():
    from navstack.reference_point_constraints import settings,VERSION
    from navstack.steering_retiming import VERSION as JOINT
    p=dict(methods=['H'],evaluation_only=True,
        reference_road_refinement=dict(version='sampled_native_reference_refinement_v1',max_attempts=3),
        reference_steering_transition=dict(version='measured_steering_slew_32000_memo_v2',max_nodes=32000),
        reference_compute_profile=dict(version='exact_static_reference_compute_v3'),
        reference_joint_speed=dict(version=JOINT,spatial_speed_ceiling=.25),reference_point_constraints=dict(version=VERSION))
    assert settings(p)=={'version':VERSION} and settings({}) is None
    for patch in [dict(evaluation_only=False),dict(reference_joint_speed=None),dict(reference_point_constraints={'version':'unknown'})]:
        with pytest.raises(ValueError):settings({**p,**patch})

import numpy as np
from scipy.interpolate import CubicSpline
from navstack.config import PlannerConfig, VehicleConfig, EnvConfig
from navstack.experiments import signature
from navstack.grid_smoothing import certify_curve
from navstack.planning import GridAStarPlanner
from navstack.world import World


def test_swept_certificate_and_invalid_curvature():
    vehicle = VehicleConfig()
    curve = CubicSpline([0,10],[[4,10],[14,10]],bc_type=((1,[1,0]),(1,[1,0])))
    path,diagnostic = certify_curve(curve,vehicle,World(25,20))
    assert path is not None and diagnostic['curvature_upper_bound'] == 0
    # A thin rectangle must not vanish between samples, even with coarse intervals.
    blocked,_ = certify_curve(curve,vehicle,World(25,20,[(9.01,9.5,9.02,10.5)]),spacing=2)
    assert blocked is None
    cusp = CubicSpline([0,1,2],[[4,10],[5,10],[4,10]])
    assert certify_curve(cusp,vehicle,World(25,20))[0] is None


def test_b2_optimized_2d_path_and_checkpoint_version():
    vehicle = VehicleConfig()
    planner = GridAStarPlanner(vehicle,PlannerConfig(max_seconds=None,grid_postprocess='body_spline_v1'))
    world = World(32,24,[(14,8,16,12)])
    result = planner.plan((4,6,0),(25,6,0),world)
    assert result.status == 'success' and world.free_many(result.path,vehicle).all()
    np.testing.assert_allclose(result.path[[0,-1],:2],[[4,6],[25,6]])
    assert 'grid_postprocess' not in signature(EnvConfig())['planner']
    cfg = EnvConfig(planner=planner.config)
    assert signature(cfg)['planner']['grid_postprocess'] == 'body_spline_v1'

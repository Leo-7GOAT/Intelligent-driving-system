import math
import numpy as np
import pytest
from navstack.config import VehicleConfig
from navstack.planning import analytic_connection
from navstack.terminal_region import candidate_poses,connect_region
from navstack.world import World,body_corners


def test_all_candidates_stay_strictly_inside_original_position_yaw_gates():
    for yaw in (0.,-math.pi,math.pi,1.9):
        goal=(4.,7.,yaw);poses=candidate_poses(goal,.8,.35)
        assert len(poses)==27
        for x,y,angle in poses:
            assert math.hypot(x-goal[0],y-goal[1])<=.8/math.sqrt(2)+1e-14
            assert abs((angle-yaw+math.pi)%(2*math.pi)-math.pi)<=.175+1e-14
    for radius,angle in ((0.,.35),(.8,0.),(math.nan,.35),(.8,math.inf)):
        with pytest.raises(ValueError):candidate_poses((4.,7.,0.),radius,angle)


def test_feasible_exact_goal_path_is_preserved_without_extra_queries():
    vehicle=VehicleConfig();world=World(30,20)
    for start,goal in (((6,8,0),(12,8,0)),((15,8,math.pi),(9,8,math.pi)),((15,8,-math.pi),(9,8,-math.pi))):
        expected=analytic_connection(start,goal,world,vehicle)
        result=connect_region(start,goal,world,vehicle,.8,.35,analytic_connection)
        assert result['exact_goal'] and result['queries']==1
        np.testing.assert_array_equal(result['path'],expected)


def test_body_blocked_exact_pose_can_have_legal_interior_terminal_pose():
    vehicle=VehicleConfig();world=World(20,14)
    goal=(world.width-vehicle.front+.2,7.,0.);start=(goal[0]-3,7.,0.)
    assert world.free(start,vehicle) and not world.free(goal,vehicle)
    assert analytic_connection(start,goal,world,vehicle) is None
    result=connect_region(start,goal,world,vehicle,.8,.35,analytic_connection)
    assert result is not None and not result['exact_goal'] and result['queries']==27
    target=result['target'];assert world.free(target,vehicle)
    assert math.dist(target[:2],goal[:2])<=.8
    assert abs(target[2]-goal[2])<=.35
    assert world.free_many(result['path'],vehicle).all()
    assert np.all(body_corners(target,vehicle)[:,0]<=world.width)

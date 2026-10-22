import math
import pytest
from navstack.subgoals import goal_frame_candidates
from navstack.vehicle import VehicleState
from navstack.config import VehicleConfig


def test_goal_frame_candidates_are_pose_equivariant_and_can_target_exact_parking_pose():
    state = VehicleState(4, 12, yaw=.3)
    goal = (27, 12, 0)
    original = goal_frame_candidates(state, goal)
    assert len(original) == 31 and original[0] == (None, 0)
    rotated = goal_frame_candidates(VehicleState(-12, 4, yaw=.3+math.pi/2), (-12, 27, math.pi/2))
    for (a, speed), (b, other_speed) in zip(original[1:], rotated[1:]):
        assert b[:2] == pytest.approx((-a[1], a[0]))
        assert speed == other_speed
    near = goal_frame_candidates(VehicleState(25, 12, yaw=.15), goal)
    assert near[5][0] == goal
    assert state == VehicleState(4, 12, yaw=.3)


def test_terminal_candidates_never_request_lateral_goal_dead_ends():
    vehicle=VehicleConfig()
    goal=(27,12,0)
    candidates=goal_frame_candidates(VehicleState(25,12),goal,vehicle)
    assert all(pose==goal for pose,speed in candidates[1:])
    earlier=goal_frame_candidates(VehicleState(15,12),goal,vehicle)
    assert all(abs(pose[1]-12)<3 for pose,speed in earlier[1:])

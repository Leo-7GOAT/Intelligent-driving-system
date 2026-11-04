from dataclasses import asdict
import math
import numpy as np
import pytest
from navstack.config import VehicleConfig
from navstack.subgoals import route_frame_candidates
from navstack.vehicle import VehicleState


def test_route_candidates_follow_tangent_preserve_state_and_stop_at_route_end():
    angles=np.linspace(0,math.pi/3,31)
    route=np.c_[20*np.sin(angles),20*(1-np.cos(angles)),angles]
    state=VehicleState(0,0);before=asdict(state)
    candidates=route_frame_candidates(state,route,VehicleConfig())
    assert len(candidates)==31 and candidates[0]==(None,0.) and asdict(state)==before
    # Zero lateral offset at six metres uses local tangent, not final heading.
    assert 0.25<candidates[5][0][2]<0.35
    terminal=VehicleState(*route[-2,:2],yaw=route[-2,2])
    near=route_frame_candidates(terminal,route,VehicleConfig())
    for target,speed in near[1:]:
        np.testing.assert_allclose(target,route[-1],atol=1e-12)
        assert speed in (.75,1.5)


def test_route_candidates_rigid_transform_and_invalid_routes():
    route=np.array([[0,0,0],[10,0,0],[20,0,0]],float)
    angle=1.1;rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    offset=np.array([4,-7]);transformed=route.copy()
    transformed[:,:2]=route[:,:2]@rotation.T+offset;transformed[:,2]+=angle
    state=VehicleState(2,.3);xy=rotation@np.array([state.x,state.y])+offset
    a=route_frame_candidates(state,route,VehicleConfig())
    b=route_frame_candidates(VehicleState(*xy,yaw=angle),transformed,VehicleConfig())
    for (p,speed),(q,new_speed) in zip(a[1:],b[1:]):
        np.testing.assert_allclose(q[:2],rotation@np.array(p[:2])+offset,atol=1e-10)
        assert abs(q[2]-p[2]-angle)<1e-10 and speed==new_speed
    with pytest.raises(ValueError):route_frame_candidates(state,[[0,0,0],[0,0,0]],VehicleConfig())

import math
import numpy as np
import pytest
from navstack.config import VehicleConfig
from navstack.vehicle import VehicleState,Command
from navstack.trajectory import Trajectory
from navstack.mpc import MPC


def equilibrium_reference(delta):
    vehicle=VehicleConfig();dt=.05;n=20;speed=1.
    curvature=math.tan(delta)/vehicle.wheelbase;yaw=np.arange(n+1)*dt*speed*curvature
    x=np.r_[10.,10.+np.cumsum(dt*speed*np.cos(yaw[:-1]))]
    y=np.r_[10.,10.+np.cumsum(dt*speed*np.sin(yaw[:-1]))]
    ref=Trajectory(x,y,yaw,np.arange(n+1)*dt*speed,np.full(n+1,curvature),
        np.full(n+1,speed),np.zeros(n+1),np.arange(n+1)*dt,0.)
    state=VehicleState(10.,10.,0.,speed,delta)
    return vehicle,state,ref,Command(0.,delta)


def test_known_constant_turn_equilibrium_has_no_zero_steer_effort_bias():
    vehicle,state,ref,previous=equilibrium_reference(.3)
    old=MPC(vehicle).step(state,ref,previous)
    new=MPC(vehicle,reference_centered_effort=True).step(state,ref,previous)
    assert old.status==new.status=='solved'
    assert abs(new.command.steering-.3)<2e-4
    assert abs(new.command.steering-.3)<abs(old.command.steering-.3)*.1
    assert abs(new.command.acceleration)<2e-4
    assert new.diagnostics['constraint_violation']<.002


def test_straight_equilibrium_and_input_limits_are_unchanged():
    vehicle,state,ref,previous=equilibrium_reference(0.)
    a=MPC(vehicle).step(state,ref,previous)
    b=MPC(vehicle,reference_centered_effort=True).step(state,ref,previous)
    assert a.status==b.status=='solved'
    assert b.command.steering==pytest.approx(a.command.steering,abs=1e-10)
    assert b.command.acceleration==pytest.approx(a.command.acceleration,abs=1e-10)
    assert -vehicle.max_decel<=b.command.acceleration<=vehicle.max_accel

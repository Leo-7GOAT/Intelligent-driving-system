from dataclasses import asdict
import numpy as np
import pytest
from navstack.config import VehicleConfig
from navstack.pursuit_controller import TimedPursuitPI
from navstack.trajectory import parameterize
from navstack.vehicle import Command,VehicleState,VehiclePlant


def test_timed_pursuit_issues_bounded_commands_without_mutating_state_or_reference():
    v=VehicleConfig();reference=parameterize([[4,8,0],[16,8,0]],1.,0.,v,stop=True)
    snapshots={k:getattr(reference,k).copy() for k in ('x','y','yaw','v','a','s','t')}
    state=VehicleState(4,8,yaw=.1);controller=TimedPursuitPI(v)
    result=controller.step(state,reference,Command())
    assert result.status=='solved' and -v.max_decel<=result.command.acceleration<=v.max_accel
    assert abs(result.command.steering)<=v.steer_rate*.05 and result.command.steering<0
    assert state==VehicleState(4,8,yaw=.1)
    assert result.diagnostics['qp_applicable'] is False
    for key,value in snapshots.items():np.testing.assert_array_equal(getattr(reference,key),value)


def test_actual_physical_integrator_straight_reference_reaches_stopped_goal_without_teleport():
    v=VehicleConfig();reference=parameterize([[4,8,0],[16,8,0]],1.,0.,v,stop=True)
    plant=VehiclePlant(v,VehicleState(4,8));controller=TimedPursuitPI(v)
    for _ in range(600):
        before=plant.state;result=controller.step(before,reference,plant.previous_command)
        state=plant.step(result.command,.05)
        assert state.time==pytest.approx(before.time+.05)
        assert abs(state.x-before.x)<=v.max_speed*.05+1e-6
        if abs(state.x-16)<.8 and state.v<.25:break
    assert abs(state.x-16)<.8 and state.v<.25 and abs(state.y-8)<1e-6


def test_no_reference_still_returns_the_common_physical_brake_and_invalid_gains_reject():
    v=VehicleConfig();assert TimedPursuitPI(v).step(VehicleState(4,8),None,Command()).status=='failed'
    with pytest.raises(ValueError):TimedPursuitPI(v,speed_kp=-1)
    with pytest.raises(ValueError):TimedPursuitPI(v,unknown_gain=1)

import numpy as np
import pytest

from navstack.carla_actuation import AccelerationPI, longitudinal_control


def test_legacy_mapping_exact_and_coast_preserves_urgent_braking():
    gains = {"speed_kp": .35, "brake_decel": 8.}
    for speed in np.linspace(0., 3.5, 8):
        for target in (0., .02, .1, 1., 3.):
            for acceleration in (-2.5, -1., -.1, 0., 1.5):
                effort = .12 + gains["speed_kp"]*(target-speed) + .06*acceleration
                if acceleration < -.05:
                    throttle, brake = 0., min(1., -acceleration/gains["brake_decel"]+max(0., speed-target)*.1)
                else:
                    throttle, brake = max(0., min(.6, effort)), 0.
                if target < .03 and acceleration <= 0:
                    throttle, brake = 0., max(brake, .3)
                if speed >= 3.:
                    throttle, brake = 0., max(brake, min(.4, (speed-3)*.3))
                assert longitudinal_control(speed, target, acceleration, gains)[:2] == (throttle, brake)
                revised = longitudinal_control(speed, target, acceleration, dict(gains, longitudinal_mode="coast_mild_v1"))
                if acceleration < -1. or target <= .1 or speed >= 2. or speed <= 0.:
                    assert revised[:2] == (throttle, brake)
    assert longitudinal_control(1., 1., -.5, dict(gains, longitudinal_mode="coast_mild_v1")) == (0., 0., "mild_deceleration_coast")
    with pytest.raises(ValueError, match="version"):
        longitudinal_control(1., 1., -.5, dict(gains, longitudinal_mode="typo"))


def test_acceleration_feedback_reduces_throttle_before_braking_and_bounds_state():
    gains = {"acceleration_integral_gain": .15, "acceleration_filter_seconds": .2, "brake_decel": 6.25}
    controller = AccelerationPI(gains)
    for _ in range(20):
        throttle, brake, _ = controller.step(.5, 1., 1., 0., gains)
    previous = throttle
    throttle, brake, _ = controller.step(.5, 1., -.3, 0., gains)
    assert 0 < throttle < previous and brake == 0
    for _ in range(1000):
        controller.step(.5, 1., 1.5, -1., gains)
    assert controller.state()["signed_pedal"] == .6
    emergency = controller.step(1., .8, -2.5, 0., gains)
    assert emergency[0] == 0. and emergency[1] == pytest.approx(.42) and emergency[2] == "emergency_brake"
    assert controller.state()["signed_pedal"] == 0
    assert controller.step(.01, 0., -.1, 0., gains) == (0., .3, "final_hold")
    assert AccelerationPI(gains).state() == {"signed_pedal": 0., "filtered_acceleration": 0.}
    with pytest.raises(ValueError, match="Finite"):
        controller.step(1., 1., 0., float("nan"), gains)


def test_measured_feedforward_offsets_coast_drag_and_exports_integral():
    gains = {"acceleration_integral_gain": .15, "acceleration_filter_seconds": .2, "brake_decel": 6.25,
             "throttle_model": {"throttle_gain": 7., "speed_coefficient": -.08, "intercept": -.65}}
    controller = AccelerationPI(gains)
    throttle, brake, mode = controller.step(1., 1., -.3, 0., gains)
    assert throttle > 0 and brake == 0 and mode == "acceleration_ff_v2"
    assert "feedback_integral" in controller.state()
    emergency = controller.step(1., .8, -2.5, 0., gains)
    assert emergency[1] == pytest.approx(.42) and emergency[2] == "emergency_brake"
    assert controller.state()["feedback_integral"] == 0.
    with pytest.raises(ValueError, match="model"):
        AccelerationPI(dict(gains, throttle_model={"throttle_gain": -1., "speed_coefficient": 0., "intercept": 0.}))

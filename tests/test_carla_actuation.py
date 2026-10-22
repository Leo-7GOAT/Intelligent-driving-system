import numpy as np
import pytest

from navstack.carla_actuation import longitudinal_control


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

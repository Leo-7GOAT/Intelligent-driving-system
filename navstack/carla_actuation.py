"""Versioned CARLA longitudinal mapping; compatible with the CPython 3.7 owner."""


def longitudinal_control(speed, target, acceleration, gains):
    """Return throttle, brake and diagnostic mode; never modify measured state.

    The optional coast branch addresses measured low-speed brake discontinuities.
    It does not suppress strong deceleration, overspeed braking or final hold.
    This is an actuator experiment, not a safety guarantee.
    """
    mode = gains.get("longitudinal_mode", "legacy")
    if mode not in ("legacy", "coast_mild_v1"):
        raise ValueError("Unknown longitudinal adapter version")
    effort = .12 + gains["speed_kp"] * (target-speed) + .06*acceleration
    reason = "legacy"
    if acceleration < -.05:
        throttle = 0.
        brake = min(1., -acceleration/gains["brake_decel"] + max(0., speed-target)*.1)
        if mode == "coast_mild_v1" and 0. < speed < 2. and acceleration >= -1. and target > .1:
            brake = 0.
            reason = "mild_deceleration_coast"
    else:
        throttle, brake = max(0., min(.6, effort)), 0.
    if target < .03 and acceleration <= 0:
        throttle, brake, reason = 0., max(brake, .3), "final_hold"
    if speed >= 3.:
        throttle, brake, reason = 0., max(brake, min(.4, (speed-3)*.3)), "overspeed"
    return throttle, brake, reason

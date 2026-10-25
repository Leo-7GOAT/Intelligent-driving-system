"""Versioned CARLA longitudinal mapping; compatible with the CPython 3.7 owner."""

import math


ACTUATOR_STATE_FIELDS = ("signed_pedal", "filtered_acceleration", "feedback_integral")


def actuator_state_values(evidence):
    """Four measured-frame adapter inputs; never synthesize missing state."""
    if evidence.get("actuator_observation") != "ff_state_v1":
        raise ValueError("Missing ff_state_v1 actuator observation")
    state = evidence.get("actuation_state")
    if not isinstance(state, dict) or set(state) != set(ACTUATOR_STATE_FIELDS):
        raise ValueError("Missing complete feedforward actuator state")
    values = [evidence.get("speed_setpoint")] + [state[k] for k in ACTUATOR_STATE_FIELDS]
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        raise ValueError("Actuator state must be finite measured-frame values")
    if not 0 <= values[0] <= 3 or not -1 <= values[1] <= .6 or not -.15 <= values[3] <= .15:
        raise ValueError("Actuator state exceeds registered adapter bounds")
    return values


def validate_actuator_interface(gains, observation, road_observation, scenario, diagnostic=False):
    """Explicit opt-in before actors or ticks; fixed-reference diagnostics remain supported."""
    mode = gains.get("longitudinal_mode", "legacy")
    if observation not in ("off", "ff_state_v1"):
        raise ValueError("Unknown actuator observation version")
    if observation == "ff_state_v1":
        if (not road_observation or scenario != "native_curve" or mode != "acceleration_ff_v2"
                or gains.get("drivetrain_mode") != "manual_first_v1" or not gains.get("throttle_model")):
            raise ValueError("ff_state_v1 requires native road and measured feedforward adapter")
    elif mode in ("acceleration_pi_v1", "acceleration_ff_v2"):
        if road_observation:
            raise ValueError("Measured-road PPO requires explicit actuator internal state")
        if scenario != "native_curve" and not diagnostic:
            raise ValueError("Acceleration feedback requires explicit diagnostic scope or shared state")


class AccelerationPI:
    """Incremental signed-pedal feedback; actual CARLA acceleration is the input.

    Positive pedal means throttle, negative means brake. Thus a mildly negative
    acceleration request can reduce an existing throttle without immediately
    applying brake. The versioned ff_state_v1 training interface exposes the
    filter, integral, signed pedal and speed setpoint to every architecture.
    """
    def __init__(self, gains):
        self.ki = float(gains["acceleration_integral_gain"])
        self.tau = float(gains["acceleration_filter_seconds"])
        if not math.isfinite(self.ki) or not math.isfinite(self.tau) or self.ki <= 0 or self.tau <= 0:
            raise ValueError("Invalid acceleration feedback gains")
        self.pedal = 0.
        self.filtered_acceleration = 0.
        self.integral = 0.
        self.model = gains.get("throttle_model")
        if self.model is not None and (not all(math.isfinite(v) for v in self.model.values())
                                       or self.model["throttle_gain"] <= 0):
            raise ValueError("Invalid measured throttle model")

    def state(self):
        result = {"signed_pedal": self.pedal, "filtered_acceleration": self.filtered_acceleration}
        if self.model is not None:
            result["feedback_integral"] = self.integral
        return result

    def step(self, speed, target, requested, measured, gains, dt=.05):
        if not all(math.isfinite(v) for v in (speed, target, requested, measured, dt)) or dt <= 0:
            raise ValueError("Finite measured inputs and positive dt required")
        self.filtered_acceleration += dt/(self.tau+dt)*(measured-self.filtered_acceleration)
        # Saturation is applied to the integrator itself: no hidden wind-up.
        if self.model is None:
            self.pedal = max(-1., min(.6, self.pedal+self.ki*dt*(requested-self.filtered_acceleration)))
        else:
            coast = self.model["speed_coefficient"]*speed+self.model["intercept"]
            difference = requested-coast
            feedforward = difference/(self.model["throttle_gain"] if difference >= 0 else gains["brake_decel"])
            # The fitted model is throttle-only; negative effort still uses the
            # old measured brake gain, whose low-speed limitations are reported.
            self.integral = max(-.15, min(.15, self.integral+self.ki*dt*(requested-self.filtered_acceleration)))
            self.pedal = max(-1., min(.6, feedforward+self.integral))
        if requested <= -2.4:
            self.pedal = self.integral = 0.
            return 0., min(1., -requested/gains["brake_decel"]+max(0., speed-target)*.1), "emergency_brake"
        if abs(speed) < .1 and target < .03 and requested <= 0:
            self.pedal = self.integral = 0.
            return 0., .3, "final_hold"
        if speed >= 3.:
            self.pedal = min(self.pedal, 0.)
            return 0., max(-self.pedal, min(.4, (speed-3)*.3)), "overspeed"
        return max(0., self.pedal), max(0., -self.pedal), "acceleration_ff_v2" if self.model is not None else "acceleration_pi_v1"


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

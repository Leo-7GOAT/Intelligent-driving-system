from dataclasses import dataclass
import math
import numpy as np
from .vehicle import wrap


@dataclass
class Trajectory:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    s: np.ndarray
    kappa: np.ndarray
    v: np.ndarray
    a: np.ndarray
    t: np.ndarray
    start_time: float

    def sample(self, times):
        times = np.asarray(times) - self.start_time
        return np.column_stack(
            [
                np.interp(times, self.t, values)
                for values in (self.x, self.y, self.v, self.yaw)
            ]
        )


def parameterize(
    path, target_speed, initial_speed, vehicle, start_time=0.0, stop=False, spacing=0.12
):
    if path is None or len(path) < 2:
        raise ValueError("no valid spatial path")
    path = np.asarray(path, float)
    raw_s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    keep = np.r_[True, np.diff(raw_s) > 1e-8]
    path, raw_s = path[keep], raw_s[keep]
    if len(path) < 2:
        raise ValueError("zero_length_path")
    s = np.linspace(0, raw_s[-1], max(3, math.ceil(raw_s[-1] / spacing) + 1))
    x, y, yaw = (
        np.interp(s, raw_s, a) for a in (path[:, 0], path[:, 1], np.unwrap(path[:, 2]))
    )
    kappa = np.gradient(yaw, s)
    if (
        not np.isfinite(kappa).all()
        or np.max(abs(kappa)) > 1 / vehicle.min_turn_radius + 0.015
    ):
        raise ValueError("curvature_limit")
    v = np.minimum(
        min(target_speed, vehicle.max_speed),
        np.sqrt(1.5 / np.maximum(abs(kappa), 1e-8)),
    )
    # Initial measured speed must be preserved even if a lower speed was requested.
    v[0] = initial_speed
    for i in range(1, len(s)):
        ds = s[i] - s[i - 1]
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * vehicle.max_accel * ds))
        v[i] = max(v[i], math.sqrt(max(0, v[i - 1] ** 2 - 2 * vehicle.max_decel * ds)))
    if stop:
        v[-1] = 0.0
        for i in range(len(s) - 2, 0, -1):
            v[i] = min(
                v[i],
                math.sqrt(v[i + 1] ** 2 + 2 * vehicle.max_decel * (s[i + 1] - s[i])),
            )
        if v[0] ** 2 > v[1] ** 2 + 2 * vehicle.max_decel * (s[1] - s[0]) + 1e-6:
            raise ValueError("insufficient_stopping_distance")
    denom = v[:-1] + v[1:]
    if np.any(denom < 1e-8):
        raise ValueError("zero_speed_segment")
    t = np.r_[0, np.cumsum(2 * np.diff(s) / denom)]
    a = np.r_[np.diff(v) / np.diff(t), 0]
    return Trajectory(x, y, yaw, s, kappa, v, a, t, start_time)


def dynamic_safe(reference, world, vehicle, horizon, dt=0.05):
    end = min(horizon, reference.t[-1])
    times = np.linspace(0, end, max(2, math.ceil(end / dt) + 1))
    poses = reference.sample(reference.start_time + times)
    return all(
        world.clearance((p[0], p[1], p[3]), vehicle, o, float(t)) > 0.12
        for t, p in zip(times, poses)
        for o in world.dynamic
    )


def tracking_error(state, reference):
    """Project onto segments, then interpolate unwrapped yaw and speed by arc fraction."""
    if reference is None or len(reference.x) < 2:
        return {
            "lateral": None,
            "heading": None,
            "speed": None,
            "missing_reason": "no_valid_reference",
        }
    pts = np.column_stack([reference.x, reference.y])
    d = np.diff(pts, axis=0)
    fractions = np.clip(
        np.sum((np.array([state.x, state.y]) - pts[:-1]) * d, axis=1)
        / np.maximum(np.sum(d * d, axis=1), 1e-12),
        0,
        1,
    )
    projected = pts[:-1] + fractions[:, None] * d
    distance = np.linalg.norm(projected - [state.x, state.y], axis=1)
    i = int(np.argmin(distance))
    f = fractions[i]
    yaw = (1 - f) * reference.yaw[i] + f * reference.yaw[i + 1]
    # Velocity tracking uses the time assigned to this execution instant.
    desired_v = float(reference.sample([state.time])[0, 2])
    return {
        "lateral": float(distance[i]),
        "heading": float(wrap(state.yaw - yaw)),
        "speed": float(state.v - desired_v),
        "missing_reason": None,
    }

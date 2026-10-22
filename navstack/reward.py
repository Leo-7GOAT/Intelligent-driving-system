"""Shared physical-time reward; shaping potential never depends on local subgoals."""

import math
import numpy as np


class RewardCalculator:
    def __init__(self, config, dt):
        self.config, self.dt = config, dt
        self.gamma = config.gamma_per_second**dt

    def potential(self, state, goal):
        # Fixed 40 m default normalization, finite even for unreachable static layouts.
        return -min(
            math.hypot(state.x - goal[0], state.y - goal[1])
            / self.config.distance_scale,
            2.0,
        )

    def calculate(self, before, after, goal, risk, event=None, terminated=False):
        c, dt = self.config, self.dt
        jerk = (after.acceleration - before.acceleration) / dt
        # Actual lateral acceleration derived from yaw increment and average speed.
        angle = (after.yaw - before.yaw + math.pi) % (2 * math.pi) - math.pi
        lateral = (before.v + after.v) / 2 * angle / dt
        steer_rate = (after.steering - before.steering) / dt
        comfort = float(
            np.mean(
                np.clip(
                    np.array([abs(jerk) / 5, abs(lateral) / 2, abs(steer_rate) / 0.6]),
                    0,
                    1,
                )
            )
        )
        parts = {
            "event": float(getattr(c, event)) if event else 0.0,
            "time": -dt * c.time_weight,
            "risk": -dt * c.risk_weight * float(np.clip(risk, 0, 1)),
            "comfort": -dt * c.comfort_weight * comfort,
            "potential": c.beta
            * (
                self.gamma * (0 if terminated else self.potential(after, goal))
                - self.potential(before, goal)
            ),
        }
        if c.progress_weight:
            # Unlike terminal-zero PBRS, this intentionally rewards the measured
            # remaining-distance reduction on failed finite-horizon episodes too.
            # Shared across architectures; never uses a local target or planner.
            parts["task_progress"] = c.progress_weight * (
                math.hypot(before.x - goal[0], before.y - goal[1])
                - math.hypot(after.x - goal[0], after.y - goal[1])
            )
        return (
            sum(parts.values()),
            parts,
            {
                "jerk": jerk,
                "lateral_acceleration": lateral,
                "steer_rate": steer_rate,
                "risk": risk,
            },
        )

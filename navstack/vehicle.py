"""The plant alone advances physical state. Prediction never receives a mutable plant."""

from collections import deque
from dataclasses import dataclass
import math
import numpy as np
from .config import VehicleConfig


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Command:
    acceleration: float = 0.0
    steering: float = 0.0


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw: float = 0.0
    v: float = 0.0
    steering: float = 0.0
    acceleration: float = 0.0
    time: float = 0.0


class VehiclePlant:
    def __init__(self, config: VehicleConfig, initial: VehicleState):
        self.config = config
        self._state = initial
        self.previous_command = Command()
        self.delay_queue = deque([Command()] * config.input_delay_steps)

    @property
    def state(self):
        return self._state

    def step(self, command: Command, dt: float):
        c, s = self.config, self.state
        if dt <= 0 or not np.isfinite([command.acceleration, command.steering]).all():
            raise ValueError("finite control and positive dt required")
        # Rate-limit the issued command, then model delay and steering lag.
        steer = float(
            np.clip(
                command.steering,
                self.previous_command.steering - c.steer_rate * dt,
                self.previous_command.steering + c.steer_rate * dt,
            )
        )
        issued = Command(
            float(np.clip(command.acceleration, -c.max_decel, c.max_accel)),
            float(np.clip(steer, -c.max_steer, c.max_steer)),
        )
        self.previous_command = issued
        self.delay_queue.append(issued)
        applied = self.delay_queue.popleft()
        fraction = 1.0 if c.steer_tau <= 0 else 1 - math.exp(-dt / c.steer_tau)
        delta = s.steering + np.clip(
            fraction * (applied.steering - s.steering),
            -c.steer_rate * dt,
            c.steer_rate * dt,
        )
        v = float(
            np.clip(
                s.v + applied.acceleration * c.acceleration_scale * dt, 0, c.max_speed
            )
        )
        actual_a = (v - s.v) / dt
        vmid = (v + s.v) / 2
        dyaw = vmid / (c.wheelbase * c.wheelbase_scale) * math.tan(delta) * dt
        self._state = VehicleState(
            s.x + vmid * math.cos(s.yaw + dyaw / 2) * dt,
            s.y + vmid * math.sin(s.yaw + dyaw / 2) * dt,
            wrap(s.yaw + dyaw),
            v,
            float(delta),
            actual_a,
            s.time + dt,
        )
        return self.state


def brake(state, config):
    return Command(-config.max_decel, state.steering)

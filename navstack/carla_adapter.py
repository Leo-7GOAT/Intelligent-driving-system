"""Conditional CARLA plant adapter. Requires a compatible, already running server.

No installation/download, no autopilot, no set_transform, only owned actor cleanup.
This module is unverified unless conditional tests actually run against CARLA.
"""

from dataclasses import dataclass
import importlib.util
import math
import socket
from .vehicle import VehicleState, Command


@dataclass
class CarlaConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    dt: float = 0.05
    timeout: float = 3.0
    blueprint: str = "vehicle.tesla.model3"
    spawn_index: int = 0
    throttle_accel: float = 3.0
    brake_decel: float = 6.0
    acceleration_kp: float = 0.15


def probe(host="127.0.0.1", port=2000):
    from pathlib import Path
    import subprocess, tempfile, json

    root = Path(__file__).resolve().parents[1]
    isolated = root / ".carla-python37/runtime/python.exe"
    if importlib.util.find_spec("carla") is None and isolated.exists():
        with tempfile.TemporaryDirectory(prefix="carla_probe_") as directory:
            output = Path(directory) / "connection.json"
            result = subprocess.run([str(isolated), str(root / "scripts/probe_carla37.py"),
                                     "--host", host, "--port", str(port), "--output", str(output)],
                                    capture_output=True, timeout=25,
                                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            if output.exists():
                report = json.loads(output.read_text(encoding="utf-8"))
                report.update({"client_installed":True,"client_environment":"isolated_cp37",
                               "server_reachable":report.get("status")=="CONNECTED",
                               "status":"AVAILABLE" if report.get("status")=="CONNECTED" else "BLOCKED",
                               "closed_loop_verified":False,
                               "reason":"Read-only connection probe; closed-loop evidence is recorded separately"})
                return report
    result = {
        "host": host,
        "port": port,
        "client_installed": importlib.util.find_spec("carla") is not None,
        "server_reachable": False,
        "status": "BLOCKED",
        "closed_loop_verified": False,
    }
    try:
        with socket.create_connection((host, port), timeout=0.5):
            result["server_reachable"] = True
    except OSError as exc:
        result["connection_reason"] = str(exc)
    if not result["client_installed"]:
        result["reason"] = "CARLA Python client not installed in project environment"
        return result
    if not result["server_reachable"]:
        result["reason"] = "CARLA server not reachable"
        return result
    import carla

    try:
        client = carla.Client(host, port)
        client.set_timeout(3.0)
        result.update(
            {
                "client_version": client.get_client_version(),
                "server_version": client.get_server_version(),
                "map": client.get_world().get_map().name,
            }
        )
        result["status"] = (
            "AVAILABLE"
            if result["client_version"] == result["server_version"]
            else "BLOCKED"
        )
        result["reason"] = (
            "interface integration still requires execution"
            if result["status"] == "AVAILABLE"
            else "client/server version mismatch"
        )
    except Exception as exc:
        result["reason"] = str(exc)
    return result


class CarlaPlant:
    def __init__(self, config=None):
        import carla

        self.carla = carla
        self.config = config or CarlaConfig()
        c = self.config
        self.client = carla.Client(c.host, c.port)
        self.client.set_timeout(c.timeout)
        if self.client.get_client_version() != self.client.get_server_version():
            raise RuntimeError("CARLA versions differ")
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.owned = []
        self.actor = None
        self.collision_events = []
        self.previous_command = Command()
        self.frame = None
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = c.dt
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
            self.world.apply_settings(settings)
            blueprint = self.world.get_blueprint_library().find(c.blueprint)
            blueprint.set_attribute("role_name", "codex_hierarchical_ppo_test")
            spawn = self.world.get_map().get_spawn_points()[c.spawn_index]
            self.actor = self.world.try_spawn_actor(blueprint, spawn)
            if self.actor is None:
                raise RuntimeError("spawn occupied; no actor was moved")
            self.owned.append(self.actor)
            self.actor.set_autopilot(False)
            wheels = self.actor.get_physics_control().wheels
            # WheelPhysicsControl positions are WORLD coordinates (centimeters in
            # CARLA's vehicle physics implementation), not actor-local axle offsets.
            tr = self.actor.get_transform()
            forward = tr.get_forward_vector()
            local_x = [
                (w.position.x / 100 - tr.location.x) * forward.x
                + (w.position.y / 100 - tr.location.y) * forward.y
                + (w.position.z / 100 - tr.location.z) * forward.z
                for w in wheels
            ]
            ordered = sorted(local_x)
            front_x = sum(ordered[-2:]) / 2
            self.rear_x = sum(ordered[:2]) / 2
            self.wheelbase = front_x - self.rear_x
            self.max_steer = math.radians(max(w.max_steer_angle for w in wheels))
            if (
                not 1 < self.wheelbase < 5
                or not -4 < self.rear_x < 1
                or self.max_steer <= 0
            ):
                raise RuntimeError(
                    "CARLA wheel geometry/unit mismatch; check version before running"
                )
            box = self.actor.bounding_box
            self.geometry = {
                "wheelbase": self.wheelbase,
                "front": box.location.x + box.extent.x - self.rear_x,
                "rear": self.rear_x - (box.location.x - box.extent.x),
                "width": 2 * box.extent.y,
                "max_steer": self.max_steer,
            }
            sensor = self.world.spawn_actor(
                self.world.get_blueprint_library().find("sensor.other.collision"),
                carla.Transform(),
                attach_to=self.actor,
            )
            self.owned.append(sensor)
            sensor.listen(
                lambda event: self.collision_events.append(
                    {
                        "frame": event.frame,
                        "timestamp": event.timestamp,
                        "other_actor": event.other_actor.id,
                    }
                )
            )
            self.world.tick()
            self._read()
        except Exception:
            self.close()
            raise

    def _read(self):
        transform = self.actor.get_transform()
        velocity = self.actor.get_velocity()
        acceleration = self.actor.get_acceleration()
        snapshot = self.world.get_snapshot()
        self.frame = snapshot.frame
        carla_yaw = math.radians(transform.rotation.yaw)
        co, si = math.cos(carla_yaw), math.sin(carla_yaw)
        # Right-handed x/y, CCW radians, rear axle reference; CARLA y and yaw reversed.
        x = transform.location.x + self.rear_x * co
        y = -(transform.location.y + self.rear_x * si)
        left = math.radians(
            self.actor.get_wheel_steer_angle(self.carla.VehicleWheelLocation.FL_Wheel)
        )
        right = math.radians(
            self.actor.get_wheel_steer_angle(self.carla.VehicleWheelLocation.FR_Wheel)
        )
        actual_steering = -math.atan((math.tan(left) + math.tan(right)) / 2)
        self._state = VehicleState(
            x,
            y,
            -carla_yaw,
            velocity.x * co + velocity.y * si,
            actual_steering,
            acceleration.x * co + acceleration.y * si,
            snapshot.timestamp.elapsed_seconds,
        )
        return self._state

    @property
    def state(self):
        return self._state

    def step(self, command, dt):
        if not math.isclose(dt, self.config.dt):
            raise ValueError("CARLA dt must match synchronous fixed delta")
        c = self.config
        # Explicit uncalibrated feedback adapter; throttle is not acceleration.
        target = command.acceleration + c.acceleration_kp * (
            command.acceleration - self.state.acceleration
        )
        throttle = min(1, max(0, target / c.throttle_accel))
        braking = min(1, max(0, -target / c.brake_decel))
        delta = max(
            self.previous_command.steering - 0.6 * dt,
            min(self.previous_command.steering + 0.6 * dt, command.steering),
        )
        self.actor.apply_control(
            self.carla.VehicleControl(
                throttle=throttle,
                brake=braking,
                steer=max(-1, min(1, -delta / self.max_steer)),
            )
        )
        self.previous_command = Command(command.acceleration, delta)
        self.world.tick()
        return self._read()

    def close(self):
        errors = []
        try:
            for actor in reversed(self.owned):
                try:
                    if hasattr(actor, "stop"):
                        actor.stop()
                    if actor.is_alive:
                        actor.destroy()
                except Exception as exc:
                    errors.append(str(exc))
        finally:
            self.owned = []
            self.world.apply_settings(self.original_settings)
        self.cleanup_errors = errors

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

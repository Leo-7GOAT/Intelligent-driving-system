import hashlib
import json
import math
import pytest

from navstack.native_route import local_layout
from navstack.carla_bridge import CarlaNavigationEnv
from navstack.config import EnvConfig
from scripts.native_curve_control import terminal_reason


def route():
    points = [{"x": 100., "y": 200., "z": .2, "yaw_degrees_carla": 90.},
              {"x": 98., "y": 210., "z": .2, "yaw_degrees_carla": 120.}]
    digest = hashlib.sha256(json.dumps([[round(p[k], 3) for k in ("x", "y", "yaw_degrees_carla")] for p in points]).encode()).hexdigest()
    return {"points": points, "map": "fixture", "opendrive_sha256": "map_digest", "geometry_id": digest}


def test_native_world_local_roundtrip_and_heading_handedness():
    native = route()
    layout = local_layout(native, "fixture", "map_digest")
    angle = math.radians(layout["angle"])
    for world, local in zip(native["points"], layout["path"]):
        assert layout["origin"][0]+local[0]*math.cos(angle)+local[1]*math.sin(angle) == pytest.approx(world["x"])
        assert layout["origin"][1]+local[0]*math.sin(angle)-local[1]*math.cos(angle) == pytest.approx(world["y"])
    assert layout["path"][0][2] == pytest.approx(0.)
    assert layout["path"][-1][2] == pytest.approx(-math.pi/6)
    assert layout["spawn_z"] == pytest.approx(.8)
    with pytest.raises(ValueError, match="different map"):
        local_layout(native, "different", "map_digest")
    native["points"][-1]["x"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        local_layout(native, "fixture", "map_digest")


def test_native_road_failure_precedes_goal_and_no_rectangle_only_training():
    protocol = {"goal_tolerance_m": .8, "goal_speed_m_s": .25, "goal_yaw_tolerance_rad": .35, "episode_seconds": 45}
    measured = {"state": {"x": 0, "y": 0, "yaw": 0, "v": 0, "time": 45},
                "collisions": [], "road_surface_probe": {"offroad_points_world": []}}
    assert terminal_reason(measured, [0, 0, 0], protocol) == "success"
    measured["state"]["v"] = -1.
    assert terminal_reason(measured, [0, 0, 0], protocol) == "deadline"
    measured["state"]["v"] = 0.
    measured["road_surface_probe"]["offroad_points_world"] = [[1, 2, 0]]
    assert terminal_reason(measured, [0, 0, 0], protocol) == "sampled_footprint_offroad"
    measured["collisions"] = [{"frame": 1}]
    assert terminal_reason(measured, [0, 0, 0], protocol) == "collision"
    # Isolated guard: no RPC is made for unsupported PPO native-road observations.
    env = CarlaNavigationEnv(EnvConfig(), None)
    try:
        with pytest.raises(ValueError, match="fixed-reference"):
            env.reset(options={"case": {"scenario": "native_curve"}})
    finally:
        env.close()

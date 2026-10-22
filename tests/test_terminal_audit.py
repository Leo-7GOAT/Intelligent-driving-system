import math
from scripts.audit_terminal_failures import summarize_frames


def test_goal_gates_must_coincide_and_heading_wraps():
    config = {"goal_tolerance": .8, "goal_speed": .25, "goal_yaw_tolerance": .35, "parking": True}
    frames = [{"state": {"x": x, "y": 0, "yaw": yaw, "v": v, "time": t},
               "requested_command": {"acceleration": -1}} for x, yaw, v, t in
              [(0, -math.pi, 1, 1), (2, math.pi, 0, 2), (0, 0, 0, 3)]]
    result = summarize_frames(frames, (0, 0, math.pi), config)
    assert result["ever_in_position_tolerance"]
    assert result["goal_frames_above_speed_limit"] == 1
    assert result["goal_frames_outside_heading_limit"] == 1
    assert result["simultaneous_goal_gate_frames"] == 0
    frames.append({"state": {"x": 0, "y": 0, "yaw": -math.pi, "v": 0, "time": 4},
                   "requested_command": {"acceleration": -1}})
    assert summarize_frames(frames, (0, 0, math.pi), config)["simultaneous_goal_gate_frames"] == 1

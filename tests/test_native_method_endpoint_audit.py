import gzip
import hashlib
import json

import pytest

from navstack.experiments import write_json
from scripts.audit_native_method_endpoint import audit


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _response(episode, request, frame, x):
    return {
        "ok": True, "episode_id": episode, "request_id": request,
        "frame": frame, "imu_frame": frame,
        "state": {"x": x, "y": 0.0, "yaw": 0.0, "v": 1.0,
                  "steering": 0.0, "acceleration": 0.0, "time": 0.05},
        "command": {"acceleration": 0.0, "steering": 0.0},
        "collisions": [], "sensor_health": "frame_heartbeat_ok",
        "vehicle_control": {"throttle": 0.2, "brake": 0.0, "steer": 0.0},
        "speed_setpoint": 1.0, "road_surface_probe": None,
    }


def _tick(response):
    evidence = {key: response[key] for key in (
        "frame", "request_id", "episode_id", "imu_frame", "collisions",
        "sensor_health", "vehicle_control", "speed_setpoint", "road_surface_probe")}
    return {"state": response["state"], "command": response["command"],
            "backend_frame": response["frame"], "backend_evidence": evidence}


def _batch(tmp_path):
    batch = tmp_path / "batch"; method = batch / "B1"; checkpoint = method / "checkpoint_000000001"
    (batch / "infrastructure").mkdir(parents=True); checkpoint.mkdir(parents=True)
    write_json(batch / "protocol.json", {"methods": ["B1"], "recovery": {}})
    write_json(method / "config.json", {"env": {"method": "B1"}})
    train = _response("train", 2, 10, 1.0)
    evaluate = _response("evaluate", 4, 20, 2.0)
    responses = [
        {"response": {"ok": True, "episode_id": "train", "request_id": 1,
                      "reset_ticks": 2, "frame": 9}},
        {"response": train},
        {"response": {"ok": True, "episode_id": "evaluate", "request_id": 3,
                      "reset_ticks": 2, "frame": 19}},
        {"response": evaluate},
    ]
    _write_gzip(batch / "infrastructure/bridge_fixture_responses.jsonl.gz", responses)
    _write_gzip(method / "transitions_000.jsonl.gz", [{
        "episode_id": "train", "trace": [_tick(train)], "termination_reason": "deadline"}])
    terminal = {"physics_steps": 1, "actions": 1, "success": False,
                "termination_reason": "deadline"}
    _write_gzip(method / "evaluation_trace.jsonl.gz", [{
        "case_id": 0, "info": {"trace": [_tick(evaluate)],
                                  "termination_reason": "deadline",
                                  "episode_metrics": terminal}}])
    for name, content in (("model.zip", b"model"), ("normalization.pkl", b"norm"),
                          ("resume_state.pkl", b"rng")):
        (checkpoint / name).write_bytes(content)
    (method / "model.zip").write_bytes(b"model")
    (method / "normalization.pkl").write_bytes(b"norm")
    metadata = {"physics_steps": 1, "lineage_control_steps": 1,
                "charged_control_steps_lower_bound": 1, "optimizer_steps": 2,
                "model_sha256": _sha(checkpoint / "model.zip"),
                "normalization_sha256": _sha(checkpoint / "normalization.pkl"),
                "resume_state_sha256": _sha(checkpoint / "resume_state.pkl")}
    write_json(checkpoint / "metadata.json", metadata)
    write_json(method / "latest_checkpoint.json", {"folder": checkpoint.name})
    write_json(method / "summary.json", {
        "physics_steps": 1, "lineage_control_steps": 1,
        "charged_control_steps_lower_bound": 1,
        "inherited_controls_not_in_checkpoint": 0,
        "optimization_phases": 1, "optimizer_steps": 2, "world_ticks": 3})
    (method / "optimization.jsonl").write_text(json.dumps({
        "physics_steps": 1, "optimization_phases": 1, "optimizer_steps": 2}) + "\n",
        encoding="utf-8")
    write_json(method / "evaluation.json", {
        "episodes": [{"case_id": 0, **terminal}], "world_ticks": 3,
        "model_sha256": _sha(method / "model.zip"),
        "normalization_sha256": _sha(method / "normalization.pkl")})
    return batch


def test_method_audit_binds_every_control_to_bridge(tmp_path):
    result = audit(_batch(tmp_path), "B1")
    assert result["training_control_frames_verified"] == 1
    assert result["evaluation_control_frames_verified"] == 1
    assert result["total_applied_controls_verified"] == 2
    assert result["training_world_ticks_verified"] == 3


def test_method_audit_rejects_trace_state_not_measured_by_carla(tmp_path):
    batch = _batch(tmp_path)
    path = batch / "B1/transitions_000.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        row = json.loads(next(stream))
    row["trace"][0]["state"]["x"] = 99.0
    _write_gzip(path, [row])
    with pytest.raises(ValueError, match="measured CARLA state"):
        audit(batch, "B1")

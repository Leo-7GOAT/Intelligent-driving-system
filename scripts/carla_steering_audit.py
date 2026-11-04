"""Bounded actual-CARLA steering sweep; diagnostic only, never PPO evidence.

Prepare with the project interpreter; execute --batch with the isolated 3.7
client. The worker reuses the bridge's owned actors and frame/sensor fences.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare(calibration_batch):
    from navstack.research import resources_ok
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError("Resource guard: " + str(reason))
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_carla_steering_audit")
    batch.mkdir()
    (batch / "infrastructure").mkdir()
    (batch / "calibration").mkdir()
    calibration = Path(calibration_batch) / "calibration" / "calibration.json"
    shutil.copy2(calibration, batch / "calibration" / "calibration.json")
    hashes = {}
    for folder in ("scripts", "navstack"):
        for source in (ROOT / folder).rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            relative = source.relative_to(ROOT)
            target = batch / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    phases = [{"target_speed_m_s": speed, "normalized_steer": steer}
              for speed in (.5, 1.5) for steer in (0, .1, -.1, .25, -.25, .46, -.46, .75, -.75, 1., -1.)]
    now = time.time()
    write(batch / "protocol.json", {"scope": "Actuator diagnosis, no PPO/planning/controller performance claim; old mapping unchanged",
          "created_epoch": now, "deadline_epoch": now+900, "phases": phases,
          "phase_steps": 160, "straight_steps": 60, "steer_end_step": 140,
          "measurement_steps": [100, 140], "dt": .05, "source_hashes": hashes,
          "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
          "resource_limits": {"wall_seconds": 900, "disk_min_GiB": 12, "ram_min_GiB": 4,
                              "gpu_stop_temperature_C": 85, "automatic_retries": 0},
          "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write(ROOT / "runs" / "carla_steering_audit.json", {"batch": str(batch)})
    print(batch)


def summarize_phase(rows, expected_gain):
    measured = [r for r in rows if 100 <= r["step"] < 140]
    if len(measured) != 40:
        raise ValueError("Incomplete registered steering measurement interval")
    steer = rows[0]["phase"]["normalized_steer"]
    actual = [r["measurement"]["state"]["steering"] for r in measured]
    errors = [a + expected_gain*steer for a in actual]
    return {"phase": rows[0]["phase"], "control_ticks": len(rows),
            "actual_speed_mean_m_s": sum(r["measurement"]["state"]["v"] for r in measured)/len(measured),
            "actual_steer_mean_rad": sum(actual)/len(actual),
            "old_linear_mapping_error_mean_rad": sum(errors)/len(errors),
            "old_linear_mapping_error_max_abs_rad": max(abs(e) for e in errors),
            "collision_events": rows[-1]["measurement"]["collisions"],
            "measurement_frames": [r["measurement"]["frame"] for r in measured]}


def guard(batch, protocol):
    if time.time() >= protocol["deadline_epoch"] or (batch / "STOP").exists():
        raise RuntimeError("Diagnostic deadline/STOP reached")
    if shutil.disk_usage(batch).free < 12 * 1024**3:
        raise RuntimeError("Disk below12GiB")
    import ctypes
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
            ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) or memory.avail_phys < 4*1024**3:
        raise RuntimeError("Memory guard unavailable or below4GiB")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if gpu.returncode or max(int(v) for v in gpu.stdout.split()) >= 85:
        raise RuntimeError("GPU guard unavailable or at85C")


def run(batch):
    import carla
    from scripts.carla_bridge37 import Bridge
    batch = Path(batch).resolve()
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    for name, digest in protocol["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Execute the registered frozen source")
    calibration = batch / "calibration" / "calibration.json"
    if hashlib.sha256(calibration.read_bytes()).hexdigest() != protocol["calibration_sha256"]:
        raise ValueError("Calibration changed")
    if (batch / "responses.jsonl").exists():
        raise RuntimeError("Preserve prior evidence; automatic repeats are disabled")
    guard(batch, protocol)
    bridge, phases = None, []
    try:
        bridge = Bridge(calibration, batch / "carla_tick_owner.lock")
        original_frame = bridge.world.get_snapshot().frame
        with (batch / "responses.jsonl").open("w", encoding="utf-8") as stream:
            for index, phase in enumerate(protocol["phases"]):
                guard(batch, protocol)
                initial = bridge.begin({"episode_id": "steering_%02d" % index, "scenario": "empty", "condition": 7301})
                rows = []
                for step in range(protocol["phase_steps"]):
                    if time.time() >= protocol["deadline_epoch"]:
                        raise RuntimeError("Diagnostic deadline")
                    speed = bridge.read()["state"]["v"]
                    target = phase["target_speed_m_s"]
                    steer = phase["normalized_steer"] if protocol["straight_steps"] <= step < protocol["steer_end_step"] else 0.
                    braking = .4 if step >= protocol["steer_end_step"] else max(0., (speed-target)*.4)
                    throttle = 0. if braking > .02 else max(0., min(.6, .12+.35*(target-speed)))
                    control = carla.VehicleControl(throttle=throttle, brake=braking, steer=steer)
                    bridge.actor.apply_control(control)
                    bridge.tick()
                    measurement = bridge.read()
                    snapshot = bridge.world.get_snapshot()
                    if snapshot.frame != measurement["frame"] or measurement["imu_frame"] != measurement["frame"]:
                        raise RuntimeError("Measurement/world/sensor frame mismatch")
                    # This is direct raw actuator calibration, not bridge.step or PPO.
                    measurement.pop("command", None)
                    wheels = [bridge.actor.get_wheel_steer_angle(k) for k in
                              (carla.VehicleWheelLocation.FL_Wheel, carla.VehicleWheelLocation.FR_Wheel)]
                    row = {"phase_id": index, "phase": phase, "step": step,
                           "world_elapsed_seconds": snapshot.timestamp.elapsed_seconds,
                           "raw_vehicle_control": {"throttle": throttle, "brake": braking, "steer": steer},
                           "front_wheel_angles_deg": wheels, "measurement": measurement}
                    stream.write(json.dumps(row)+"\n")
                    rows.append(row)
                # Preserve delayed collision callbacks before classifying the phase.
                time.sleep(.3)
                rows[-1]["measurement"]["collisions"] = bridge.read()["collisions"]
                summary = summarize_phase(rows, bridge.cal["recommended_adapter"]["steer_gain"])
                summary.update(phase_id=index, initial_frame=initial["frame"], map=bridge.map.name)
                phases.append(summary)
                stream.flush()
                write(batch / "heartbeat.json", {"epoch": time.time(), "phase": index, "frame": bridge.frame})
                write(batch / "partial_summary.json", {"scope": "diagnostic_not_PPO", "phases": phases})
                print(json.dumps(summary), flush=True)
        write(batch / "summary.json", {"scope": "diagnostic_not_PPO", "complete": True,
              "client_version": bridge.client.get_client_version(), "server_version": bridge.client.get_server_version(),
              "world_ticks_before_final_cleanup": bridge.frame-original_frame, "control_ticks": sum(p["control_ticks"] for p in phases),
              "phases": phases, "original_adapter_unchanged": True})
    except BaseException as exc:
        write(batch / "failure.json", {"error": repr(exc), "completed_phases": len(phases)})
        raise
    finally:
        if bridge is not None:
            bridge.close()
            remaining = [a.id for a in bridge.world.get_actors() if a.type_id.startswith(("vehicle.", "sensor.", "walker."))]
            write(batch / "cleanup.json", {"remaining_actor_ids": remaining, "empty": not remaining,
                  "world_settings": "Owned server left empty/synchronous pending identity-checked shutdown"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", metavar="CALIBRATION_BATCH")
    group.add_argument("--batch")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        run(args.batch)

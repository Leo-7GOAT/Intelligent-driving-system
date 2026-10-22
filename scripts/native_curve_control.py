"""Actual CARLA native-road fixed-reference MPC diagnosis, explicitly not PPO."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from navstack.carla_bridge import BridgeClient
from navstack.config import VehicleConfig
from navstack.experiments import jsonable
from navstack.metrics import timing_summary
from navstack.mpc import MPC
from navstack.native_route import select_routes
from navstack.research import resources_ok
from navstack.trajectory import parameterize, tracking_error
from navstack.vehicle import VehicleState, Command, wrap


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=jsonable), encoding="utf-8")


def prepare(survey_batch, adapter="legacy", fit_path=None):
    if adapter == "acceleration_ff_v2" and fit_path is None:
        raise ValueError("Measured throttle fit is required")
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError("Resource guard: "+str(reason))
    parent = Path(survey_batch)
    survey = json.loads((parent / "native_routes.json").read_text(encoding="utf-8"))
    routes = select_routes(survey)
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_native_curve_control")
    batch.mkdir()
    (batch / "infrastructure").mkdir()
    (batch / "calibration").mkdir()
    shutil.copy2(parent / "calibration/calibration.json", batch / "calibration/calibration.json")
    calibration_path = batch / "calibration/calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if adapter != "legacy":
        calibration["recommended_adapter"]["longitudinal_mode"] = "legacy" if adapter == "manual_first_v1" else adapter
        if adapter in ("manual_first_v1", "acceleration_pi_v1", "acceleration_ff_v2"):
            calibration["recommended_adapter"]["drivetrain_mode"] = "manual_first_v1"
        if adapter in ("acceleration_pi_v1", "acceleration_ff_v2"):
            calibration["recommended_adapter"].update(acceleration_integral_gain=.15, acceleration_filter_seconds=.2)
        if adapter == "acceleration_ff_v2":
            fit_file = Path(fit_path)
            fitted = json.loads(fit_file.read_text(encoding="utf-8"))
            calibration["recommended_adapter"]["throttle_model"] = fitted["coefficients"]
            calibration["throttle_fit_sha256"] = hashlib.sha256(fit_file.read_bytes()).hexdigest()
            shutil.copy2(fit_file, batch / "calibration/throttle_fit.json")
        calibration["status"] = "experimental_actuation_not_validated_for_training"
        write(calibration_path, calibration)
    hashes = {}
    for folder in ("scripts", "navstack", "tests"):
        for source in (ROOT / folder).rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            relative = source.relative_to(ROOT)
            target = batch / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    now = time.time()
    write(batch / "protocol.json", {"scope": "Native-road fixed-reference MPC, no PPO or Hybrid A* performance claim; four geometrically selected development routes",
          "longitudinal_adapter": adapter, "created_epoch": now, "deadline_epoch": now+900, "source_hashes": hashes,
          "survey_sha256": hashlib.sha256((parent / "native_routes.json").read_bytes()).hexdigest(),
          "calibration_sha256": hashlib.sha256((batch / "calibration/calibration.json").read_bytes()).hexdigest(),
          "routes": routes, "target_speed_m_s": 1., "episode_seconds": 45., "dt": .05,
          "mpc_horizon": 20, "goal_tolerance_m": .8, "goal_speed_m_s": .25, "goal_yaw_tolerance_rad": .35,
          "resource_limits": {"wall_seconds": 900, "disk_min_GiB": 12, "ram_min_GiB": 4,
                              "gpu_stop_temperature_C": 85, "automatic_retries": 0},
          "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write(ROOT / "runs/native_curve_control.json", {"batch": str(batch)})
    print(batch)


def terminal_reason(measurement, goal, protocol):
    state = measurement["state"]
    if measurement["collisions"]:
        return "collision"
    if measurement["road_surface_probe"]["offroad_points_world"]:
        return "sampled_footprint_offroad"
    if (math.hypot(state["x"]-goal[0], state["y"]-goal[1]) <= protocol["goal_tolerance_m"]
            and abs(state["v"]) <= protocol["goal_speed_m_s"]
            and abs(wrap(state["yaw"]-goal[2])) <= protocol["goal_yaw_tolerance_rad"]):
        return "success"
    if state["time"] >= protocol["episode_seconds"]-1e-8:
        return "deadline"
    return None


def run(batch):
    from scripts.carla_steering_audit import guard
    batch = Path(batch).resolve()
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    for name, digest in protocol["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Execute the frozen registered source")
    if hashlib.sha256((batch / "calibration/calibration.json").read_bytes()).hexdigest() != protocol["calibration_sha256"]:
        raise ValueError("Calibration changed")
    if (batch / "started.json").exists():
        raise RuntimeError("Refuse automatic repeat; preserve prior evidence")
    guard(batch, protocol)
    write(batch / "started.json", {"epoch": time.time()})
    vehicle = VehicleConfig(**json.loads((batch / "calibration/calibration.json").read_text(encoding="utf-8"))["vehicle"])
    client, episodes = None, []
    try:
        client = BridgeClient(batch)
        for case_id, route in enumerate(protocol["routes"]):
            guard(batch, protocol)
            measured = client.request("begin", scenario="native_curve", native_route=route, condition=case_id)
            if measured["road_surface_probe"]["offroad_points_world"]:
                raise RuntimeError("Initial actual footprint is off the native road")
            state = VehicleState(**measured["state"])
            goal = measured["scene"]["goal"]
            reference = parameterize(measured["scene"]["native_layout"]["path"], protocol["target_speed_m_s"], state.v, vehicle, state.time, stop=True)
            case = batch / ("case_%02d" % case_id)
            case.mkdir()
            write(case / "scene.json", measured["scene"])
            write(case / "reference.json", asdict(reference))
            controller = MPC(vehicle, protocol["dt"], protocol["mpc_horizon"])
            previous = Command(**measured["command"])
            errors, cycle_times, qp_times, path_length, failures = [], [], [], 0., 0
            reason = None
            with gzip.open(case / "transitions.jsonl.gz", "wt", encoding="utf-8") as stream:
                while reason is None:
                    if time.time() >= protocol["deadline_epoch"]:
                        raise RuntimeError("Diagnostic deadline reached")
                    if len(cycle_times) % 100 == 0:
                        guard(batch, protocol)
                    started = time.perf_counter()
                    control = controller.step(state, reference, previous)
                    before = state
                    measured = client.request("step", expected_frame=measured["frame"], **asdict(control.command))
                    state = VehicleState(**measured["state"])
                    previous = Command(**measured["command"])
                    error = tracking_error(state, reference)
                    errors.append(error)
                    failures += control.status != "solved"
                    path_length += math.hypot(state.x-before.x, state.y-before.y)
                    qp_times.append(control.diagnostics["qp_seconds"])
                    reason = terminal_reason(measured, goal, protocol)
                    cycle_times.append(time.perf_counter()-started)
                    stream.write(json.dumps({"measurement": measured, "requested_command": asdict(control.command),
                          "controller_status": control.status, "controller_diagnostics": control.diagnostics,
                          "tracking": error, "cycle_seconds": cycle_times[-1]}, default=jsonable)+"\n")
                final = client.request("finalize")
                write(case / "terminal_measurement.json", final)
                reason = terminal_reason(final, goal, protocol)
            summary = {"case_id": case_id, "geometry_id": route["geometry_id"], "heading_change_degrees": route["heading_change_degrees"],
                  "success": reason == "success", "termination_reason": reason, "control_ticks": len(errors),
                  "actual_seconds": state.time, "actual_path_length_m": path_length, "reference_length_m": float(reference.s[-1]),
                  "final_state": asdict(state), "final_goal_distance_m": math.hypot(state.x-goal[0], state.y-goal[1]),
                  "tracking_rms": {key: float(np.sqrt(np.mean([e[key]**2 for e in errors]))) for key in ("lateral", "heading", "speed")},
                  "mpc_failures": failures, "full_control_cycle": timing_summary(cycle_times, .05),
                  "qp_times": timing_summary(qp_times, .05), "collisions": final["collisions"]}
            write(case / "summary.json", summary)
            episodes.append(summary)
            write(batch / "partial_summary.json", {"scope": "diagnostic_fixed_reference_MPC_not_PPO", "episodes": episodes})
            print(json.dumps(summary), flush=True)
        write(batch / "summary.json", {"scope": "diagnostic_fixed_reference_MPC_not_PPO", "episodes": episodes,
              "world_ticks_before_final_cleanup": client.total_ticks, "control_ticks": sum(e["control_ticks"] for e in episodes),
              "complete": True})
    except BaseException as exc:
        write(batch / "failure.json", {"error": repr(exc), "completed_cases": len(episodes)})
        raise
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", metavar="SURVEY_BATCH")
    group.add_argument("--batch")
    parser.add_argument("--adapter", choices=["legacy", "coast_mild_v1", "manual_first_v1", "acceleration_pi_v1", "acceleration_ff_v2"], default="legacy")
    parser.add_argument("--fit", help="Measured throttle fit for acceleration_ff_v2")
    args = parser.parse_args()
    prepare(args.prepare, args.adapter, args.fit) if args.prepare else run(args.batch)

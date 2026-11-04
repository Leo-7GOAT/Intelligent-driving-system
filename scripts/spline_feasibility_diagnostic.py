"""Post-test independent spline search, followed by actual MPC/plant execution.

All fixed unseen cases are retained. Privileged global geometry and full-route
references make this diagnostic unsuitable as a policy result. Failure to find
a witness is inconclusive. No training, scene filtering, or policy changes.
"""

import argparse
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.config import EnvConfig
from navstack.experiments import jsonable, write_json
from navstack.mpc import MPC
from navstack.trajectory import parameterize
from navstack.vehicle import VehiclePlant, VehicleState, wrap
from navstack.world import World


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def search(scene, config, until, seed):
    start = VehicleState(**scene["initial"])
    goal = scene["goal"]
    if scene["dynamic"] or abs(start.yaw) > 1e-8 or abs(goal[2]) > 1e-8:
        raise ValueError("This finite family requires static, aligned headings")
    vehicle = config.vehicle
    knots = np.linspace(start.x, goal[0], 9)
    xs = np.linspace(start.x, goal[0], 240)
    boxes = np.asarray(scene["static"])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) / 2
    half_length = (vehicle.front + vehicle.rear) / 2 + .18
    half_width = vehicle.width / 2 + .18
    offset = (vehicle.front - vehicle.rear) / 2
    evaluations = 0

    def spline(values):
        return CubicSpline(knots, np.r_[start.y, values, goal[1]], bc_type=((1, 0), (1, 0)))

    def objective(values):
        nonlocal evaluations
        evaluations += 1
        curve = spline(values)
        y, derivative, second = curve(xs), curve(xs, 1), curve(xs, 2)
        angle = np.arctan(derivative)
        c, s = np.cos(angle), np.sin(angle)
        cx, cy = xs + offset * c, y + offset * s
        dx, dy = cx[:, None] - centers[:, 0], cy[:, None] - centers[:, 1]
        ac, ass = abs(c[:, None]), abs(s[:, None])
        gaps = np.stack([
            abs(dx) - half_length * ac - half_width * ass - half[:, 0],
            abs(dy) - half_length * ass - half_width * ac - half[:, 1],
            abs(dx * c[:, None] + dy * s[:, None]) - half_length - half[:, 0] * ac - half[:, 1] * ass,
            abs(-dx * s[:, None] + dy * c[:, None]) - half_width - half[:, 0] * ass - half[:, 1] * ac,
        ])
        overlap = np.minimum(gaps.max(axis=0), 0)
        ex = half_length * abs(c) + half_width * abs(s)
        ey = half_length * abs(s) + half_width * abs(c)
        boundary = np.maximum(np.stack([ex-cx, cx+ex-config.width, ey-cy, cy+ey-config.height]), 0)
        curvature = second / (1 + derivative**2)**1.5
        excess = np.maximum(abs(curvature) - .98 / vehicle.min_turn_radius, 0)
        return float(100 * np.sum(overlap**2) + 100 * np.sum(boundary**2)
                     + 3000 * np.sum(excess**2) + .0001 * np.mean(derivative**2))

    result = differential_evolution(
        objective, [(2, config.height-2)] * 7, seed=seed,
        popsize=10, maxiter=350, tol=1e-7, polish=False,
        callback=lambda _x, _convergence: time.time() >= until,
        workers=1, updating="immediate",
    )
    dense_x = np.linspace(start.x, goal[0], 2000)
    curve = spline(result.x)
    derivative = curve(dense_x, 1)
    curvature = curve(dense_x, 2) / (1 + derivative**2)**1.5
    path = np.column_stack([dense_x, curve(dense_x), np.arctan(derivative)])
    world = World(config.width, config.height, static=[tuple(r) for r in scene["static"]])
    curvature_ok = float(max(abs(curvature))) <= 1 / vehicle.min_turn_radius
    geometry_ok = curvature_ok and all(world.free(p, vehicle, .10) for p in path)
    return (path if geometry_ok else None), {
        "objective_evaluations": evaluations, "objective": float(result.fun),
        "optimizer_converged": bool(result.success), "dense_samples": len(path),
        "max_curvature": float(max(abs(curvature))), "curvature_ok": curvature_ok,
        "full_body_geometry_ok": geometry_ok, "knot_x": knots.tolist(),
        "knot_y": np.r_[start.y, result.x, goal[1]].tolist(),
    }


def execute(path, scene, config, stream, case_id, until):
    reference = parameterize(path, 1.0, 0.0, config.vehicle, stop=True)
    plant = VehiclePlant(config.vehicle, VehicleState(**scene["initial"]))
    controller = MPC(config.vehicle, config.control_dt, config.mpc_horizon)
    world = World(config.width, config.height, static=[tuple(r) for r in scene["static"]])
    goal = scene["goal"]
    reason, ticks, mpc_failures = "deadline", 0, 0
    while plant.state.time < config.deadline - 1e-8:
        if time.time() >= until:
            reason = "diagnostic_wall_budget"
            break
        before = plant.state
        control = controller.step(before, reference, plant.previous_command)
        state = plant.step(control.command, config.control_dt)
        ticks += 1
        mpc_failures += not control.status.startswith("solved")
        stream.write(json.dumps({"case_id": case_id, "policy": "diagnostic_spline_MPC_no_PPO",
                                 "state": asdict(state), "command": asdict(control.command),
                                 "controller_status": control.status}, default=jsonable) + "\n")
        poses = [(before.x+f*(state.x-before.x), before.y+f*(state.y-before.y),
                  before.yaw+f*wrap(state.yaw-before.yaw)) for f in (.25,.5,.75,1)]
        if any(world.out_of_bounds(p, config.vehicle) for p in poses):
            reason = "out_of_bounds"
            break
        if any(world.static_collision(p, config.vehicle) for p in poses):
            reason = "collision"
            break
        if (math.hypot(state.x-goal[0], state.y-goal[1]) <= config.goal_tolerance
                and abs(state.v) <= config.goal_speed
                and abs(wrap(state.yaw-goal[2])) <= config.goal_yaw_tolerance):
            reason = "success"
            break
    return {"physical_steps": ticks, "outcome": reason, "mpc_failure_ticks": mpc_failures,
            "final_state": asdict(plant.state), "reference_duration": float(reference.t[-1])}


def main(batch):
    batch = Path(batch)
    marker = read(batch / "formal_v5/H_seed11/final_primary.json")
    if not marker.get("complete"):
        raise ValueError("Diagnostic is explicitly post-test")
    source = Path(marker["run"])
    config = EnvConfig.from_dict(read(source / "config.json")["env"])
    cases = [json.loads(line) for line in (source / "scenes.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = [r for r in cases if r["case"]["scenario"] == "unseen"]
    output = batch / "analysis/posthoc_spline_feasibility"
    output.mkdir(exist_ok=False)
    until = min(time.time()+600, read(batch / "batch.json")["deadline_epoch"]-90)
    results = []
    write_json(output / "protocol.json", {"case_ids": [r["case_id"] for r in cases],
        "scope": "Post-test descriptive geometry diagnostic, not policy results or scenario selection",
        "per_case_search_seconds": 35, "total_seconds": 600, "seed_base": 20261016,
        "ppo_training_steps": 0, "physical_deadline_seconds": config.deadline})
    with gzip.open(output / "physical_trace.jsonl.gz", "wt", encoding="utf-8") as stream:
        for case in cases:
            result = {"case_id": case["case_id"], "case": case["case"], "diagnostic_only": True}
            if time.time() >= until:
                result["status"] = "diagnostic_wall_budget_not_searched"
            else:
                path, diagnostics = search(case["initial_scenario"], config,
                    min(time.time()+35, until), 20261016+case["case_id"])
                result.update(diagnostics)
                if path is None:
                    result["status"] = "no_witness_in_this_family_inconclusive"
                else:
                    np.save(output / f"path_case{case['case_id']}.npy", path)
                    result["status"] = "physical_witness_executed"
                    result.update(execute(path, case["initial_scenario"], config, stream, case["case_id"], until))
            results.append(result)
            write_json(output / "results.json", {"scope": "Independent bounded spline search; actual MPC/plant execution only after dense full-body/curvature validation. No witness is not proof of infeasibility. All fixed cases retained.",
                "physical_cases_executed": sum(r.get("physical_steps",0)>0 for r in results),
                "physical_steps_executed": sum(r.get("physical_steps",0) for r in results),
                "successful_physical_witnesses": sum(r.get("outcome")=="success" for r in results),
                "scheduled_cases": len(cases), "complete": len(results)==len(cases), "results": results})
            print(json.dumps({k:v for k,v in result.items() if k not in ["knot_x","knot_y","final_state"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    main(parser.parse_args().batch)

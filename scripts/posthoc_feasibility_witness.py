"""Post-test diagnostic on every fixed unseen-layout case, with no case exclusion.

Analytic smooth lateral bumps are independent of PPO and Hybrid A*. A found route
is executed through the same MPC and physical plant. No witness is inconclusive.
These privileged full-route diagnostics are never included in policy statistics.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.config import EnvConfig
from navstack.mpc import MPC
from navstack.trajectory import parameterize
from navstack.vehicle import VehiclePlant, VehicleState, wrap
from navstack.world import World
from navstack.experiments import jsonable, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidates(start, goal, vehicle):
    x = np.linspace(start.x, goal[0], 800)
    for length in (14.0, 16.0, 18.0, 20.0, 23.0):
        for amplitude in (
            0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            2.0,
            -2.0,
            2.5,
            -2.5,
            3.0,
            -3.0,
            4.0,
            -4.0,
            5.0,
            -5.0,
            6.0,
            -6.0,
        ):
            u = np.clip((x - start.x) / length, 0, 1)
            angle = np.pi * u
            y = start.y + amplitude * np.sin(angle) ** 3
            derivative = (
                amplitude * 3 * np.sin(angle) ** 2 * np.cos(angle) * np.pi / length
            )
            second = (
                amplitude
                * (6 * np.sin(angle) * np.cos(angle) ** 2 - 3 * np.sin(angle) ** 3)
                * (np.pi / length) ** 2
            )
            curvature = second / (1 + derivative**2) ** 1.5
            if max(abs(curvature)) > 1 / vehicle.min_turn_radius:
                continue
            yield length, amplitude, np.column_stack([x, y, np.arctan(derivative)])


def main(batch):
    batch = Path(batch)
    marker = read(batch / "formal_v5/H_seed11/final_primary.json")
    if not marker.get("complete"):
        raise ValueError("This diagnostic is explicitly post-test")
    source = Path(marker["run"])
    config = EnvConfig.from_dict(read(source / "config.json")["env"])
    cases = [
        json.loads(line)
        for line in (source / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cases = [r for r in cases if r["case"]["scenario"] == "unseen"]
    output = batch / "analysis/posthoc_feasibility_witness"
    output.mkdir(exist_ok=False)
    limit = min(time.time() + 180, read(batch / "batch.json")["deadline_epoch"] - 60)
    results = []
    with gzip.open(
        output / "diagnostic_physical_trace.jsonl.gz", "wt", encoding="utf-8"
    ) as stream:
        for case in cases:
            result = {
                "case_id": case["case_id"],
                "case": case["case"],
                "diagnostic_only": True,
                "ppo_training_steps": 0,
            }
            scene = case["initial_scenario"]
            start = VehicleState(**scene["initial"])
            goal = scene["goal"]
            world = World(config.width, config.height)
            world.static = [tuple(r) for r in scene["static"]]
            if scene["dynamic"] or abs(start.y - goal[1]) > 1e-8 or abs(goal[2]) > 1e-8:
                raise ValueError(
                    "Witness family assumes static geometry and aligned endpoints"
                )
            witness = None
            searched = 0
            for length, amplitude, path in candidates(start, goal, config.vehicle):
                if time.time() >= limit:
                    break
                searched += 1
                if all(world.free(pose, config.vehicle, 0.10) for pose in path):
                    witness = (length, amplitude, path)
                    break
            result["candidate_paths_checked"] = searched
            if witness is None:
                result["status"] = (
                    "diagnostic_wall_budget"
                    if time.time() >= limit
                    else "no_witness_in_this_family_inconclusive"
                )
                results.append(result)
                continue
            length, amplitude, path = witness
            reference = parameterize(path, 1.0, 0.0, config.vehicle, stop=True)
            plant = VehiclePlant(config.vehicle, start)
            controller = MPC(config.vehicle, config.control_dt, config.mpc_horizon)
            reason = "deadline"
            ticks = 0
            while plant.state.time < config.deadline - 1e-8:
                if time.time() >= limit:
                    reason = "diagnostic_wall_budget"
                    break
                before = plant.state
                control = controller.step(before, reference, plant.previous_command)
                state = plant.step(control.command, config.control_dt)
                ticks += 1
                stream.write(
                    json.dumps(
                        {
                            "case_id": case["case_id"],
                            "policy": "diagnostic_analytic_route_MPC_no_PPO",
                            "state": asdict(state),
                            "command": asdict(control.command),
                            "controller_status": control.status,
                        },
                        default=jsonable,
                    )
                    + "\n"
                )
                poses = [
                    (
                        before.x + f * (state.x - before.x),
                        before.y + f * (state.y - before.y),
                        before.yaw + f * wrap(state.yaw - before.yaw),
                    )
                    for f in (0.25, 0.5, 0.75, 1.0)
                ]
                if any(world.out_of_bounds(p, config.vehicle) for p in poses):
                    reason = "out_of_bounds"
                    break
                if any(world.static_collision(p, config.vehicle) for p in poses):
                    reason = "collision"
                    break
                if (
                    math.hypot(state.x - goal[0], state.y - goal[1])
                    <= config.goal_tolerance
                    and abs(state.v) <= config.goal_speed
                    and abs(wrap(state.yaw - goal[2])) <= config.goal_yaw_tolerance
                ):
                    reason = "success"
                    break
            result.update(
                status="physical_witness_executed",
                bump_length=length,
                amplitude=amplitude,
                reference_duration=reference.t[-1],
                physical_steps=ticks,
                outcome=reason,
                final_state=asdict(plant.state),
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    write_json(
        output / "results.json",
        {
            "scope": "Exploratory feasibility diagnostic after final-test exposure. The independent analytic route family is checked first; same MPC/plant execution occurs only if a witness is found. All ten fixed unseen-layout cases retained; no policy result, scene, or selection changes. No witness is not proof of infeasibility.",
            "physical_cases_executed": sum(r.get("status") == "physical_witness_executed" for r in results),
            "physical_steps_executed": sum(r.get("physical_steps", 0) for r in results),
            "results": results,
        },
    )
    print(
        json.dumps(
            {
                "cases": len(results),
                "physical_successful_witnesses": sum(
                    r.get("outcome") == "success" for r in results
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    main(parser.parse_args().batch)

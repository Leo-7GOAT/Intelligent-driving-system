"""Counterfactual target feasibility on development geometry, no PPO claims."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.config import EnvConfig, PlannerConfig, VehicleConfig
from navstack.env import NavigationEnv
from navstack.subgoals import goal_frame_candidates
from navstack.trajectory import parameterize
from navstack.vehicle import VehicleState, VehiclePlant
from navstack.experiments import append_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--calibration", required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=False, parents=True)
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    started = time.time()
    write_json(output / "protocol.json", {"purpose": "development geometry diagnostic only", "node_budget_per_target": 1800,
               "wall_budget_seconds": 300, "seeds": [7301], "vehicle_profiles": ["lightweight", "carla_geometry"],
               "scenes": ["static", "unseen", "parking"], "physics_steps": 0, "ppo_updates": 0,
               "note": "CARLA geometry does not imply CARLA execution. No success filtering or final evaluation."})
    rows = []
    for profile, vehicle in (("lightweight", VehicleConfig()), ("carla_geometry", VehicleConfig(**cal["vehicle"]))):
        for scenario in ("static", "unseen", "parking"):
            env = NavigationEnv(EnvConfig(vehicle=vehicle, scenario="empty" if scenario == "parking" else scenario,
                                         split="validation", obstacle_num=0, planner=PlannerConfig(max_seconds=None), terminal_goal_prefix=True))
            env.reset(seed=7301)
            if scenario == "parking":
                env.plant = VehiclePlant(vehicle, VehicleState(env.goal[0]-2, env.goal[1]-.2, yaw=.15, v=.75))
            old = [env.decode_action(a) for a in range(1, 31, 2)]
            new = list(goal_frame_candidates(env.state, env.goal))[1::2]
            for kind, candidates in (("old_arc", old), ("goal_frame", new)):
                for index, (target, speed) in enumerate(candidates):
                    if time.time() - started > 300:
                        write_json(output / "summary.json", {"state": "budget_censored", "completed": len(rows), "wall_seconds": time.time()-started})
                        return
                    plan = env.planner.plan((env.state.x, env.state.y, env.state.yaw), target, env.world)
                    trajectory_status = None
                    if plan.status == "success":
                        try:
                            reference = parameterize(plan.path, speed, env.state.v, vehicle, stop=math.dist(target[:2], env.goal[:2]) < .01)
                            trajectory_status = "valid" if all(env.world.free(p, vehicle) for p in zip(reference.x, reference.y, reference.yaw)) else "body_collision"
                        except ValueError as exc:
                            trajectory_status = str(exc)
                    row = {"vehicle": profile, "scenario": scenario, "kind": kind, "candidate": index,
                           "state": asdict(env.state), "goal": env.goal, "target": target, "plan_status": plan.status,
                           "plan_reason": plan.failure_reason, "expanded_nodes": plan.expanded_nodes,
                           "planning_seconds": plan.planning_time, "trajectory_status": trajectory_status}
                    append_json(output / "targets.jsonl", row); rows.append(row)
                subset = [r for r in rows if r["vehicle"] == profile and r["scenario"] == scenario and r["kind"] == kind]
                print(json.dumps({"vehicle": profile, "scenario": scenario, "kind": kind, "valid": sum(r["trajectory_status"] == "valid" for r in subset), "total": len(subset)}), flush=True)
    write_json(output / "summary.json", {"state": "complete", "completed": len(rows), "wall_seconds": time.time()-started})


if __name__ == "__main__":
    main()

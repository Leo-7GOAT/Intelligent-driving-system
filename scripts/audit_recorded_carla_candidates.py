"""Offline spatial counterfactuals at recorded CARLA failures; never ticks CARLA."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.vehicle import VehicleState
from navstack.world import World
from navstack.planning import HybridAStar
from navstack.subgoals import goal_frame_candidates
from navstack.experiments import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", required=True)
    args = parser.parse_args()
    batch = Path(args.transfer).resolve()
    output = batch / "spatial_candidate_audit"
    output.mkdir(exist_ok=False)
    hashes = {}
    for source in [*(ROOT / "navstack").glob("*.py"), Path(__file__)]:
        target = output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = []
    for label in ("H", "H0"):
        run = batch / "carla" / label
        config = EnvConfig.from_dict(read(run / "config.json")["env"])
        planner = HybridAStar(config.vehicle, config.planner)
        scenes = {r["case_id"]: r for r in [json.loads(s) for s in (run / "scenes.jsonl").read_text(encoding="utf-8").splitlines()]}
        failures = [r for r in [json.loads(s) for s in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()] if not r["success"]]
        actions = read(run / "action_failure_audit.json")
        for failure in failures:
            case_id = failure["case_id"]
            scene = scenes[case_id]
            world = World(scene["scene"]["width"], scene["scene"]["height"], scene["scene"]["static"])
            goal = scene["scene"]["goal"]
            for stage, values in (("initial", scene["initial_state"]["initial"]), ("failed_final", actions[str(case_id)]["last"]["state"])):
                state = VehicleState(**values)
                pose = (state.x, state.y, state.yaw)
                results, unique = [], {}
                for action, (target, speed) in enumerate(goal_frame_candidates(state, goal, config.vehicle)):
                    if action == 0:continue
                    if target not in unique:
                        plan = planner.plan(pose, target, world)
                        unique[target] = {"status": plan.status, "failure_reason": plan.failure_reason,
                                          "expanded_nodes": plan.expanded_nodes, "planning_time": plan.planning_time}
                    results.append({"action": action, "target": target, "speed": speed, **unique[target]})
                direct = planner.plan(pose, goal, world)
                row = {"policy": label, "case": failure["case"], "stage": stage,
                       "unique_target_count": len(unique), "spatially_plannable_unique_targets": sum(r["status"] == "success" for r in unique.values()),
                       "direct_goal_spatial_status": direct.status, "candidates": results}
                rows.append(row)
                write_json(output / "partial.json", {"rows": rows})
                print(json.dumps({k: row[k] for k in row if k != "candidates"}), flush=True)
    write_json(output / "summary.json", {"scope": "Offline spatial feasibility only; no CARLA ticks, PPO updates, dynamic safety or tracking-success claim", "source_hashes": hashes, "rows": rows})


if __name__ == "__main__":
    main()

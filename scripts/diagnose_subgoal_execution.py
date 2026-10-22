"""Development B3 closed-loop diagnostic; never counts as trained PPO evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import shutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.config import EnvConfig, PlannerConfig, VehicleConfig
from navstack.env import NavigationEnv, rule_action
from navstack.experiments import write_json, append_json, jsonable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--seconds", type=int, default=600)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    started = time.time()
    hashes = {str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (Path(__file__).resolve().parents[1]/"navstack").glob("*.py")}
    write_json(output / "protocol.json", {"scope": "rule/low-level development only, zero PPO updates", "scene_seed": 7301,
              "deadline_seconds": 45, "wall_budget": args.seconds, "random_extra_obstacles": 0,
              "variants": ["legacy", "analytic_arcs", "analytic_goal_frame"], "source_hashes": hashes})
    for name in hashes:
        target = output / "source" / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[1]/name,target)
    shutil.copy2(__file__,output/"executed_script.py")
    rows = []
    for profile, vehicle in (("lightweight", VehicleConfig()), ("carla_geometry", VehicleConfig(**cal["vehicle"]))):
        for scene in ("empty", "static", "unseen"):
            for variant in ("legacy", "analytic_arcs", "analytic_goal_frame"):
                config = EnvConfig(method="B3", scenario=scene, split="validation", obstacle_num=0, deadline=45,
                                   vehicle=vehicle, terminal_goal_prefix=True,
                                   subgoal_mode="goal_frame" if variant == "analytic_goal_frame" else "ego_arcs",
                                   planner=PlannerConfig(max_seconds=None,analytic_expansion=variant != "legacy"))
                env = NavigationEnv(config)
                env.reset(seed=7301)
                case = profile + "_" + scene + "_" + variant
                decisions = []
                with gzip.open(output / (case + ".jsonl.gz"), "wt", encoding="utf-8") as stream:
                    while time.time() - started < args.seconds:
                        before = time.perf_counter()
                        action = rule_action(env)
                        _, _, done, _, info = env.step(action)
                        cycle = time.perf_counter() - before
                        decisions.append(cycle)
                        stream.write(json.dumps({"action": action, "info": info, "full_decision_seconds": cycle},default=jsonable) + "\n")
                        if done:
                            row = {"profile": profile,"scenario": scene,"variant": variant, "initial": env.initial_scenario,
                                   "full_decision_seconds": decisions, **info["episode_metrics"]}
                            append_json(output/"episodes.jsonl",row); rows.append(row)
                            print(json.dumps({k:row[k] for k in ("profile","scenario","variant","success","progress","intervention_rate","plan_failure_rate")}),flush=True)
                            break
                    else:
                        write_json(output/"summary.json",{"status":"budget_censored","completed":len(rows),"unfinished_case":case})
                        return
                env.close()
    write_json(output/"summary.json",{"status":"complete","completed":len(rows),"wall_seconds":time.time()-started})


if __name__ == "__main__":
    main()

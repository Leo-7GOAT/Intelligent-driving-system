"""B3 against an existing frozen development protocol, with all failures kept."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=600)
    args = parser.parse_args()
    batch, output = Path(args.batch).resolve(), Path(args.output).resolve()
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    for name, digest in protocol["source_hashes"].items():
        if hashlib.sha256((batch / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source differs: " + name)
    sys.path.insert(0, str(batch / "source"))
    from navstack.env import NavigationEnv, rule_action
    from navstack.config import EnvConfig
    from navstack.experiments import write_json, jsonable
    from navstack.research import atomic_json
    config = EnvConfig.from_dict(next(j for j in protocol["jobs"] if j["id"] == "H_seed11")["env"])
    config.method, config.split = "B3", "validation"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", {"source_batch": str(batch), "source_hashes": protocol["source_hashes"],
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "env": config.to_dict(),
               "policy": "rule_only_B3", "scope": "development", "training_physics_steps": 0,
               "independent_training_seed": None, "cases": protocol["development_cases"]})
    started = time.time()
    rows = []
    with gzip.open(output / "transitions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index, case in enumerate(protocol["development_cases"]):
            if time.time() - started >= args.seconds:
                break
            config.scenario = case["scenario"]
            env = NavigationEnv(config)
            env.reset(seed=case["seed"], options=case.get("options"))
            costs = []
            try:
                while not env.done and time.time() - started < args.seconds:
                    before = time.perf_counter()
                    action = rule_action(env)
                    selection_seconds = time.perf_counter() - before
                    _, _, _, _, info = env.step(action)
                    costs.append(time.perf_counter() - before)
                    stream.write(json.dumps({"case_id": index, "case": case, "action": action,
                        "selection_seconds": selection_seconds, "full_cycle_seconds": costs[-1],
                        "rule_diagnostics": env.rule_diagnostics, "info": info}, default=jsonable) + "\n")
                if env.done:
                    rows.append({"case_id": index, **case, **info["episode_metrics"], "full_cycle_seconds": costs})
                    print(json.dumps({"case": case, "reason": info["termination_reason"]}), flush=True)
                else:
                    atomic_json(output / "censored.json", {"case_id": index, "reason": "wall_budget"})
            finally:
                env.close()
    complete = len(rows) == len(protocol["development_cases"])
    atomic_json(output / "summary.json", {"scope": "development", "policy": "rule_only_B3", "complete": complete,
        "scheduled": len(protocol["development_cases"]), "completed": len(rows), "episodes": rows,
        "safe_completion": sum(r["success"] for r in rows) / len(rows) if complete else None,
        "training_physics_steps": 0, "evaluation_physics_steps": sum(r["physics_steps"] for r in rows),
        "wall_seconds": time.time() - started})


if __name__ == "__main__":
    main()

"""Describe real native action/reward evidence; no counterfactual causal claims."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    if not values:
        return None
    return dict(zip(("min", "median", "p95", "max"),
                    map(float, np.quantile(values, [0, .5, .95, 1]))))


def analyze(batch, method):
    folder = batch / method
    summary = read(folder / "summary.json")
    audit_path = folder / "endpoint_raw_audit.json"
    audit = read(audit_path)
    if audit["summary_sha256"] != sha(folder / "summary.json"):
        raise ValueError("Endpoint audit does not cover this summary")
    episodes = {}
    with (folder / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            episodes[row["episode_id"]] = row
    groups = defaultdict(lambda: {"actions": 0, "controls": 0, "modes": Counter(),
        "plan_failures": Counter(), "reward": [], "potential": [], "progress_m": [],
        "intervention_controls": 0, "terminations": Counter()})
    files = sorted(folder.glob("transitions_*.jsonl.gz"))
    previous = {}
    total = 0
    for path in files:
        if audit["transition_hashes"][path.name] != sha(path):
            raise ValueError("Transition changed after raw audit")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                action = row["policy_action"]
                if isinstance(action, list):
                    # Direct continuous controls are described by longitudinal
                    # sign and steering magnitude, not artificial discrete IDs.
                    action = ("forward" if action[0] > 0 else "brake") + (
                        "/large_steer" if abs(action[1]) > .5 else "/small_steer")
                group = groups[str(action)]
                ticks = row["trace"]
                group["actions"] += 1
                group["controls"] += len(ticks)
                total += len(ticks)
                group["modes"][row["mode"]] += 1
                if row.get("plan") and row["plan"].get("failure_reason"):
                    group["plan_failures"][row["plan"]["failure_reason"]] += 1
                group["reward"].append(sum(t["reward"] for t in ticks))
                group["potential"].append(sum(t["reward_components"]["potential"] for t in ticks))
                group["intervention_controls"] += sum(t["intervention"] for t in ticks)
                episode_id = row["episode_id"]
                if episode_id in episodes:
                    initial = episodes[episode_id]["initial_scenario"]
                    before = previous.get(episode_id, initial["initial"])
                    after = ticks[-1]["state"]
                    goal = initial["goal"]
                    group["progress_m"].append(math.hypot(before["x"]-goal[0], before["y"]-goal[1]) -
                                               math.hypot(after["x"]-goal[0], after["y"]-goal[1]))
                previous[episode_id] = ticks[-1]["state"]
                if row["termination_reason"]:
                    group["terminations"][row["termination_reason"]] += 1
    if total != summary["physics_steps"] or total != audit["training_control_frames_verified"]:
        raise ValueError("Training control accounting mismatch")
    rows = []
    for action, group in sorted(groups.items()):
        result = {"action": action, **group}
        for name in ("reward", "potential", "progress_m"):
            values = group[name]
            result[name] = {"n": len(values), "mean": float(np.mean(values)) if values else None,
                            "quantiles": quantiles(values)}
        result["intervention_fraction"] = group["intervention_controls"] / group["controls"]
        rows.append(result)
    metrics = defaultdict(list)
    with (folder / "optimization.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            for key, value in json.loads(line)["metrics"].items():
                if key in ("train/entropy_loss", "train/approx_kl", "train/clip_fraction",
                           "train/policy_gradient_loss", "train/value_loss", "train/explained_variance"):
                    metrics[key].append(value)
    return {"method": method, "training_controls": total, "source_batch": batch.name,
            "raw_audit_sha256": sha(audit_path), "groups": rows,
            "optimization_metrics": {key: quantiles(value) for key, value in metrics.items()},
            "categorical_max_entropy": math.log(31) if read(folder / 'config.json')['env']['method'] != 'B1' else None,
            "training_terminations": dict(Counter(e["termination_reason"] for e in episodes.values())),
            "training_successes": sum(e["success"] for e in episodes.values()),
            "completed_episodes": len(episodes),
            "scope": "Observed conditional action outcomes on actually visited states. State distributions differ by action; these are not causal action values. Rewards are undiscounted physical sums; unfinished-episode progress is omitted. No new environment controls, teacher queries or policy updates."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous diagnostic")
    result = analyze(args.batch, args.method)
    result["script_sha256"] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, indent=2))

"""Re-score saved physical traces; never claim new policy performance."""
import argparse
import gzip
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--weight", type=float, default=.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.weight < 0 or not math.isfinite(args.weight):
        raise ValueError("Invalid progress weight")
    rows = []
    for filename in args.evaluation:
        path = Path(filename)
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(evaluation["checkpoint"])
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["env"]
        if config["reward"].get("progress_weight", 0) != 0:
            raise ValueError("Reference already contains an explicit progress reward")
        gamma = config["reward"]["gamma_per_second"] ** config["control_dt"]
        traces, scenes = {}, {}
        with gzip.open(path.with_name(path.stem + "_trace.jsonl.gz"), "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                traces.setdefault(row["case_id"], []).extend(row["info"]["trace"])
                scenes[row["case_id"]] = row["info"]["initial_scenario"]
        for episode in evaluation["episodes"]:
            case = episode["case_id"]
            scene = scenes[case]
            before = scene["initial"]
            goal = scene["goal"]
            old_return = progress_return = 0.
            for i, tick in enumerate(traces[case]):
                after = tick["state"]
                delta = math.hypot(before["x"]-goal[0], before["y"]-goal[1]) - math.hypot(after["x"]-goal[0], after["y"]-goal[1])
                old_return += gamma**i * tick["reward"]
                progress_return += gamma**i * args.weight * delta
                before = after
            if abs(old_return - episode["discounted_return"]) > 1e-6:
                raise ValueError("Physical reward trace does not reproduce saved return")
            rows.append({"run": path.parent.name, "milestone": path.stem, "case_id": case,
                         "scenario": episode["scenario"], "success": episode["success"],
                         "reason": episode["termination_reason"], "progress_m": episode["progress"],
                         "recorded_return": old_return, "added_progress_return": progress_return,
                         "counterfactual_reward_return": old_return + progress_return})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"scope": "offline reward-only sensitivity of unchanged recorded trajectories; not new policy results",
              "progress_weight": args.weight, "physical_returns_verified": len(rows), "rows": rows}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"verified_episodes": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()

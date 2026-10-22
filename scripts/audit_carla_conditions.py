"""Audit actual CARLA initial scenes: distinct conditions and matched geometry."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def vector(record):
    scene, state = record["scene"], record["initial_state"]
    initial = state["initial"]
    values = [initial[k] for k in ("x", "y", "yaw", "v")]
    values.extend(state["goal"])
    values.extend(np.asarray(state["static"]).reshape(-1).tolist())
    for obstacle in state["dynamic"]:
        values.extend(obstacle[k] for k in ("x", "y", "vx", "vy", "radius"))
    if scene["name"] == "dynamic":
        values.append(scene["walker_control_speed"])
    return np.asarray(values, dtype=float)


def audit(batch):
    batch = Path(batch)
    runs = {}
    details = []
    unavailable = []
    for file in sorted((batch / "carla_formal_evaluation_v5").glob("*/evaluation_summary.json")):
        summary = read(file)
        if not summary.get("complete") or summary.get("split") != "test":
            continue
        if summary.get("infrastructure_unresolved", 0):
            unavailable.append({"run":str(file.parent), "infrastructure_unresolved":summary["infrastructure_unresolved"],
                                "reason":"No full observed condition set; not an initialization equivalence result"})
            continue
        rows = [json.loads(line) for line in (file.parent / "scenes.jsonl").read_text(encoding="utf-8").splitlines()]
        # Later attempts overwrite earlier failed/restarted initialization records.
        scenes = {r["case_id"]: r for r in rows}
        unique = {}
        for name in ("static", "curve", "dynamic"):
            selected = [r for r in scenes.values() if r["scene"]["name"] == name]
            unique[name] = {"conditions": len(selected), "distinct_initial_condition_vectors_round_2dp":
                len({tuple(np.round(vector(r), 2)) for r in selected})}
        key = (summary["transfer"], summary["seed"], summary["method"])
        runs[key] = scenes
        details.append({"run": str(file.parent), "transfer": summary["transfer"], "seed": summary["seed"],
                        "method": summary["method"], "scenarios": unique})
    pairs = []
    for (transfer, seed, method), left in runs.items():
        if method != "H" or (transfer, seed, "B1") not in runs:
            continue
        right = runs[(transfer, seed, "B1")]
        if set(left) != set(right):
            raise ValueError("Paired completed CARLA runs have different condition IDs")
        differences = []
        for case in left:
            if left[case]["scene"]["condition"] != right[case]["scene"]["condition"] or left[case]["scene"]["name"] != right[case]["scene"]["name"]:
                raise ValueError("Paired condition identity mismatch")
            a, b = vector(left[case]), vector(right[case])
            if a.shape != b.shape:
                raise ValueError("Paired geometry dimension mismatch")
            differences.append({"case_id": case, "maximum_absolute_component_difference": float(max(abs(a-b)))})
        pairs.append({"transfer": transfer, "seed": seed, "paired_cases": len(left),
                      "maximum_absolute_component_difference": max(r["maximum_absolute_component_difference"] for r in differences),
                      "cases": differences})
    result = {"scope": "Observed initial state/geometry comparison, no policy-score filtering. Vector components include metres, radians and m/s; maximum is a numeric discrepancy check, not a distance metric. Condition distinctness uses 0.01 rounding for those components. Not a proof of deterministic future CARLA dynamics.",
              "runs": details, "paired_initial_conditions": pairs, "unavailable":unavailable}
    (batch / "analysis/carla_condition_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    result = audit(parser.parse_args().batch)
    print(json.dumps({"complete_test_runs": len(result["runs"]),
                      "pairs": [{k:v for k,v in r.items() if k != "cases"} for r in result["paired_initial_conditions"]]}))

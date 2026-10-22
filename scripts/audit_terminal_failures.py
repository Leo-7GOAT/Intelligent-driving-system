"""Classify goal-gate failures from saved physical frames, without rerunning policy."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def summarize_frames(frames, goal, config):
    """All success gates must hold on the same frame; safety is reported separately."""
    if not frames:
        raise ValueError("No physical frames: cannot infer goal behavior")
    distances = [math.hypot(f["state"]["x"]-goal[0], f["state"]["y"]-goal[1]) for f in frames]
    headings = [abs(math.atan2(math.sin(f["state"]["yaw"]-goal[2]),
                               math.cos(f["state"]["yaw"]-goal[2]))) for f in frames]
    inside = [d <= config["goal_tolerance"] for d in distances]
    slow = [f["state"]["v"] <= config["goal_speed"] for f in frames]
    aligned = [h <= config["goal_yaw_tolerance"] for h in headings]
    geometric = [a and (not config["parking"] or (b and c)) for a, b, c in zip(inside, slow, aligned)]
    late = [i for i, f in enumerate(frames) if f["state"]["time"] >= frames[-1]["state"]["time"]-5]
    closest = min(range(len(frames)), key=distances.__getitem__)
    return {"physical_frames": len(frames), "minimum_goal_distance_m": min(distances),
            "final_goal_distance_m": distances[-1], "final_heading_error_rad": headings[-1],
            "final_speed_m_s": frames[-1]["state"]["v"],
            "ever_in_position_tolerance": any(inside),
            "goal_frames_above_speed_limit": sum(a and not b for a, b in zip(inside, slow)),
            "goal_frames_outside_heading_limit": sum(a and not c for a, c in zip(inside, aligned)),
            "simultaneous_goal_gate_frames": sum(geometric),
            "closest_frame": {"time_s": frames[closest]["state"]["time"],
                              "speed_m_s": frames[closest]["state"]["v"], "heading_error_rad": headings[closest]},
            "last5s_speed_mean_m_s": sum(frames[i]["state"]["v"] for i in late)/len(late),
            "last5s_requested_braking_fraction": sum(frames[i]["requested_command"]["acceleration"] < 0 for i in late)/len(late)}


def audit(batch, job_id, target):
    batch = Path(batch).resolve()
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    job = next(j for j in protocol["jobs"] if j["id"] == job_id)
    run = batch / protocol["run_group"] / job_id
    validation = run / f"validation_{target}.json"
    trace = run / f"validation_{target}_trace.jsonl.gz"
    result = json.loads(validation.read_text(encoding="utf-8"))
    grouped, goals = {}, {}
    with gzip.open(trace, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            grouped.setdefault(row["case_id"], []).extend(row["info"]["trace"])
            goals[row["case_id"]] = row["info"]["initial_scenario"]["goal"]
    episodes = []
    for episode in result["episodes"]:
        frames = grouped[episode["case_id"]]
        if len(frames) != episode["physics_steps"]:
            raise ValueError("Physical trace does not cover the complete episode")
        row = {k: episode[k] for k in ("case_id", "scenario", "seed", "success", "termination_reason", "progress")}
        row.update(summarize_frames(frames, goals[episode["case_id"]], job["env"]))
        if row["success"] and not row["simultaneous_goal_gate_frames"]:
            raise ValueError("Recorded success lacks a simultaneous goal gate")
        episodes.append(row)
    output = {"scope": "Offline audit of existing development physical traces, not new evaluation or independent seeds. Goal gates exclude safety; recorded outcome retains collision/bounds priority.",
              "batch": batch.name, "job": job_id, "target": target, "model_sha256": result["model_sha256"],
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (validation, trace)},
              "goal_thresholds": {k: job["env"][k] for k in ("goal_tolerance", "goal_speed", "goal_yaw_tolerance", "parking")},
              "episodes": episodes}
    destination = run / f"terminal_audit_{target}.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != output:
            raise ValueError("Existing audit differs; preserve it and investigate")
    else:
        destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "episodes": episodes}))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    audit(args.batch, args.job, args.target)

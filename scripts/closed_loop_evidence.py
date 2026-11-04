"""Extract an auditable link from saved PPO updates to physical CARLA frame/control logs."""

import argparse, gzip, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.experiments import write_json


def evidence(evaluation):
    run = Path(evaluation)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    checkpoint = Path(cfg["checkpoint"])
    training = json.loads(
        (checkpoint / "training_summary.json").read_text(encoding="utf-8")
    )
    episodes = [
        json.loads(s)
        for s in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    attempts = {
        r["case_id"]: r["attempt"] for r in episodes if r.get("status") == "completed"
    }
    collected = {}
    for file in sorted(run.glob("transitions*.jsonl.gz")):
        per_case = defaultdict(
            lambda: {
                "frames": [],
                "positions": [],
                "actions": Counter(),
                "controller_status": Counter(),
                "nonzero_actuator_ticks": 0,
                "collision_events": [],
            }
        )
        with gzip.open(file, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                case = row["case_id"]
                if row["attempt"] != attempts.get(case):
                    continue
                r = per_case[case]
                r["actions"][json.dumps(row["policy_action"])] += 1
                for tick in row["trace"]:
                    state = tick["state"]
                    backend = tick["backend_evidence"]
                    assert backend and backend["frame"] == tick["backend_frame"]
                    r["frames"].append(backend["frame"])
                    r["positions"].append((state["x"], state["y"]))
                    r["controller_status"][tick["controller_status"]] += 1
                    u = backend.get("vehicle_control", {})
                    r["nonzero_actuator_ticks"] += any(
                        abs(v) > 1e-8 for v in u.values()
                    )
                    r["collision_events"].extend(backend["collisions"])
        collected.update(per_case)
    cases = []
    for case, r in sorted(collected.items()):
        frames = r.pop("frames")
        positions = r.pop("positions")
        continuous = all(b - a == 1 for a, b in zip(frames, frames[1:]))
        if not continuous:
            raise ValueError(
                "Nonconsecutive physical frames inside a completed episode"
            )
        cases.append(
            {
                "case_id": case,
                "measured_control_ticks": len(frames),
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "frames_consecutive": continuous,
                "first_position": positions[0],
                "last_position": positions[-1],
                "displacement_m": math.dist(positions[0], positions[-1]),
                **r,
            }
        )
    return {
        "evaluation": str(run),
        "checkpoint": str(checkpoint),
        "rng_preserving_checkpoint_training": checkpoint.parent.parent.name in ("formal_v5", "carla_formal_training_v5", "carla_v5_100k_source_finetune", "carla_v5_100k_source_finetune_warmup1"),
        "model_sha256": hashlib.sha256(
            (checkpoint / "model.zip").read_bytes()
        ).hexdigest(),
        "method": cfg["env"]["method"],
        "policy": "learned_PPO",
        "transfer": cfg["transfer"],
        "training_physics_steps": training["physics_steps"],
        "optimization_phases": training["optimization_phases"],
        "actor_l2_change": training["actor_l2_change"],
        "reload_exact": training["reload_exact"],
        "cases": cases,
        "interpretation": "Logged PPO actions feed planning/MPC or direct control; controls and frame states come from the actual CARLA bridge. This is execution evidence, not convergence or superiority evidence.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    args = p.parse_args()
    batch = Path(args.batch)
    records = []
    for file in (batch / "carla").glob("*/evaluation_summary.json"):
        row = json.loads(file.read_text(encoding="utf-8"))
        if row["transfer"] == "carla_finetuned":
            records.append(evidence(file.parent))
    for file in (batch / "carla_formal_evaluation_v5").glob("*/evaluation_summary.json"):
        if json.loads(file.read_text(encoding="utf-8")).get("complete"):
            records.append(evidence(file.parent))
    for file in (batch / "carla_v5_100k_development").glob("*/evaluation_summary.json"):
        if json.loads(file.read_text(encoding="utf-8")).get("complete"):
            records.append(evidence(file.parent))
    for file in (batch / "carla_v5_100k_source_finetune").glob("*/evaluation_summary.json"):
        if json.loads(file.read_text(encoding="utf-8")).get("complete"):
            records.append(evidence(file.parent))
    for file in (batch / "carla_v5_100k_source_finetune_warmup1").glob("*/evaluation_summary.json"):
        if json.loads(file.read_text(encoding="utf-8")).get("complete"):
            records.append(evidence(file.parent))
    write_json(batch / "ppo_closed_loop_evidence.json", records)
    print(
        json.dumps(
            {
                "evaluations": len(records),
                "verified_control_ticks": sum(
                    c["measured_control_ticks"] for r in records for c in r["cases"]
                ),
            }
        )
    )

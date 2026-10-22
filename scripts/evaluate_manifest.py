"""One fixed evaluation manifest, immutable model, frozen normalization and bounded wall time."""

import argparse, gzip, hashlib, json, sys, time
from pathlib import Path
import numpy as np


def main(args):
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv, rule_action
    from navstack.experiments import (
        load_run,
        new_run,
        write_json,
        append_json,
        jsonable,
    )

    checkpoint = Path(args.checkpoint).resolve()
    batch = Path(args.batch).resolve()
    marker = checkpoint.parent / ("final_" + args.label + ".json")
    previous = (
        json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    )
    if previous and previous.get("complete"):
        raise RuntimeError(
            "A completed result marker already exists; refuse an unregistered repeat before running any episode"
        )
    metadata = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    training = json.loads(
        (checkpoint / "training_summary.json").read_text(encoding="utf-8")
    )
    if args.split == "test":
        if not (batch / "development_freeze.json").exists():
            raise RuntimeError("Development scheme must be frozen before final test")
        if training["physics_steps"] < 1000000:
            raise RuntimeError("Primary final test requires the actual 1M milestone")
    config = EnvConfig.from_dict(metadata["env"])
    config.split = args.split
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    case_hash = hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()
    if args.split == "test":
        freeze = json.loads(
            (batch / "development_freeze.json").read_text(encoding="utf-8")
        )
        if checkpoint.parent.parent.name not in freeze["registered_source_versions"]:
            raise ValueError("Checkpoint is not from the registered confirmation version")
        if case_hash not in freeze["manifest_hashes"].values():
            raise ValueError("Unregistered held-out manifest")
    if previous and (
        previous["manifest_sha256"] != case_hash
        or previous["checkpoint"] != str(checkpoint)
    ):
        raise ValueError("Resume must preserve the exact checkpoint and case manifest")
    run = (
        Path(previous["run"])
        if previous
        else new_run(checkpoint.parent, f"fixed_{args.split}_{args.label}")
    )
    write_json(
        run / "config.json",
        {
            "env": config.to_dict(),
            "checkpoint": str(checkpoint),
            "manifest": cases,
            "manifest_sha256": case_hash,
            "label": args.label,
            "seed": metadata["seed"],
            "inference": "deterministic",
        },
    )
    deadline = (
        json.loads((batch / "batch.json").read_text(encoding="utf-8"))["deadline_epoch"]
        - 60
    )
    model, normalizer = load_run(checkpoint)
    means = normalizer.obs_rms.mean.copy()
    started = time.time()
    records = (
        [
            json.loads(line)
            for line in (run / "episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if (run / "episodes.jsonl").exists()
        else []
    )
    done_cases = {r["case_id"] for r in records}
    if len(done_cases) != len(records):
        raise ValueError("Duplicate completed case IDs in stored evidence")
    write_json(
        marker,
        {
            "run": str(run),
            "complete": False,
            "checkpoint": str(checkpoint),
            "manifest_sha256": case_hash,
            "completed_cases": len(records),
            "status": "running",
        },
    )
    try:
        with gzip.open(
            run / f"transitions_{time.time_ns()}.jsonl.gz", "wt", encoding="utf-8"
        ) as stream:
            for index, case in enumerate(cases):
                if index in done_cases:
                    continue
                if time.time() > deadline or (batch / "STOP").exists():
                    break
                c = EnvConfig.from_dict(config.to_dict())
                c.scenario = case["scenario"]
                for key, value in case.get("overrides", {}).items():
                    if key == "vehicle":
                        for field, val in value.items():
                            setattr(c.vehicle, field, val)
                    else:
                        setattr(c, key, value)
                env = NavigationEnv(c)
                obs, _ = env.reset(seed=case["seed"])
                append_json(
                    run / "scenes.jsonl",
                    {
                        "case_id": index,
                        "case": case,
                        "initial_scenario": env.initial_scenario,
                    },
                )
                while not env.done:
                    action = model.predict(
                        normalizer.normalize_obs(obs), deterministic=True
                    )[0]
                    obs, _, _, _, info = env.step(action)
                    stream.write(
                        json.dumps(
                            {
                                "case_id": index,
                                "action": action,
                                "mode": info["mode"],
                                "plan": info["plan"],
                                "trace": info["trace"],
                            },
                            default=jsonable,
                        )
                        + "\n"
                    )
                    if time.time() > deadline or (batch / "STOP").exists():
                        break
                if not env.done:
                    append_json(
                        run / "censored.jsonl",
                        {
                            "case_id": index,
                            "reason": "batch_budget",
                            "physics_steps": env.episode["physics_steps"],
                        },
                    )
                    break
                row = {"case_id": index, "case": case, **info["episode_metrics"]}
                records.append(row)
                append_json(run / "episodes.jsonl", row)
                print(
                    json.dumps(
                        {
                            "run": str(run),
                            "case_id": index,
                            "reason": row["termination_reason"],
                            "progress": row["progress"],
                        }
                    ),
                    flush=True,
                )
                env.close()
        complete = len(records) == len(cases)
        summary = {
            "run": str(run),
            "method": config.method,
            "seed": metadata["seed"],
            "split": args.split,
            "label": args.label,
            "training_physics_steps": training["physics_steps"],
            "manifest_sha256": case_hash,
            "scheduled_cases": len(cases),
            "completed_cases": len(records),
            "complete": complete,
            "success_rate": (
                float(np.mean([r["success"] for r in records])) if complete else None
            ),
            "completed_case_successes": sum(r["success"] for r in records),
            "failure_counts": {
                event: sum(r["termination_reason"] == event for r in records)
                for event in ("collision", "out_of_bounds", "deadline")
            },
            "mean_progress_m": (
                float(np.mean([r["progress"] for r in records])) if records else None
            ),
            "physics_steps": sum(r["physics_steps"] for r in records),
            "session_wall_seconds": time.time() - started,
            "normalization_frozen": bool(
                np.array_equal(means, normalizer.obs_rms.mean)
            ),
            "checkpoint": str(checkpoint),
        }
        write_json(run / "evaluation_summary.json", summary)
        write_json(marker, summary)
        print(json.dumps(summary), flush=True)
    finally:
        normalizer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--label", default="primary")
    main(p.parse_args())

"""Validation-only checkpoint selection within one independent training seed."""

import json
from pathlib import Path
import shutil
from .config import EnvConfig
from .experiments import new_run, evaluate, write_json


def development_rank(records, maximum_failure_rate=0.05):
    """Maximize completion among policies meeting the development safety limit.

    This is an empirical selection rule, not a statistical safety guarantee.
    Missing successful travel time is worst, never treated as fast travel.
    """
    if not records or not 0 <= maximum_failure_rate <= 1:
        raise ValueError("Complete validation records and a valid safety threshold required")
    safety = sum(r["termination_reason"] in ("collision", "out_of_bounds") for r in records) / len(records)
    success = sum(r["success"] for r in records) / len(records)
    progress = sum(r["progress"] for r in records) / len(records)
    interventions = sum(r["intervention_rate"] for r in records) / len(records)
    times = [r["travel_time_success_only"] for r in records if r["success"]]
    travel = sum(times) / len(times) if times else 1e10
    if safety <= maximum_failure_rate:
        return (0, -success, -progress, interventions, travel, safety)
    return (1, safety, -success, -progress, interventions, travel)


def select_checkpoints(candidates, episodes=16, root="runs", maximum_failure_rate=0.05):
    candidates = [Path(c) for c in candidates]
    if not candidates:
        raise ValueError("at least one checkpoint required")
    metadata = [
        json.loads((p / "config.json").read_text(encoding="utf-8")) for p in candidates
    ]
    if len({m["seed"] for m in metadata}) != 1:
        raise ValueError("cannot choose the best training seed")
    if any(m["env"] != metadata[0]["env"] for m in metadata[1:]):
        raise ValueError("candidate task/configs differ")
    destination = new_run(root, "validation_selection")
    results = []
    for path, m in zip(candidates, metadata):
        config = EnvConfig.from_dict(m["env"])
        config.split = "validation"
        run, summary = evaluate(
            config, path, episodes, destination / "validation", seed=91
        )
        records = [
            json.loads(line)
            for line in (run / "episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if len(records) != episodes:
            raise ValueError("Incomplete validation cannot select a checkpoint")
        rank = development_rank(records, maximum_failure_rate)
        results.append(
            {"candidate": str(path), "validation_run": str(run), "rank": rank}
        )
    winner = min(results, key=lambda r: r["rank"])
    for name in ("model.zip", "normalization.pkl", "config.json"):
        shutil.copy2(Path(winner["candidate"]) / name, destination / name)
    write_json(
        destination / "selection.json",
        {
            "rule": "v6: safety constraint first; among feasible policies maximize completion, progress, then minimize interventions and success-only time",
            "maximum_empirical_collision_or_bounds_rate": maximum_failure_rate,
            "safety_constraint_met": winner["rank"][0] == 0,
            "results": results,
            "selected": winner,
            "test_set_used": False,
        },
    )
    print(destination)
    return destination

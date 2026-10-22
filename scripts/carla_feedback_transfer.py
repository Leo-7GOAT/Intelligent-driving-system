"""Paired learned H/H0 transfer on previously inspected CARLA development cases."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.experiments import write_json
from carla_evaluate import evaluate_carla


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-parent")
    group.add_argument("--batch")
    args = parser.parse_args()
    if args.prepare_parent:
        parent = Path(args.prepare_parent).resolve()
        batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_carla_v8_transfer")
        batch.mkdir()
        (batch / "infrastructure").mkdir()
        shutil.copytree(parent / "calibration", batch / "calibration")
        jobs = []
        for label in ("H", "H0"):
            run = parent / "hierarchy_v8" / (label + "_seed11")
            completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
            checkpoint = Path(completed["summary"]["checkpoint"])
            jobs.append({"label": label, "checkpoint": str(checkpoint),
                         "training_physics_steps": completed["summary"]["physics_steps"],
                         "model_sha256": hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest()})
        hashes = {}
        for source in [*(ROOT / "navstack").glob("*.py"), *(ROOT / "scripts").glob("*.py")]:
            target = batch / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(batch / "batch.json", {"deadline_epoch": time.time()+900, "jobs": jobs,
                   "source_hashes": hashes, "scope": "Frozen learned policies with calibrated vehicle; development only; zero CARLA PPO updates",
                   "curve_scope": "curved obstacle route on existing straight roadbed, not native curved road",
                   "resource_limits": {"wall_seconds": 900, "carla_owners": 1, "infrastructure_retries_per_case": 1}})
        write_json(batch / "cases.json", [{"scenario": scene, "condition": condition}
                                          for condition in (7301, 7302) for scene in ("static", "curve", "dynamic")])
        write_json(ROOT / "runs/carla_transfer_development.json", {"batch": str(batch)})
        print(batch)
        return
    batch = Path(args.batch).resolve()
    protocol = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    for name, digest in protocol["source_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, "Source changed before transfer: " + name
    results = []
    try:
        for job in protocol["jobs"]:
            checkpoint = Path(job["checkpoint"])
            assert hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest() == job["model_sha256"]
            run, summary = evaluate_carla(batch, checkpoint, batch / "cases.json",
                                         label=job["label"]+"_calibrated_vehicle_frozen_policy",
                                         run=batch / "carla" / job["label"])
            results.append({"label": job["label"], **summary})
        write_json(batch / "paired_summary.json", {"ppo_updates": 0, "results": results})
    except BaseException as exc:
        write_json(batch / "failure.json", {"error": repr(exc), "completed_methods": len(results)})
        raise


if __name__ == "__main__":
    main()

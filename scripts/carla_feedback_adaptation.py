"""Versioned, interleaved H/H0 CARLA adaptation after paired frozen transfer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.experiments import write_json
from navstack.research import atomic_json
from navstack.process_lease import ProcessLease
from carla_finetune_pair import finetune
from carla_evaluate import evaluate_carla


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(parent):
    parent = Path(parent).resolve()
    transfer = read(parent / "batch.json")
    completed = read(parent / "paired_summary.json")
    if not all(r["complete"] for r in completed["results"]):
        raise ValueError("Complete both frozen-transfer evaluations first")
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_carla_v10_adaptation")
    batch.mkdir()
    (batch / "infrastructure").mkdir()
    shutil.copytree(parent / "calibration", batch / "calibration")
    shutil.copy2(parent / "cases.json", batch / "cases.json")
    hashes = {}
    for folder in ("navstack", "scripts", "configs", "tests"):
        for source in (ROOT / folder).rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix in (".py", ".ps1", ".json"):
                rel = source.relative_to(ROOT)
                target = batch / "source" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                hashes[rel.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    now = time.time()
    protocol = {"version": "carla_v10_feedback_adaptation", "created_epoch": now,
                "deadline_epoch": now+7200, "source_hashes": hashes, "run_group": "carla_adaptation",
                "jobs": [{"id": "paired_adaptation", "worker_script": "scripts/carla_feedback_adaptation.py"}],
                "models": transfer["jobs"], "milestones": [25000, 100000], "parent_transfer": str(parent),
                "scope": "Single-seed development adaptation, not final confirmation or a healthy H/B1 comparison",
                "budget": "Each method gets cumulative real CARLA world ticks INCLUDING resets; control ticks and lightweight source cost reported separately",
                "selection": "Evaluate both fixed milestones on all six existing development cases; do not select favorable scenes",
                "training_cases": "CARLA static/curve/dynamic, condition integers0..9; default seeded generator",
                "ppo": "Retain source optimizer/hyperparameters; no action replacement, masking, or teacher",
                "client_python": str((ROOT / ".carla-python37/runtime/python.exe").resolve()),
                "resource_limits": {"cpu_workers": 1, "disk_min_GiB": 12, "ram_min_GiB": 4,
                                    "gpu_stop_temperature_C": 85, "no_progress_seconds": 300,
                                    "automatic_failure_retries": 0}}
    write_json(batch / "protocol.json", protocol)
    write_json(batch / "batch.json", protocol)
    write_json(ROOT / "runs/carla_adaptation_development.json", {"batch": str(batch)})
    print(batch)


def execute(batch, job):
    batch = Path(batch).resolve()
    protocol = read(batch / "protocol.json")
    if job != "paired_adaptation":raise ValueError("Unknown job")
    coordinator = batch / "carla_adaptation" / job
    coordinator.mkdir(parents=True, exist_ok=True)
    os.environ["NAVSTACK_CARLA_PYTHON"] = protocol["client_python"]
    with ProcessLease(coordinator / "worker.lock"):
        if (coordinator / "completed.json").exists():return
        if (coordinator / "failure.json").exists():raise RuntimeError("Inspect prior failure before retry")
        try:
            for target in protocol["milestones"]:
                for model in protocol["models"]:
                    if time.time() >= protocol["deadline_epoch"]-180 or (batch / "STOP").exists():return
                    label = model["label"]
                    checkpoint = Path(model["checkpoint"])
                    if hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest() != model["model_sha256"]:
                        raise ValueError("Source model hash changed")
                    state = {"pid": os.getpid(), "epoch": time.time(), "label": label, "target": target, "stage": "training"}
                    atomic_json(coordinator / "worker_status.json", state)
                    training = finetune(batch, checkpoint, target, 11, run=batch / "carla_adaptation" / label)
                    if training["physics_steps"] < target:return
                    state.update(stage="evaluating", epoch=time.time())
                    atomic_json(coordinator / "worker_status.json", state)
                    output = batch / "evaluations" / (label + "_" + str(target))
                    if (output / "evaluation_summary.json").exists():
                        result = read(output / "evaluation_summary.json")
                        if not result["complete"]:raise RuntimeError("Incomplete evaluation requires diagnosis")
                    else:
                        _, result = evaluate_carla(batch, training["checkpoint"], batch / "cases.json",
                                                  label=label+"_actual_CARLA_adaptation_"+str(target), run=output)
                    atomic_json(coordinator / (label+"_"+str(target)+".json"), {"label": label, "training": training, "evaluation": result})
            atomic_json(coordinator / "completed.json", {"epoch": time.time(), "scope": protocol["scope"]})
        except BaseException as exc:
            atomic_json(coordinator / "failure.json", {"error": repr(exc), "epoch": time.time()})
            raise
        finally:
            # The bridge closes before this point. The script probes actor count
            # and exact server process identity; it refuses foreign ownership.
            cleanup = subprocess.run(["powershell.exe", "-NoProfile", "-File", str(ROOT / "scripts/stop_owned_carla.ps1"),
                                      "-Batch", str(batch)], capture_output=True, text=True,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=60)
            write_json(coordinator / "server_cleanup.json", {"returncode": cleanup.returncode,
                       "stdout": cleanup.stdout, "stderr": cleanup.stderr})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-transfer")
    group.add_argument("--batch")
    parser.add_argument("--job", default="paired_adaptation")
    args = parser.parse_args()
    if args.prepare_transfer:prepare(args.prepare_transfer)
    else:execute(args.batch, args.job)

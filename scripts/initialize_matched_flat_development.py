"""Train flat PPO on the H development task with the same feedback information."""
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
from navstack.config import EnvConfig
from navstack.experiments import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()
    reference = Path(args.reference).resolve()
    parent = json.loads((reference / "protocol.json").read_text(encoding="utf-8"))
    high = next(job for job in parent["jobs"] if job["id"] == "H_seed11")
    config = EnvConfig.from_dict(high["env"])
    if config.execution_feedback != "enabled":raise ValueError("Expected execution-aware H reference")
    config.method = "B1"
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_matched_flat_v11_development")
    batch.mkdir()
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
    job = {"id": "B1_feedback_seed11", "seed": 11, "env": config.to_dict(),
           "worker_script": "scripts/hierarchy_development.py", "milestones": [0, 100000, 300000, 1000000],
           "ppo_options": {"target_kl": .03, "use_sde": True, "sde_sample_freq": 10,
                           "log_std_init": -2, "forward_mean_bias": .2, "entropy": .001}}
    protocol = {"version": "matched_flat_v11_development", "created_epoch": now, "deadline_epoch": now+7200,
                "run_group": "matched_flat_v11", "jobs": [job], "source_hashes": hashes,
                "development_cases": parent["development_cases"], "reference_H_batch": str(reference),
                "scope": "Single-seed from-scratch development, not independent confirmation",
                "fairness": "Same mixed training distribution, reward, task deadline, vehicle and full450observation including candidate geometry/previous execution. Flat outputs direct control; no planner/controller generates its commands. Native decision frequency difference retained and must be disclosed.",
                "rationale": "Fixed23m flat sanity learned long-route parking; now test its healthy hyperparameters on the common mixed task, without hidden warm start",
                "compute_disclosure": "Candidate geometry features recomputed at each flat observation; full decision time includes this additional cost",
                "resource_limits": {"cpu_workers": 1, "disk_min_GiB": 12, "ram_min_GiB": 4,
                                    "gpu_stop_temperature_C": 85, "no_progress_seconds": 300,
                                    "automatic_failure_retries": 0}}
    write_json(batch / "protocol.json", protocol)
    write_json(ROOT / "runs/matched_flat_development.json", {"batch": str(batch)})
    print(batch)


if __name__ == "__main__":
    main()

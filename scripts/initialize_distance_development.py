"""Freeze B1 full-distance vs variable-distance learning from scratch."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig, PlannerConfig


def main():
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_distance_v9_development")
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
    jobs = []
    for label, route_range in (("full23", None), ("uniform6to23", [6, 23])):
        config = EnvConfig(method="B1", scenario="empty", deadline=45, obstacle_num=0,
                           training_route_range=route_range, planner=PlannerConfig(max_seconds=None))
        config.reward.beta = 40
        jobs.append({"id": "B1_" + label + "_seed11", "seed": 11, "env": config.to_dict(),
                     "worker_script": "scripts/hierarchy_development.py", "milestones": [0, 100000, 300000, 1000000],
                     "ppo_options": {"target_kl": .03, "use_sde": True, "sde_sample_freq": 10,
                                     "log_std_init": -2, "forward_mean_bias": .2, "entropy": .001}})
    now = time.time()
    protocol = {"version": "distance_v9_development", "run_group": "distance_v9", "created_epoch": now,
                "deadline_epoch": now + 7200, "jobs": jobs, "source_hashes": hashes, "not_confirmation": True,
                "purpose": "Diagnose full-route learning versus distance randomization; both fresh B1 models, 45s deadline, shared beta40 PBRS",
                "fairness": "One-seed baseline development only, not H/B1 superiority evidence. No old checkpoint or hidden warm-start budget.",
                "development_cases": [{"scenario": "empty", "seed": seed, "route_m": length,
                                       "options": {"diagnostic_route_length": length}}
                                      for length in (6, 12, 18, 23) for seed in (7301, 7302)],
                "resource_limits": {"cpu_workers": 2, "disk_min_GiB": 12, "ram_min_GiB": 4,
                                    "gpu_stop_temperature_C": 85, "no_progress_seconds": 300,
                                    "automatic_failure_retries": 0}}
    (batch / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (ROOT / "runs/distance_development.json").write_text(json.dumps({"batch": str(batch)}, indent=2), encoding="utf-8")
    print(batch)


if __name__ == "__main__":
    main()

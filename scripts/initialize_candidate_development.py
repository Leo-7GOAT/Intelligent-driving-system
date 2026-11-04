"""Freeze a shared-candidate PPO / feedback ablation, trained from scratch."""
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
from navstack.config import EnvConfig, PlannerConfig
from navstack.experiments import write_json


def reference_protocol(reference=None):
    if reference is not None:
        return json.loads((Path(reference) / "protocol.json").read_text(encoding="utf-8"))
    cfg = EnvConfig(method="H", deadline=45, obstacle_num=0, subgoal_mode="goal_frame_terminal",
                    execution_feedback="enabled", terminal_goal_prefix=True,
                    planner=PlannerConfig(max_seconds=None, analytic_expansion=True, vectorized_primitives=True))
    cfg.reward.beta = 40
    return {"jobs": [{"id": "H_seed11", "env": cfg.to_dict()}],
            "development_cases": [{"scenario": scenario, "seed": seed} for seed in (7301, 7302)
                                  for scenario in ("empty", "static", "narrow", "dynamic", "unseen")]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", help="Optional local reference protocol; default is the documented v8 task")
    args = parser.parse_args()
    reference = Path(args.reference).resolve() if args.reference else None
    parent = reference_protocol(reference)
    previous = next(j for j in parent["jobs"] if j["id"] == "H_seed11")
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_candidate_v12_development")
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
    for label, feedback in (("H", "enabled"), ("H0", "ablated")):
        cfg = EnvConfig.from_dict(previous["env"])
        cfg.candidate_observation = True
        cfg.execution_feedback = feedback
        jobs.append({"id": label + "_seed11", "seed": 11, "env": cfg.to_dict(),
                     "worker_script": "scripts/hierarchy_development.py",
                     "milestones": [0, 25000, 100000, 300000, 1000000],
                     "ppo_options": {"policy": "candidate_shared", "target_kl": .03}})
    now = time.time()
    protocol = {"version": "candidate_v12_development", "created_epoch": now, "deadline_epoch": now+7200,
                "run_group": "candidate_v12", "jobs": jobs, "source_hashes": hashes,
                "development_cases": parent["development_cases"], "reference_batch": str(reference) if reference else "built_in_v8_task",
                "not_confirmation": True,
                "hypothesis": "Shared scoring of explicit candidate geometry improves generalization of action-feasibility associations; H/H0 isolate witness plus previous-outcome information with the same architecture.",
                "changes": "605 observation; fixed candidate scales bypass per-slot running normalization; shared actor scorer; unchanged reward, low level, actions, scene distribution and PPO hyperparameters",
                "limitations": "Single seed and ten near-identical existing development conditions. H/v8 comparison changes both architecture and geometry exposure; cannot attribute that contrast to feedback alone. Matched flat and strong rules must receive identical derived information for formal comparison.",
                "no_teacher_or_mask": True,
                "resource_limits": {"cpu_workers": 2, "disk_min_GiB": 12, "ram_min_GiB": 4,
                                    "gpu_stop_temperature_C": 85, "no_progress_seconds": 300,
                                    "automatic_failure_retries": 0}}
    write_json(batch / "protocol.json", protocol)
    write_json(ROOT / "runs/candidate_development.json", {"batch": str(batch)})
    print(batch)


if __name__ == "__main__":
    main()

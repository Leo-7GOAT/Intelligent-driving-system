"""Matched605-input H/H0/B1 test of full-length basic-task pretraining."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import write_json
from navstack.research import resources_ok
from initialize_candidate_development import reference_protocol


def main():
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError("Resource guard: " + str(reason))
    reference = reference_protocol()
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_staged_v13_development")
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
    for label, method, feedback in (("H", "H", "enabled"), ("H0", "H", "ablated"), ("B1", "B1", "enabled")):
        cfg = EnvConfig.from_dict(reference["jobs"][0]["env"])
        cfg.method = method
        cfg.candidate_observation = True
        cfg.execution_feedback = feedback
        cfg.empty_pretraining_steps = 300000
        options = ({"policy": "candidate_shared", "target_kl": .03} if method != "B1" else
                   {"target_kl": .03, "use_sde": True, "sde_sample_freq": 10,
                    "log_std_init": -2, "forward_mean_bias": .2, "entropy": .001})
        jobs.append({"id": label+"_seed11", "seed": 11, "env": cfg.to_dict(), "ppo_options": options,
                     "worker_script": "scripts/hierarchy_development.py", "milestones": [0, 100000, 300000, 1000000]})
    now = time.time()
    write_json(batch / "protocol.json", {"version": "staged_v13_development", "created_epoch": now,
        "deadline_epoch": now+7200, "run_group": "staged_v13", "jobs": jobs, "source_hashes": hashes,
        "development_cases": reference["development_cases"], "not_confirmation": True,
        "rationale": "v11 mixed-task B1 has zero training successes in1505episodes/1Msteps while v9 full23m-empty training learns parking. Test whether common full-length basic-task pretraining supplies a useful initialization before mixed tasks.",
        "hypothesis": "First300k full23m empty task, then original mixed distribution at episode boundaries; actual physical cost includes pretraining. All methods start from scratch and share this schedule.",
        "fixed": "Reward, dynamics, low level, action spaces, PPO hyperparameters and development cases unchanged. H/H0 share candidate scorer; B1 outputs direct controls and receives identical605inputs to H.",
        "limitations": "Single-seed development only. v13vsv12 changes curriculum; B1vsv11 also receives explicit geometry. No five-seed, final-test or independent novelty claim.",
        "resource_limits": {"cpu_workers": 3, "disk_min_GiB": 12, "ram_min_GiB": 4,
                            "gpu_stop_temperature_C": 85, "no_progress_seconds": 300, "automatic_failure_retries": 0},
        "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write_json(ROOT / "runs/staged_development.json", {"batch": str(batch)})
    print(batch)


if __name__ == "__main__":
    main()

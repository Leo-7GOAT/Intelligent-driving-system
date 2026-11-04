"""Isolate a shared candidate-set encoder in the direct-control PPO baseline."""
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
    config = EnvConfig.from_dict(reference["jobs"][0]["env"])
    config.method = "B1"
    config.candidate_observation = True
    config.empty_pretraining_steps = 300000
    config.flat_hold = 10
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_flat_pool_v15_development")
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
    job = {"id": "B1_pool_hold10_seed11", "seed": 11, "env": config.to_dict(),
           "worker_script": "scripts/hierarchy_development.py", "milestones": [0, 100000, 300000, 1000000],
           "ppo_options": {"policy": "flat_candidate_pool", "target_kl": .03, "use_sde": True,
                           "sde_sample_freq": 1, "log_std_init": -2, "forward_mean_bias": .2, "entropy": .001}}
    write_json(batch / "protocol.json", {"version": "flat_pool_v15_development", "created_epoch": now,
        "deadline_epoch": now + 7200, "run_group": "flat_pool_v15", "jobs": [job], "source_hashes": hashes,
        "development_cases": reference["development_cases"], "not_confirmation": True,
        "hypothesis": "A learned shared encoding and mean/max pooling of candidate features may make the605-input direct-control baseline easier to train than raw concatenation.",
        "evidence": "At300k v13/v14 fail both23m empty cases; v14 lowers jerk without completion. v13 first-layer tanh saturation is17.2percent versus4.2percent for the earlier351-input healthy23m model. Unexecuted feature-zeroing changes action means substantially; this motivates but does not prove the representation hypothesis.",
        "comparison": "v14 hold10 seed11: identical task, observations, normalization, reward, curriculum, action space, direct execution, critic architecture, PPO hyperparameters, physical budget and evaluation cases; only actor architecture changes",
        "actor": "357-global→64→64; each of31candidate8-vectors→32→16 shared; mean/max pool; concatenate with global64 then96→64; standard SB3 continuous action head/gSDE",
        "scope": "Single-seed development; different initialization/parameter count is disclosed. No new information, teacher, action mask, planner or MPC control replacement. Original failed baselines retained.",
        "resource_limits": {"cpu_workers": 1, "disk_min_GiB": 12, "ram_min_GiB": 4,
                            "gpu_stop_temperature_C": 85, "no_progress_seconds": 300, "automatic_failure_retries": 0},
        "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write_json(ROOT / "runs/flat_pool_development.json", {"batch": str(batch)})
    print(batch)


if __name__ == "__main__":
    main()

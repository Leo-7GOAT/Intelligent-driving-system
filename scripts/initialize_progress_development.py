"""Test a declared base-task progress reward, preserving all failed references."""
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
    config.reward.progress_weight = .25
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_progress_v16_development")
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
    job = {"id": "B1_progress025_hold10_seed11", "seed": 11, "env": config.to_dict(),
           "worker_script": "scripts/hierarchy_development.py", "milestones": [0, 100000, 300000, 1000000],
           "ppo_options": {"target_kl": .03, "use_sde": True, "sde_sample_freq": 1,
                           "log_std_init": -2, "forward_mean_bias": .2, "entropy": .001}}
    write_json(batch / "protocol.json", {"version": "progress_v16_development", "created_epoch": now,
        "deadline_epoch": now+7200, "run_group": "progress_v16", "jobs": [job], "source_hashes": hashes,
        "development_cases": reference["development_cases"], "not_confirmation": True,
        "evidence": "v14 reaches1,000,149physics steps then0/10 with zero progress in every development case. Re-scoring38saved episodes confirms returns. On empty full23m, 10.078m partial progress has12.992 return, while stationary deadline has13.010. Correct terminal-zero PBRS cancels final progress among same-duration failures; motion costs slightly favor stationary failures.",
        "hypothesis": "Add0.25 per metre of signed actual remaining-distance reduction to the shared base-task reward, retaining existingPBRS40, so partial execution quality contributes at finite task deadlines.",
        "comparison": "Exactly the v14 MLP hold10 configuration and seed, except reward.progress_weight0→0.25; same605observations, normalization, critic/actor, PPO, common300k-empty curriculum,1Mphysical budget and evaluation cases",
        "reward_scope": "This is not policy-invariant shaping and changes the task objective. No positive-only clipping, subgoal bonus or future trajectory. Must apply/retrain H/H0/B2/B3 with the same reward before any architecture-advantage comparison; current run is B1 learning diagnosis only.",
        "weight_rationale": "For the fixed23m start-to-goal distance, discounted signed-progress return is at most0.25*23=5.75, below10*0.99**45=6.36 (bounds-vs-deadline event margin). Reward unit tests cover success, waiting, loops, partial progress and early/late safety failures; this is not a safety guarantee.",
        "resource_limits": {"cpu_workers": 1, "disk_min_GiB": 12, "ram_min_GiB": 4,
                            "gpu_stop_temperature_C": 85, "no_progress_seconds": 300, "automatic_failure_retries": 0},
        "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write_json(ROOT / "runs/progress_development.json", {"batch": str(batch)})
    print(batch)


if __name__ == "__main__":
    main()

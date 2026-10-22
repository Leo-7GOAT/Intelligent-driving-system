"""Freeze a direct-control PPO frequency diagnostic with matched physical cost."""
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
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_flat_frequency_v14_development")
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
    job = {"id": "B1_hold10_seed11", "seed": 11, "env": config.to_dict(),
           "worker_script": "scripts/hierarchy_development.py", "milestones": [0, 100000, 300000, 1000000],
           "ppo_options": {"target_kl": .03, "use_sde": True, "sde_sample_freq": 1,
                           "log_std_init": -2, "forward_mean_bias": .2, "entropy": .001}}
    write_json(batch / "protocol.json", {"version": "flat_frequency_v14_development", "created_epoch": now,
        "deadline_epoch": now + 7200, "run_group": "flat_frequency_v14", "jobs": [job], "source_hashes": hashes,
        "development_cases": reference["development_cases"], "not_confirmation": True,
        "hypothesis": "Test whether direct-control PPO learning benefits from holding its own action for0.5s instead of changing it every0.05s; v13 B1 remains0/10 at300k with high measured jerk and slow progress.",
        "comparison": "v13 B1 seed11; identical605inputs, common full23m empty first300k then mixed, same reward/vehicle/actuators/scenes/total1Mphysical budget and PPO options except physical-frequency adjustments",
        "frequency": {"physics_seconds": .05, "action_hold_steps": 10, "decision_seconds": .5,
                      "sde_noise_resample_actions": 1, "sde_noise_resample_seconds": .5,
                      "reference_hold1_sde_noise_resample_actions": 10,
                      "gamma_and_gae": "Derived from physical duration; unchanged per-second values"},
        "limits": "Fresh policy, no MPC or rule action replacement, no reward redesign. Rollout is defined by physical budget; report actions/optimizer work separately. Single-seed development, not a causal proof of hierarchy advantage.",
        "resource_limits": {"cpu_workers": 1, "disk_min_GiB": 12, "ram_min_GiB": 4,
                            "gpu_stop_temperature_C": 85, "no_progress_seconds": 300, "automatic_failure_retries": 0},
        "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write_json(ROOT / "runs/flat_frequency_development.json", {"batch": str(batch)})
    print(batch)


if __name__ == "__main__":
    main()

"""Isolate final-heading PBRS after the completed3M flat-policy failure audit."""
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
from navstack.research import resources_ok
from initialize_candidate_development import reference_protocol


def main():
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError("Resource guard: "+str(reason))
    reference = reference_protocol()
    config = EnvConfig.from_dict(reference["jobs"][0]["env"])
    config.method = "B1"
    config.candidate_observation = True
    config.empty_pretraining_steps = 300000
    config.flat_hold = 10
    config.reward.progress_weight = .25
    config.reward.heading_potential_weight = .1
    config.reward.heading_potential_radius = 6.
    options = {"target_kl": .03, "use_sde": True, "sde_sample_freq": 1,
               "log_std_init": -2, "forward_mean_bias": .2, "entropy": .001}
    comparison = {"reference": "v16 progress-reward B1", "only_active_config_change": "reward.heading_potential_weight0to0.1", "local_parent_verified": False}
    pointer = ROOT / "runs/progress_development.json"
    if pointer.exists():
        parent = Path(json.loads(pointer.read_text(encoding="utf-8"))["batch"])
        prior = json.loads((parent / "protocol.json").read_text(encoding="utf-8"))
        expected = EnvConfig.from_dict(config.to_dict())
        expected.reward.heading_potential_weight = 0.
        if (EnvConfig.from_dict(prior["jobs"][0]["env"]).to_dict() != expected.to_dict()
                or prior["jobs"][0]["ppo_options"] != options or prior["development_cases"] != reference["development_cases"]):
            raise ValueError("v16 comparison differs beyond the declared potential")
        comparison.update(local_parent_verified=True, parent_batch=parent.name)
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_pose_potential_v17_development")
    batch.mkdir()
    hashes = {}
    for folder in ("navstack", "scripts", "configs", "tests"):
        for source in (ROOT / folder).rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix in (".py", ".ps1", ".json"):
                relative = source.relative_to(ROOT)
                target = batch / "source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    now = time.time()
    job = {"id": "B1_posepotential_seed11", "seed": 11, "env": config.to_dict(), "worker_script": "scripts/hierarchy_development.py",
           "milestones": [0, 100000, 300000, 1000000, 2000000, 3000000], "ppo_options": options}
    write_json(batch / "protocol.json", {"version": "pose_potential_v17_development", "run_group": "pose_potential_v17",
          "jobs": [job], "created_epoch": now, "deadline_epoch": now+7200, "source_hashes": hashes,
          "development_cases": reference["development_cases"], "not_confirmation": True,
          "evidence": "v16completed3,001,791physical steps at0/10. Six noncollision cases enter goal position and stop but final headingerror0.60..0.88rad exceeds unchanged0.35rad. Four narrow/unseen collisions remain.",
          "hypothesis": "Earlier smooth alignment credit: Phi_heading=-0.1*(1-cos(yaw-goal_yaw))*exp(-distance/6m), inside existingterminal-zeroPBRS40.",
          "scope": "Fresh seed11 diagnostic, same MLP/605inputs/PPO/base reward/progress0.25/curriculum/vehicle/cases asv16. No oldweights, no controlreplacement, no relaxedgoal. All architectures may use this common reward implementation; matched retraining required before any hierarchy claim.",
          "potential_scope": "Bounded extraPhi[-0.2,0]; physical-time telescoping and terminal/truncation rules unchanged. This doesnotadd a base-task angle reward or promise convergence.",
          "resource_limits": {"cpu_workers": 1, "disk_min_GiB": 12, "ram_min_GiB": 4, "gpu_stop_temperature_C": 85,
                              "no_progress_seconds": 300, "automatic_failure_retries": 0}, "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write_json(batch / "comparison_verified.json", comparison)
    write_json(ROOT / "runs/pose_potential_development.json", {"batch": str(batch)})
    print(batch)


if __name__ == "__main__":
    main()

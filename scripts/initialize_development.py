"""Create a separate development batch and preserve historical test provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import argparse
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("scale", "progress"), default="scale")
    args = parser.parse_args()
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_development_v6")
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
    historical = ROOT / "runs/20261016T015226Z_research_batch_3377580f"
    (batch / "calibration").mkdir()
    (batch / "infrastructure").mkdir()
    shutil.copy2(historical / "calibration/calibration.json", batch / "calibration/calibration.json")
    now = time.time()
    protocol = {
        "version": "development_v6_sanity_1", "created_epoch": now, "deadline_epoch": now + 7200,
        "purpose": "Matched B1 learning sanity: potential reward scale 1 vs 40 on 6m empty route with parking",
        "not_confirmation": True, "old_batch": str(historical.relative_to(ROOT)),
        "historical_test_status": "All previously inspected final/generalization/CARLA manifests are historical evaluations only",
        "training_seed_scope": "one seed for root-cause diagnosis, never infer statistical advantage",
        "evaluation": {"seeds": list(range(7301, 7309)), "route_length": 6, "scenario": "empty", "parking": True,
                       "note": "These near-identical straight routes test basic learning only, not generalization"},
        "resource_limits": {"cpu_workers": 2, "torch_threads_per_worker": 1, "gpu_training": False,
                            "disk_min_GiB": 12, "ram_min_GiB": 4, "gpu_stop_temperature_C": 85,
                            "no_progress_seconds": 300, "drain_seconds": 120, "automatic_failure_retries": 0},
        "jobs": [{"id": "B1_beta" + str(beta) + "_seed11", "reward_beta": beta, "seed": 11,
                  "milestones": [0, 50000, 100000]} for beta in (1, 40)],
        "source_hashes": hashes,
        "requirements": {
            "lightweight_completion": {"overall": .85, "each_core_scene": .75},
            "carla_completion": {"overall": .80, "each_core_scene": .70, "collision_or_bounds": .05},
            "hierarchy": {"mean_paired_gain": .10, "ci95_lower_above": 0, "independent_seeds_min": 5},
            "carla": {"independent_train_seeds_min": 5, "unique_final_conditions_min": 120, "each_family_min": 30},
            "comparators": ["H", "H0", "B1", "B2", "B3", "frequency", "shared_safety", "fixed_reference_controller"],
            "release": ["real_private_github_push", "tag_or_prerelease", "fresh_checkout_cpu", "model_loading",
                        "raw_data_figure_rebuild", "representative_carla", "English_README", "Chinese_docs", "bilingual_CV"]},
        "completion": "unproven; all research acceptance and publication gates remain required"
    }
    if args.stage == "progress":
        protocol.update(version="development_v6_sanity_2_progress", purpose="Test dense measured task progress against PBRS; both receive 1M physical steps from scratch",
                        reward_disclosure="task_progress is a changed task reward, NOT policy-invariant PBRS; future H and all related controls must share it if retained",
                        jobs=[{"id": "B1_PBRS40_seed11", "reward_beta": 40, "progress_weight": 0, "seed": 11, "milestones": [0, 100000, 300000, 1000000]},
                              {"id": "B1_progress1_seed11", "reward_beta": 0, "progress_weight": 1, "seed": 11, "milestones": [0, 100000, 300000, 1000000]}])
    (batch / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (ROOT / "runs/active_development.json").write_text(json.dumps({"batch": str(batch)}, indent=2), encoding="utf-8")
    print(batch)


if __name__ == "__main__":
    main()

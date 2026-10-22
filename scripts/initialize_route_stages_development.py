"""Progressively teach terminal parking at fixed distances before mixed tasks."""
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
    if not healthy or shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError("Resource guard: " + str(reason))
    reference = reference_protocol()
    config = EnvConfig.from_dict(reference["jobs"][0]["env"])
    config.method = "B1"
    config.candidate_observation = True
    config.empty_pretraining_steps = 0
    config.route_stages = [{"until_physics_steps": (i+1)*1000000, "route_m": distance}
                           for i, distance in enumerate((6., 12., 18., 23.))]
    config.flat_hold = 10
    config.reward.progress_weight = .25
    config = EnvConfig.from_dict(config.to_dict())
    comparison = {"reference": "v16", "local_parent_verified": False,
                  "changes": ["empty_pretraining_steps:300000to0", "route_stages:6/12/18/23m at1M each", "budget:5M"]}
    pointer = ROOT / "runs/progress_development.json"
    if pointer.exists():
        parent = Path(json.loads(pointer.read_text(encoding="utf-8"))["batch"])
        prior = json.loads((parent / "protocol.json").read_text(encoding="utf-8"))
        expected = EnvConfig.from_dict(config.to_dict())
        expected.route_stages = []
        expected.empty_pretraining_steps = 300000
        if EnvConfig.from_dict(prior["jobs"][0]["env"]).to_dict() != expected.to_dict():
            raise ValueError("Unexpected difference from v16 configuration")
        comparison.update(local_parent_verified=True, parent_batch=parent.name)
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_route_stages_v18_development")
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
    job = {"id": "B1_route_stages_seed11", "seed": 11, "env": config.to_dict(),
           "worker_script": "scripts/hierarchy_development.py", "milestones": [0, 100000, 300000, 1000000, 2000000, 3000000, 4000000, 5000000],
           "ppo_options": {"target_kl": .03, "use_sde": True, "sde_sample_freq": 1,
                           "log_std_init": -2, "forward_mean_bias": .2, "entropy": .001}}
    write_json(batch / "protocol.json", {"version": "route_stages_v18_development", "created_epoch": now,
        "deadline_epoch": now+7200, "run_group": "route_stages_v18", "jobs": [job], "source_hashes": hashes,
        "development_cases": reference["development_cases"], "not_confirmation": True,
        "evidence": "v16 completed3M at0/10 with near-goal yaw failures; v17 heading-PBRS completed3M at0/10. Existing6mempty sanity policy learns8/8, but immediatefull23manduniform6to23training do not transfer that basic skill. The earliercommon300kempty stage endsbeforeB1learnsfull-taskcompletion.",
        "hypothesis": "Give terminal parking repeated exposure at6,12,18,23m, each1Mphysicalsteps, thenmixed1M. All budgets count, transitionsatresetonly, no teacher or validation-dependent promotion.",
        "comparison": "Samev16 MLP/605observations/hold10/gSDE/PPO/baseprogress0.25, no headingPBRS; onlytrainingcurriculumchanges. Freshseed11,5Mtotalphysicalbudget, commonunchanged10full-taskdevelopmentcases.",
        "fairness": "Single-seeddevelopment, notarchitectureevidence. ScheduleisavailabletoH/H0/B1/B2/B3withidenticalphysicalthresholds; comparativetrainingmustuseequaldistributionandtotalphysicalbudget. No hiddenpretrainingorweightsreuse.",
        "milestone_scope": "The first4Maredeclaredempty-routecurriculum, notmixedtraining. Full-taskdevelopmentmayremainlow; inspectactualtrainingroutesuccessandphasecounts separately. Last1Mis mixed, notanadditional5M.",
        "resource_limits": {"cpu_workers": 1, "disk_min_GiB": 12, "ram_min_GiB": 4,
                            "gpu_stop_temperature_C": 85, "no_progress_seconds": 300, "automatic_failure_retries": 0},
        "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write_json(ROOT / "runs/route_stages_development.json", {"batch": str(batch)})
    write_json(batch / "comparison_verified.json", comparison)
    print(batch)


if __name__ == "__main__":
    main()

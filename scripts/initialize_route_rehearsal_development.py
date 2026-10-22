"""A single-variable fresh PPO experiment after measured v18 forgetting."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import write_json
from navstack.research import resources_ok


def main():
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError("Resource guard: "+str(reason))
    pointer = ROOT/"runs/route_rehearsal_development.json"
    if pointer.exists():
        raise RuntimeError("Inspect existing rehearsal batch; do not duplicate initialization")
    parent = Path(json.loads((ROOT/"runs/route_stages_development.json").read_text(encoding="utf-8"))["batch"])
    prior = json.loads((parent/"protocol.json").read_text(encoding="utf-8"))
    original = prior["jobs"][0]
    old_run = parent/prior["run_group"]/original["id"]
    if not (old_run/"completed.json").is_file():
        raise ValueError("Original fixed-stage experiment must finish first")
    finished = json.loads((parent/"supervisor_final.json").read_text(encoding="utf-8"))
    if any(child["exit_code"] != 0 for child in finished["children"]):
        raise ValueError("Parent did not exit cleanly")
    config = EnvConfig.from_dict(original["env"])
    assert config.route_rehearsal_probability == 0 and config.route_stages
    config.route_rehearsal_probability = .25
    canonical = EnvConfig.from_dict(config.to_dict()); canonical.route_rehearsal_probability = 0.
    assert canonical.to_dict() == EnvConfig.from_dict(original["env"]).to_dict()
    batch = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_route_rehearsal_v19_development")
    batch.mkdir()
    hashes = {}
    for folder in ("navstack","scripts","configs","tests"):
        for source in (ROOT/folder).rglob("*"):
            if source.is_file() and source.suffix in (".py",".ps1",".json") and "__pycache__" not in source.parts:
                relative = source.relative_to(ROOT)
                target = batch/"source"/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
                hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    job = {**original, "id":"B1_route_rehearsal_seed11", "env":config.to_dict()}
    now = time.time()
    write_json(batch/"protocol.json", {"version":"route_rehearsal_v19_development", "created_epoch":now,
        "deadline_epoch":now+7200, "run_group":"route_rehearsal_v19", "jobs":[job], "source_hashes":hashes,
        "development_cases":prior["development_cases"], "not_confirmation":True,
        "evidence":"v18 learned6m/12m/23m at individual stage endpoints but lost earlier distances and regressed after mixed training. Frozen4M/5M normalization swaps did not rescue5M parking or destroy4M parking.",
        "hypothesis":"At each training reset, retain previously completed distance tasks with probability0.25, choosing a previous distance uniformly. Other episodes use the existing active stage or final mixed distribution. No replay-buffer, teacher or changed promotion criterion.",
        "comparison":"Only route_rehearsal_probability0to0.25 relative to canonical v18 config. Same stages,total5M,seed,network,reward,PPO,gSDE,actionhold and validation. Fresh policy, no hidden parent updates.",
        "fairness":"Available equally to all architectures; this single-seed comparison cannot establish hierarchy advantage. The final stage receives fewer mixed episodes; report actual distribution and physical interaction counts.",
        "resource_limits":prior["resource_limits"], "disk_free_bytes":shutil.disk_usage(ROOT).free,
        "parent_batch":parent.name})
    write_json(batch/"comparison_verified.json", {"canonical_parent_match":True,"only_changed_field":"route_rehearsal_probability", "parent":parent.name})
    write_json(pointer,{"batch":str(batch)})
    print(batch,flush=True)


if __name__ == "__main__":
    main()

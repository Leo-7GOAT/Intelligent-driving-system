"""Continue a completed pair in a new window using exactly its frozen source.

Copies the last complete physical/RNG/optimizer snapshot. The parent evidence is
never changed; cumulative budgets include all inherited interactions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    args = parser.parse_args()
    parent = Path(args.parent).resolve()
    old = read(parent / "protocol.json")
    final = read(parent / "supervisor_final.json")
    assert all(c["exit_code"] == 0 for c in final["children"])
    for name, digest in old["source_hashes"].items():
        assert hashlib.sha256((parent / "source" / name).read_bytes()).hexdigest() == digest
    for job in old["jobs"]:
        assert (parent / old["run_group"] / job["id"] / "completed.json").exists()
    batch = parent.parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_hierarchy_v8_continuation")
    batch.mkdir()
    shutil.copytree(parent / "source", batch / "source", ignore=shutil.ignore_patterns("__pycache__"))
    ancestry = []
    for job in old["jobs"]:
        origin = parent / old["run_group"] / job["id"]
        destination = batch / old["run_group"] / job["id"]
        destination.mkdir(parents=True)
        latest = read(origin / "latest.json")
        checkpoint = Path(latest["checkpoint"])
        shutil.copytree(checkpoint, destination / "initial_checkpoint")
        for name in ("ppo_options.json", "resolved_config.json", "version.json"):
            shutil.copy2(origin / name, destination / name)
        ancestry.append({"job": job["id"], "parent_checkpoint": str(checkpoint),
                         "inherited_physics_steps": latest["physics_steps"],
                         "model_sha256": hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest()})
        latest["checkpoint"] = str(destination / "initial_checkpoint")
        (destination / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
        job["milestones"] = [300000, 1000000]
    now = time.time()
    old.update(version="hierarchy_v8_continuation", created_epoch=now, deadline_epoch=now+7200,
               parent_batch=str(parent), inherited_checkpoints=ancestry,
               continuation_reason="At 200k H 8/10 vs H0 6/10 on development, both still fail unseen. Continue both unchanged to cumulative1M; not independent confirmation.",
               concurrent_machine_budget="Up to four one-thread PPO workers across this pair and the separate B1 distance pair; no GPU training. Machine has12 logical CPUs and approximately43GiB available RAM at launch.")
    (batch / "protocol.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
    shutil.copy2(__file__, batch / "continuation_recipe.py")
    root = Path(__file__).resolve().parents[1]
    (root / "runs/hierarchy_continuation.json").write_text(json.dumps({"batch": str(batch)}, indent=2), encoding="utf-8")
    print(batch)


if __name__ == "__main__":
    main()

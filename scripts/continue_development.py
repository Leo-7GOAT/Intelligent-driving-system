"""Create a bounded cumulative continuation only after a successful parent exit."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def prepare(parent, milestones, label, reason, wall_seconds=7200, root=ROOT):
    parent, root = Path(parent).resolve(), Path(root).resolve()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", label):
        raise ValueError("Use a short lowercase continuation label")
    if not reason.strip() or wall_seconds < 300 or wall_seconds > 7200:
        raise ValueError("An explicit reason and a300..7200second window are required")
    if (not milestones or any(type(x) is not int or x <= 0 for x in milestones)
            or milestones != sorted(set(milestones))):
        raise ValueError("Cumulative milestones must be strictly increasing positive integers")
    healthy, reason_resource = resources_ok(root)
    if not healthy or shutil.disk_usage(root).free < 12 * 1024**3:
        raise RuntimeError("Continuation resource guard: " + str(reason_resource))
    old = read(parent / "protocol.json")
    # Kernel-held leases verify ownership rather than trusting stale PID files.
    # Hold all parent leases through copying so no parent writer can restart.
    with ExitStack() as stack:
        stack.enter_context(ProcessLease(parent / "supervisor.lock"))
        for job in old["jobs"]:
            stack.enter_context(ProcessLease(parent / old["run_group"] / job["id"] / "worker.lock"))
        final = read(parent / "supervisor_final.json")
        exits = {row["job"]: row["exit_code"] for row in final["children"]}
        if exits != {job["id"]: 0 for job in old["jobs"]}:
            raise ValueError("Parent jobs did not all exit successfully")
        for name, digest in old["source_hashes"].items():
            if hashlib.sha256((parent / "source" / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Parent frozen source hash mismatch")
        parents = []
        for job in old["jobs"]:
            origin = parent / old["run_group"] / job["id"]
            completed, latest = read(origin / "completed.json"), read(origin / "latest.json")
            counters = ("physics_steps", "actions", "episodes", "optimization_phases", "optimizer_steps")
            if any(completed["summary"][key] != latest[key] for key in counters):
                raise ValueError("Parent completion does not match latest checkpoint")
            if milestones[0] <= latest["physics_steps"]:
                raise ValueError("Continuation milestone does not add physical interactions")
            checkpoint = Path(latest["checkpoint"]).resolve()
            if not checkpoint.is_relative_to(origin.resolve()):
                raise ValueError("Parent checkpoint escapes its job directory")
            for required in ("model.zip", "normalization.pkl", "resume.pkl", "config.json", "training_summary.json"):
                if not (checkpoint / required).is_file():
                    raise ValueError("Incomplete resumable checkpoint")
            saved = read(checkpoint / "training_summary.json")
            config = read(checkpoint / "config.json")
            if any(saved[key] != latest[key] for key in counters) or not saved["reload_exact"]:
                raise ValueError("Checkpoint counter/reload evidence mismatch")
            if config["env"] != job["env"] or config["seed"] != job["seed"]:
                raise ValueError("Checkpoint task/seed differs from parent protocol")
            if read(origin / "ppo_options.json") != job.get("ppo_options", {}):
                raise ValueError("Parent PPO configuration mismatch")
            parents.append((job, origin, checkpoint, latest))
        now = time.time()
        batch = parent.parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + label)
        batch.mkdir()
        shutil.copytree(parent / "source", batch / "source", ignore=shutil.ignore_patterns("__pycache__"))
        ancestry = []
        for job, origin, checkpoint, latest in parents:
            destination = batch / old["run_group"] / job["id"]
            destination.mkdir(parents=True)
            copied = destination / "initial_checkpoint"
            shutil.copytree(checkpoint, copied)
            hashes = {}
            for file in checkpoint.iterdir():
                if file.is_file():
                    digest = hashlib.sha256(file.read_bytes()).hexdigest()
                    if hashlib.sha256((copied / file.name).read_bytes()).hexdigest() != digest:
                        raise ValueError("Copied checkpoint bytes differ")
                    hashes[file.name] = digest
            for filename in ("ppo_options.json", "version.json"):
                shutil.copy2(origin / filename, destination / filename)
            resolved = read(origin / "resolved_config.json")
            write(destination / "inherited_resolved_config.json", resolved)
            resolved["deadline_epoch"] = now + wall_seconds - 120
            resolved["continuation_parent_batch"] = parent.name
            write(destination / "resolved_config.json", resolved)
            ancestry.append({"job": job["id"], "parent_checkpoint": str(checkpoint),
                             "inherited_physics_steps": latest["physics_steps"],
                             "inherited_actions": latest["actions"],
                             "inherited_optimizer_steps": latest["optimizer_steps"], "files_sha256": hashes})
            latest["checkpoint"] = str(copied)
            write(destination / "latest.json", latest)
            job["milestones"] = list(milestones)
        old.update(version=label, created_epoch=now, deadline_epoch=now+wall_seconds,
                   parent_batch=str(parent), inherited_checkpoints=ancestry,
                   continuation_reason=reason, not_confirmation=True,
                   disk_free_bytes=shutil.disk_usage(root).free,
                   continuation_scope="Same policy/source/RNG/optimizer/env/normalization; cumulative cost includes ancestors; not an independent seed")
        write(batch / "protocol.json", old)
        shutil.copy2(__file__, batch / "continuation_recipe.py")
        write(root / "runs" / (label + ".json"), {"batch": str(batch)})
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--milestones", nargs="+", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--wall-seconds", type=int, default=7200)
    args = parser.parse_args()
    print(prepare(args.parent, args.milestones, args.label, args.reason, args.wall_seconds))


if __name__ == "__main__":
    main()

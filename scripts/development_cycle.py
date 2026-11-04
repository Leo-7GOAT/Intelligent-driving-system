"""Bounded, resumable development diagnostics; never invokes a final test.

Each worker has an OS lease and immutable source snapshot. Repeat commands reuse
complete milestones. Failed jobs require inspection, not automatic retry loops.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate_short(run, checkpoint, config, label):
    import numpy as np
    from navstack.env import NavigationEnv
    from navstack.experiments import load_run, jsonable
    from navstack.research import atomic_json
    marker = run / (label + ".json")
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    model, norm = load_run(checkpoint)
    config.split = "validation"
    rows = []
    with gzip.open(run / (label + "_trace.jsonl.gz"), "wt", encoding="utf-8") as stream:
        for seed in range(7301, 7309):
            env = NavigationEnv(config)
            obs, _ = env.reset(seed=seed, options={"diagnostic_route_length": 6})
            while True:
                action, _ = model.predict(norm.normalize_obs(np.asarray(obs)[None]), deterministic=True)
                obs, reward, done, _, info = env.step(action[0])
                stream.write(json.dumps({"seed": seed, "action": action[0], "info": info}, default=jsonable) + "\n")
                if done:
                    rows.append({"seed": seed, **info["episode_metrics"]})
                    break
            env.close()
    norm.close()
    result = {"scope": "development sanity only, fixed 6m empty route", "checkpoint": str(checkpoint),
              "model_sha256": hashlib.sha256((Path(checkpoint) / "model.zip").read_bytes()).hexdigest(),
              "safe_completion": sum(r["success"] for r in rows) / len(rows), "episodes": rows}
    atomic_json(marker, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--job", required=True)
    args = p.parse_args()
    batch = Path(args.batch).resolve()
    from navstack.config import EnvConfig, PlannerConfig
    from navstack.process_lease import ProcessLease
    from navstack.research import cumulative_train, atomic_json
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    job = next(j for j in protocol["jobs"] if j["id"] == args.job)
    run = batch / "sanity_v6" / job["id"]
    run.mkdir(parents=True, exist_ok=True)
    with ProcessLease(run / "worker.lock"):
        if (run / "completed.json").exists():
            print("already complete", flush=True)
            return
        if (run / "failure.json").exists():
            raise RuntimeError("Prior failure requires explicit diagnosis; refusing automatic retry")
        if time.time() >= protocol["deadline_epoch"] - 120:
            print("window ended; checkpoint retained", flush=True)
            return
        cfg = EnvConfig(method="B1", scenario="empty", training_route_length=6, deadline=15,
                        planner=PlannerConfig(max_seconds=None))
        cfg.reward.beta = job["reward_beta"]
        cfg.reward.progress_weight = job.get("progress_weight", 0.0)
        options = {"target_kl": .03, "use_sde": True, "sde_sample_freq": 10,
                   "log_std_init": -2, "forward_mean_bias": .2, "entropy": .001}
        try:
            for target in job["milestones"]:
                atomic_json(run / "worker_status.json", {"state": "training", "pid": os.getpid(), "target": target,
                            "epoch": time.time(), "source_root": str(ROOT)})
                summary = cumulative_train(run, cfg, job["seed"], target, protocol["deadline_epoch"] - 120,
                                           ppo_options=options)
                if summary["physics_steps"] < target or summary["reason"] not in ("milestone_reached", "periodic"):
                    atomic_json(run / "worker_status.json", {"state": "saved_stop", "summary": summary})
                    return
                result = evaluate_short(run, summary["checkpoint"], EnvConfig.from_dict(cfg.to_dict()), "validation_" + str(target))
                print(json.dumps({"target": target, "safe_completion": result["safe_completion"]}), flush=True)
            atomic_json(run / "completed.json", {"epoch": time.time(), "last_summary": summary})
            atomic_json(run / "worker_status.json", {"state": "complete", "epoch": time.time()})
        except BaseException as exc:
            atomic_json(run / "failure.json", {"epoch": time.time(), "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()

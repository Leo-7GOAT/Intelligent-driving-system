"""Frozen PPO transfer or matched fine-tuning; raw CARLA frames remain the sole truth."""

import argparse, gzip, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from navstack.carla_bridge import (
    BridgeClient,
    CarlaNavigationEnv,
    InfrastructureFailure,
)
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import new_run, write_json, append_json, jsonable
from carla_diagnostic import calibrated_config


def evaluate_carla(
    batch,
    checkpoint,
    cases_path,
    split="development",
    label="low_level_recalibrated_frozen_policy",
    run=None,
):
    checkpoint = Path(checkpoint)
    saved = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    old = EnvConfig.from_dict(saved["env"])
    config = calibrated_config(batch, old.method, source_config=old)
    config.split = "validation" if split == "development" else "test"
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    case_hash = hashlib.sha256(Path(cases_path).read_bytes()).hexdigest()
    if split == "test":
        freeze = json.loads(
            (Path(batch) / "development_freeze.json").read_text(encoding="utf-8")
        )
        if case_hash not in freeze["manifest_hashes"].values():
            raise ValueError("Unregistered final CARLA cases")
        origin = Path(saved.get("source_checkpoint", checkpoint))
        if origin.parent.parent.name not in freeze["registered_source_versions"]:
            raise ValueError("CARLA source is not from the registered confirmation version")
        if (
            json.loads((origin / "training_summary.json").read_text(encoding="utf-8"))[
                "physics_steps"
            ]
            < 1000000
        ):
            raise ValueError(
                "Formal CARLA test requires the frozen lightweight 1M source"
            )
    run = (
        Path(run)
        if run
        else new_run(Path(batch) / "carla", f"PPO_{old.method}_{split}_{label}")
    )
    run.mkdir(parents=True, exist_ok=True)
    if (run / "evaluation_summary.json").exists():
        prior = json.loads(
            (run / "evaluation_summary.json").read_text(encoding="utf-8")
        )
        if prior.get("complete", True):
            raise RuntimeError("Refuse repeat of a completed CARLA evaluation")
    if (run / "config.json").exists():
        prior = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if prior["checkpoint"] != str(checkpoint) or prior["cases"] != cases:
            raise ValueError("CARLA resume checkpoint/case mismatch")
    write_json(
        run / "config.json",
        {
            "env": config.to_dict(),
            "source_env": old.to_dict(),
            "checkpoint": str(checkpoint),
            "model_sha256": hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest(),
            "policy": "learned_PPO",
            "transfer": label,
            "split": split,
            "cases": cases,
            "seed": saved["seed"],
            "manifest_sha256": case_hash,
        },
    )
    torch.set_num_threads(1)
    normalizer = VecNormalize.load(
        checkpoint / "normalization.pkl", DummyVecEnv([lambda: NavigationEnv(old)])
    )
    normalizer.training = False
    normalizer.norm_reward = False
    means = normalizer.obs_rms.mean.copy()
    model = PPO.load(checkpoint / "model.zip", device="cpu")
    client = BridgeClient(batch)
    results = (
        [
            json.loads(x)
            for x in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if (run / "episodes.jsonl").exists()
        else []
    )
    done = {r["case_id"] for r in results}
    started = time.time()
    past_ticks = 0
    censored = False
    failures = (
        [
            json.loads(x)
            for x in (run / "infrastructure_failures.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if (run / "infrastructure_failures.jsonl").exists()
        else []
    )
    deadline = json.loads((Path(batch) / "batch.json").read_text(encoding="utf-8"))[
        "deadline_epoch"
    ]
    try:
        with gzip.open(
            run
            / (
                "transitions.jsonl.gz"
                if not (run / "transitions.jsonl.gz").exists()
                else f"transitions_{time.time_ns()}.jsonl.gz"
            ),
            "wt",
            encoding="utf-8",
        ) as stream:
            for index, case in enumerate(cases):
                if index in done:
                    continue
                if time.time() > deadline - 60 or (Path(batch) / "STOP").exists():
                    censored = True
                    break
                # One predefined retry for infrastructure faults; the first failure remains recorded.
                attempts = [r["attempt"] for r in failures if r["case_id"] == index]
                first_attempt = max(attempts) + 1 if attempts else 0
                if first_attempt >= 2:
                    row = {
                        "case_id": index,
                        "case": case,
                        "status": "infrastructure_unresolved",
                        "success": False,
                    }
                    results.append(row)
                    append_json(run / "episodes.jsonl", row)
                    continue
                for attempt in range(first_attempt, 2):
                    env = CarlaNavigationEnv(config, client)
                    try:
                        obs, _ = env.reset(options={"case": case})
                        append_json(
                            run / "scenes.jsonl",
                            {
                                "case_id": index,
                                "attempt": attempt,
                                "scene": env.scene_evidence,
                                "initial_state": env.initial_scenario,
                            },
                        )
                        while not env.done:
                            if (
                                time.time() > deadline - 60
                                or (Path(batch) / "STOP").exists()
                            ):
                                censored = True
                                break
                            action = model.predict(
                                normalizer.normalize_obs(obs), deterministic=True
                            )[0]
                            obs, reward, _, _, info = env.step(action)
                            stream.write(
                                json.dumps(
                                    {
                                        "case_id": index,
                                        "attempt": attempt,
                                        "policy_action": action,
                                        "mode": info["mode"],
                                        "plan": info["plan"],
                                        "trace": info["trace"],
                                    },
                                    default=jsonable,
                                )
                                + "\n"
                            )
                        if censored:
                            append_json(
                                run / "censored.jsonl",
                                {
                                    "case_id": index,
                                    "attempt": attempt,
                                    "reason": "batch_budget_or_stop",
                                    "physics_steps": env.episode["physics_steps"],
                                },
                            )
                            break
                        row = {
                            "case_id": index,
                            "case": case,
                            "attempt": attempt,
                            "status": "completed",
                            **info["episode_metrics"],
                        }
                        results.append(row)
                        append_json(run / "episodes.jsonl", row)
                        print(
                            json.dumps(
                                {
                                    "case": case,
                                    "method": old.method,
                                    "reason": info["termination_reason"],
                                    "progress": row["progress"],
                                }
                            ),
                            flush=True,
                        )
                        break
                    except InfrastructureFailure as exc:
                        append_json(
                            run / "infrastructure_failures.jsonl",
                            {
                                "case_id": index,
                                "attempt": attempt,
                                "error": str(exc),
                                "physics_steps_before_failure": (
                                    env.episode.get("physics_steps", 0)
                                    if hasattr(env, "episode")
                                    else 0
                                ),
                            },
                        )
                        if attempt == 1:
                            row = {
                                "case_id": index,
                                "case": case,
                                "status": "infrastructure_unresolved",
                                "success": False,
                            }
                            results.append(row)
                            append_json(run / "episodes.jsonl", row)
                        else:
                            past_ticks += client.total_ticks
                            client.close()
                            client = BridgeClient(batch)
                if censored:
                    break
        complete = [r for r in results if r["status"] == "completed"]
        summary = {
            "run": str(run),
            "method": old.method,
            "backend": "CARLA_0.9.15",
            "policy": "learned_PPO",
            "transfer": label,
            "split": split,
            "cases": len(cases),
            "completed": len(complete),
            "infrastructure_unresolved": len(results) - len(complete),
            "seed": saved["seed"],
            "manifest_sha256": case_hash,
            "complete": len(results) == len(cases),
            "censored": censored,
            "success_rate_all_scheduled": sum(r["success"] for r in results)
            / len(cases),
            "task_failure_counts": {
                k: sum(r.get("termination_reason") == k for r in results)
                for k in ("collision", "out_of_bounds", "deadline")
            },
            "physics_steps": sum(r.get("physics_steps", 0) for r in results),
            "wall_seconds": time.time() - started,
            "world_physics_ticks_including_reset_this_session": past_ticks
            + client.total_ticks,
            "world_physics_ticks_including_reset": past_ticks + client.total_ticks,
            "normalization_frozen": bool(
                np.array_equal(means, normalizer.obs_rms.mean)
            ),
            "source_checkpoint": str(checkpoint),
        }
        write_json(run / "evaluation_summary.json", summary)
        return run, summary
    finally:
        client.close()
        normalizer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cases", default="configs/carla_development_cases.json")
    p.add_argument("--split", default="development")
    args = p.parse_args()
    print(evaluate_carla(args.batch, args.checkpoint, args.cases, args.split)[1])

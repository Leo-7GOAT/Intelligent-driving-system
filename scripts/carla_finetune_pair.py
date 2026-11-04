"""Matched CARLA interaction budgets after the first paired lightweight 100k milestone."""

import argparse, json, pickle, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from navstack.carla_bridge import BridgeClient, CarlaNavigationEnv
from navstack.experiments import (
    new_run,
    write_json,
    append_json,
    actor_vector,
    signature,
)
from navstack.research import ResearchRecorder, atomic_json, preserve_global_rng
from carla_diagnostic import calibrated_config
from carla_evaluate import evaluate_carla


class CarlaRecorder(ResearchRecorder):
    def __init__(self, run, deadline, client, counts=None):
        super().__init__(run, deadline, counts)
        self.client = client
        self.initial_ticks = self.counts["physics_steps"]
        self.controlled_steps = self.counts.get("controlled_physics_steps", 0)

    def _on_step(self):
        ok = super()._on_step()
        self.controlled_steps += self.locals["infos"][0]["executed_substeps"]
        self.counts["physics_steps"] = self.initial_ticks + self.client.total_ticks
        self.counts["controlled_physics_steps"] = self.controlled_steps
        self.counts["reset_physics_steps"] = (
            self.counts["physics_steps"] - self.controlled_steps
        )
        return ok


def finetune(batch, checkpoint, budget, seed=11, run=None):
    checkpoint = Path(checkpoint).resolve()
    metadata = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    method = metadata["env"]["method"]
    from navstack.config import EnvConfig

    saved = EnvConfig.from_dict(metadata["env"])
    config = calibrated_config(batch, method, source_config=saved)
    run = (
        Path(run)
        if run
        else new_run(Path(batch) / "carla", f"finetune_{method}_seed{seed}")
    )
    run.mkdir(parents=True, exist_ok=True)
    origin = checkpoint
    previous = None
    if (run / "latest.json").exists():
        previous = json.loads((run / "latest.json").read_text(encoding="utf-8"))
        saved_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if (
            saved_config["source_checkpoint"] != str(origin)
            or saved_config["seed"] != seed
        ):
            raise ValueError("CARLA resume source/seed mismatch")
        if previous["physics_steps"] >= budget:
            return previous
        checkpoint = Path(previous["checkpoint"])
    deadline = (
        json.loads((Path(batch) / "batch.json").read_text(encoding="utf-8"))[
            "deadline_epoch"
        ]
        - 60
    )
    write_json(
        run / "config.json",
        {
            "env": config.to_dict(),
            "signature": signature(config),
            "seed": seed,
            "source_checkpoint": str(origin),
            "backend": "CARLA_0.9.15",
            "budget_includes_reset_ticks": True,
            "resume_semantics": "Optimizer, normalization, counters and RNG retained; CARLA starts a new episode, with reset ticks counted",
        },
    )
    client = BridgeClient(batch)
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: CarlaNavigationEnv(config, client)])
    env.seed(seed)
    vec = VecNormalize.load(checkpoint / "normalization.pkl", env)
    vec.training = True
    vec.norm_reward = False
    model = PPO.load(checkpoint / "model.zip", env=vec, device="cpu", force_reset=True)
    model.set_logger(configure(str(run / "sb3"), ["csv"]))
    if previous:
        with (checkpoint / "resume_stats.pkl").open("rb") as stream:
            snapshot = pickle.load(stream)
        initial = snapshot["initial_actor"]
        initial_updates = snapshot["initial_updates"]
        counts = snapshot["counts"]
        random.setstate(snapshot["random"])
        np.random.set_state(snapshot["numpy_random"])
        torch.set_rng_state(snapshot["torch_random"])
        env.envs[0].np_random.bit_generator.state = snapshot["env_rng"]
        env._seeds = [None]
        append_json(
            run / "resume_events.jsonl",
            {
                "epoch": time.time(),
                "checkpoint": str(checkpoint),
                "new_episode": True,
                "physics_steps": counts["physics_steps"],
            },
        )
    else:
        initial = actor_vector(model).clone()
        initial_updates = model._n_updates
        counts = None
    recorder = CarlaRecorder(run, deadline, client, counts)
    started = time.time()
    last_save = recorder.counts["physics_steps"]
    recorder.counts.setdefault("optimizer_steps", 0)

    def count_step(*unused):
        recorder.counts["optimizer_steps"] += 1

    hook = model.policy.optimizer.register_step_post_hook(count_step)

    def save(reason):
        folder = (
            run
            / f"checkpoint_{recorder.counts['physics_steps']:09d}_{recorder.counts['optimization_phases']:04d}"
        )
        if folder.exists():
            folder = run / (folder.name + "_" + str(time.time_ns()))
        folder.mkdir()
        model.save(folder / "model.zip")
        vec.save(folder / "normalization.pkl")
        write_json(
            folder / "config.json",
            json.loads((run / "config.json").read_text(encoding="utf-8")),
        )
        with (folder / "resume_stats.pkl").open("wb") as stream:
            pickle.dump(
                {
                    "initial_actor": initial,
                    "initial_updates": initial_updates,
                    "counts": recorder.counts,
                    "random": random.getstate(),
                    "numpy_random": np.random.get_state(),
                    "torch_random": torch.get_rng_state(),
                    "env_rng": env.envs[0].np_random.bit_generator.state,
                },
                stream,
            )
        with preserve_global_rng():
            loaded = PPO.load(folder / "model.zip", device="cpu")
        summary = {
            **recorder.counts,
            "backend": "CARLA_0.9.15",
            "method": method,
            "seed": seed,
            "reason": reason,
            "policy": "learned_PPO_finetuned",
            "requested_physics_budget": budget,
            "budget_includes_reset_ticks": True,
            "ppo_epoch_updates_added": model._n_updates - initial_updates,
            "actor_l2_change": float(
                torch.linalg.vector_norm(actor_vector(model) - initial)
            ),
            "reload_exact": all(
                torch.equal(v, loaded.policy.state_dict()[k])
                for k, v in model.policy.state_dict().items()
            ),
            "checkpoint": str(folder.resolve()),
            "wall_seconds": time.time() - started,
            "conclusion": "Measured fine-tuning budget; convergence is not assumed",
        }
        write_json(folder / "training_summary.json", summary)
        atomic_json(run / "latest.json", summary)
        return summary

    try:
        while recorder.counts["physics_steps"] < budget and not recorder.stopped_reason:
            old = model._n_updates
            actor_before = actor_vector(model).clone()
            model.learn(model.n_steps, callback=recorder, reset_num_timesteps=False)
            if model._n_updates > old:
                recorder.counts["optimization_phases"] += 1
                row = {
                    **recorder.counts,
                    "ppo_epoch_updates_added": model._n_updates - initial_updates,
                    "epoch": time.time(),
                    "actor_update_l2": float(
                        torch.linalg.vector_norm(actor_vector(model) - actor_before)
                    ),
                    "session_wall_seconds": time.time() - started,
                }
                row.update(
                    {
                        k: float(v)
                        for k, v in model.logger.name_to_value.items()
                        if k.startswith("train/") and np.isscalar(v) and np.isfinite(v)
                    }
                )
                append_json(run / "optimization.jsonl", row)
                print(
                    json.dumps(
                        {
                            "run": str(run),
                            "method": method,
                            "physics_steps": recorder.counts["physics_steps"],
                            "updates": recorder.counts["optimization_phases"],
                        }
                    ),
                    flush=True,
                )
            if recorder.counts["physics_steps"] - last_save >= 5000:
                save("periodic")
                last_save = recorder.counts["physics_steps"]
        return save(recorder.stopped_reason or "milestone_reached")
    except Exception as exc:
        append_json(
            run / "failures.jsonl",
            {"error": repr(exc), "epoch": time.time(), "counts": recorder.counts,
             "known_acknowledged_world_ticks_this_session": client.total_ticks,
             "last_request": getattr(client,"last_request",None),
             "last_failure_response": getattr(client,"last_failure_response",None),
             "rpc_requests": str(client.request_audit_path), "rpc_responses": str(client.response_audit_path),
             "counting_note": "Recorder callbacks can lag completed controls when automatic reset fails. Any unacknowledged partial reset is infrastructure cost with unknown tick count, never assumed zero."},
        )
        raise
    finally:
        hook.remove()
        recorder.close()
        vec.close()
        client.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--budget", type=int, default=20000)
    args = p.parse_args()
    batch = Path(args.batch)
    checkpoints = {
        m: json.loads(
            (batch / "development_v1" / (m + "_seed11") / "latest.json").read_text(
                encoding="utf-8"
            )
        )["checkpoint"]
        for m in ("H", "B1")
    }
    for method in ("H", "B1"):
        run, summary = evaluate_carla(
            batch, checkpoints[method], "configs/carla_development_cases.json"
        )
        append_json(
            batch / "carla_pair_results.jsonl",
            {
                "stage": "frozen_recalibrated_100k",
                "method": method,
                "evaluation": summary,
            },
        )
    for method in ("H", "B1"):
        summary = finetune(batch, checkpoints[method], args.budget)
        append_json(
            batch / "carla_pair_results.jsonl",
            {"stage": "finetune", "method": method, "training": summary},
        )
        run, evaluation = evaluate_carla(
            batch,
            summary["checkpoint"],
            "configs/carla_development_cases.json",
            label="carla_finetuned",
        )
        append_json(
            batch / "carla_pair_results.jsonl",
            {
                "stage": "finetuned_evaluation",
                "method": method,
                "evaluation": evaluation,
            },
        )

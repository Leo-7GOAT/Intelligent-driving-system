"""Bounded real CARLA training for the new shared-road interface (development)."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from navstack.carla_bridge import BridgeClient, CarlaNavigationEnv
from navstack.config import EnvConfig, VehicleConfig
from navstack.experiments import actor_vector, write_json, signature, append_json
from navstack.native_route import select_routes
from navstack.research import ResearchRecorder, resources_ok
from scripts.carla_steering_audit import guard


def prepare(parent):
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError("Resource guard: "+str(reason))
    parent = Path(parent)
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_native_road_learning")
    batch.mkdir()
    for name in ("infrastructure", "calibration"):
        (batch/name).mkdir()
    shutil.copy2(parent/"calibration/calibration.json", batch/"calibration/calibration.json")
    survey = json.loads((parent/"native_routes.json").read_text(encoding="utf-8"))
    hashes = {}
    for folder in ("navstack", "scripts", "tests"):
        for source in (ROOT/folder).rglob("*.py"):
            relative = source.relative_to(ROOT)
            target = batch/"source"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(batch/"protocol.json", {"scope": "Shared native-road real PPO interface training, not convergence or hierarchy advantage",
        "created_epoch": time.time(), "deadline_epoch": time.time()+900, "source_hashes": hashes,
        "calibration_sha256": hashlib.sha256((batch/"calibration/calibration.json").read_bytes()).hexdigest(),
        "routes": select_routes(survey), "methods": ["H", "B1"], "seed": 11,
        "controlled_budget_per_method": 640, "reset_ticks_counted_separately": True,
        "rollout_actions": 32, "ppo_epochs": 2, "episode_seconds": 20,
        "resource_limits": {"wall_seconds": 900, "disk_min_GiB": 12, "ram_min_GiB": 4, "gpu_stop_temperature_C": 85, "automatic_retries": 0},
        "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write_json(ROOT/"runs/native_road_learning.json", {"batch": str(batch)})
    print(batch, flush=True)


class RecordedRoadEnv(CarlaNavigationEnv):
    def __init__(self, config, client, cases, output):
        self.output = output
        super().__init__(config, client, cases)

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        write_json(self.output/("scene_%04d.json" % self.reset_count), {"scene": self.scene_evidence,
            "initial": self.initial_scenario, "frame": self.plant.frame,
            "observation_shape": list(result[0].shape)})
        return result


def run(batch):
    batch = Path(batch).resolve()
    protocol = json.loads((batch/"protocol.json").read_text(encoding="utf-8"))
    for relative, digest in protocol["source_hashes"].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Run frozen source only")
    if hashlib.sha256((batch/"calibration/calibration.json").read_bytes()).hexdigest() != protocol["calibration_sha256"]:
        raise ValueError("Calibration mismatch")
    if (batch/"started.json").exists():
        raise RuntimeError("No automatic repeat of completed or failed evidence")
    guard(batch, protocol)
    write_json(batch/"started.json", {"epoch": time.time()})
    vehicle = VehicleConfig(**json.loads((batch/"calibration/calibration.json").read_text(encoding="utf-8"))["vehicle"])
    torch.set_num_threads(1)
    results = []
    try:
        for method in protocol["methods"]:
            guard(batch, protocol)
            if time.time() >= protocol["deadline_epoch"]:
                raise RuntimeError("Registered wall budget expired")
            config = EnvConfig(method=method, vehicle=vehicle, road_observation=True,
                flat_hold=10, deadline=protocol["episode_seconds"], obstacle_num=0)
            config.planner.max_seconds = None
            config.planner.vectorized_primitives = True
            config.planner.analytic_expansion = True
            output = batch/method
            output.mkdir()
            write_json(output/"config.json", {"env": config.to_dict(), "signature": signature(config),
                "seed": protocol["seed"], "backend": "CARLA_0.9.15", "scope": protocol["scope"]})
            cases = [{"scenario": "native_curve", "condition": index, "native_route": route} for index, route in enumerate(protocol["routes"])]
            client = BridgeClient(batch)
            vec = None
            recorder = None
            try:
                raw = DummyVecEnv([lambda: RecordedRoadEnv(config, client, cases, output)])
                raw.seed(protocol["seed"])
                vec = VecNormalize(raw, norm_reward=False, gamma=config.gamma)
                model = PPO("MlpPolicy", vec, n_steps=protocol["rollout_actions"], batch_size=32,
                    n_epochs=protocol["ppo_epochs"], gamma=config.gamma, gae_lambda=.95**.5,
                    policy_kwargs={"net_arch":{"pi":[64,64],"vf":[64,64]}}, seed=protocol["seed"], device="cpu")
                model.set_logger(configure(str(output/"sb3"), ["csv"]))
                before = actor_vector(model).clone()
                recorder = ResearchRecorder(output, protocol["deadline_epoch"])
                optimizer_steps = [0]
                def count_step(*unused):
                    optimizer_steps[0] += 1
                hook = model.policy.optimizer.register_step_post_hook(count_step)
                started = time.time()
                while recorder.counts["physics_steps"] < protocol["controlled_budget_per_method"] and not recorder.stopped_reason:
                    guard(batch, protocol)
                    previous = model._n_updates
                    model.learn(protocol["rollout_actions"], callback=recorder, reset_num_timesteps=False)
                    if model._n_updates > previous:
                        recorder.counts["optimization_phases"] += 1
                        append_json(output/"optimization.jsonl", {**recorder.counts, "optimizer_steps": optimizer_steps[0],
                            "actor_l2_change": float(torch.linalg.vector_norm(actor_vector(model)-before)),
                            "metrics": {k: float(v) for k,v in model.logger.name_to_value.items() if np.isscalar(v) and np.isfinite(v)}})
                hook.remove()
                model.save(output/"model.zip"); vec.save(output/"normalization.pkl")
                loaded = PPO.load(output/"model.zip", device="cpu")
                sample = vec.normalize_obs(raw.envs[0]._observation()[None])
                np.testing.assert_array_equal(model.predict(sample, deterministic=True)[0], loaded.predict(sample, deterministic=True)[0])
                result = {**recorder.counts, "method": method, "backend": "CARLA_0.9.15",
                    "world_ticks": client.total_ticks, "reset_ticks": client.total_ticks-recorder.counts["physics_steps"],
                    "optimizer_steps": optimizer_steps[0], "actor_l2_change": float(torch.linalg.vector_norm(actor_vector(model)-before)),
                    "reload_exact": all(torch.equal(v, loaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items()),
                    "wall_seconds": time.time()-started, "stopped_reason": recorder.stopped_reason,
                    "scope": protocol["scope"]}
                write_json(output/"summary.json", result)
                results.append(result)
                print(json.dumps(result), flush=True)
            finally:
                if recorder is not None:
                    recorder.close()
                if vec is not None:
                    vec.close()
                client.close()
        write_json(batch/"summary.json", {"status": "complete", "methods": results})
    except BaseException as exc:
        write_json(batch/"failure.json", {"error": repr(exc), "epoch": time.time(), "completed_methods": results})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare")
    parser.add_argument("--batch")
    args = parser.parse_args()
    prepare(args.prepare) if args.prepare else run(args.batch)

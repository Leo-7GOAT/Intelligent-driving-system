"""Reproducible SB3 training/evaluation. Every invocation gets an independent run directory."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .config import EnvConfig
from .env import NavigationEnv, ACTIONS, OBS_VERSION, rule_action
from .metrics import timing_summary

PROFILES = {
    "smoke": {
        "physics_budget": 640,
        "rollout_physics": 320,
        "epochs": 2,
        "wall_seconds": 600,
        "deadline": 30.0,
    },
    "debug": {
        "physics_budget": 6400,
        "rollout_physics": 640,
        "epochs": 5,
        "wall_seconds": 600,
        "deadline": 30.0,
    },
    "formal": {
        "physics_budget": 1000000,
        "rollout_physics": 2560,
        "epochs": 10,
        "wall_seconds": 600,
        "deadline": 45.0,
    },
}


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, default=jsonable, allow_nan=False
        ),
        encoding="utf-8",
    )


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, default=jsonable, allow_nan=False)
            + "\n"
        )


def manifest():
    root = Path(__file__).resolve().parent.parent
    files = (
        list((root / "navstack").glob("*.py"))
        + list((root / "tests").glob("*.py"))
        + list(root.glob("*.py"))
        + list((root / "configs").glob("*.json"))
        + list(root.glob("requirements*.txt"))
    )
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }

    def git(*args):
        p = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        return p.stdout.strip() if p.returncode == 0 else None

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "numpy",
                "scipy",
                "osqp",
                "gymnasium",
                "torch",
                "stable-baselines3",
            )
        },
        "branch": git("symbolic-ref", "--short", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "source_hashes": hashes,
    }


def new_run(root, prefix):
    run = Path(root) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + prefix
        + "_"
        + uuid.uuid4().hex[:8]
    )
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "version.json", manifest())
    return run


def signature(config):
    reward_signature = asdict(config.reward)
    # Disabled optional shaping preserves the exact signature of published
    # distance-only checkpoints; active shaping is versioned explicitly.
    if not config.reward.heading_potential_weight:
        reward_signature.pop("heading_potential_weight")
        reward_signature.pop("heading_potential_radius")
    result = {
        "observation": OBS_VERSION,
        "observation_shape": [351],
        "method": config.method,
        "action_catalog": (
            list(map(list, ACTIONS))
            if config.method != "B1"
            else "Box([-1,-1],[1,1]):acceleration,steering"
        ),
        "control_dt": config.control_dt,
        "substeps": config.substeps,
        "vehicle": asdict(config.vehicle),
        "planner": asdict(config.planner),
        "mpc_horizon": config.mpc_horizon,
        "reward": reward_signature,
    }
    if config.shared_shield:
        result["shared_shield"] = "execution_brake_v1_no_path_prediction"
    if config.terminal_goal_prefix:
        result["terminal_goal_prefix"] = "selected_candidate_crosses_goal_before_endpoint_validation_v1"
    if config.curriculum:
        result["training_curriculum"] = "6m_until_50k_16m_until_150k_then_full_one_env"
    if config.subgoal_mode != "ego_arcs":
        catalog = "goal_frame_v1:wait+lookahead(6,9,12)*offset(-3,-1.5,0,1.5,3)*speed(.75,1.5)"
        if config.method == "B1": result["candidate_catalog"] = catalog
        else: result["action_catalog"] = catalog
    if not config.planner.analytic_expansion:
        result["planner"].pop("analytic_expansion")  # unchanged historical planner semantics
    if not config.planner.vectorized_primitives:
        result["planner"].pop("vectorized_primitives")
    if config.planner.grid_postprocess == "legacy":
        result["planner"].pop("grid_postprocess")
    if config.subgoal_mode == "goal_frame_terminal":
        result["candidate_catalog" if config.method == "B1" else "action_catalog"] += ":terminal_S_arc_envelope_Rmargin1.2_v2"
    if config.subgoal_mode == "route_frame_terminal":
        result["candidate_catalog" if config.method == "B1" else "action_catalog"] = (
            "route_frame_terminal_v1:known_route_projection+lookahead(6,9,12)*offset(-3,-1.5,0,1.5,3)*speed(.75,1.5):terminal_S_arc_envelope_Rmargin1.2")
        result["route_prior"] = "known_map_route_tangents_exposed_as_shared_candidate_geometry_no_state_projection"
    if config.execution_feedback != "off":
        result["observation"] += "-candidate-witness31x3-execution6-v1"
        result["observation_shape"] = [450]
        result["execution_feedback"] = config.execution_feedback
    if config.candidate_observation:
        result["observation"] = OBS_VERSION + "-candidate-geometry-witness31x8-execution6-v2"
        result["observation_shape"] = [605]
        result["candidate_scaling"] = "ego_xy/15,sin_yaw,cos_yaw,speed/3,clear,witness,time/45"
        if config.execution_feedback.startswith('continuation'):
            result["candidate_scaling"] = "ego_xy/15,sin_yaw,cos_yaw,speed/3,clear,local_witness,target_goal_witness_v1"
    if config.empty_pretraining_steps:
        result["task_stage_curriculum"] = {"full_length_empty_until_physics_steps": config.empty_pretraining_steps,
                                           "then": "original_mixed_distribution_at_next_episode", "evaluation": "unchanged"}
    if config.road_observation:
        result["observation"] += "-measured-road64x64-v1"
        result["observation_shape"][0] += 4096
        result["road_geometry"] = config.road_geometry_version+"_closed_cells_shared_planning_risk_bounds"
    if config.route_rehearsal_probability:
        result["route_rehearsal"] = {"probability":config.route_rehearsal_probability,
            "sampling":"uniform_completed_distances_at_training_reset", "stages":config.route_stages}
    return result


def make_vector(config, seed, n_envs=1):
    env = DummyVecEnv(
        [
            lambda c=EnvConfig.from_dict(config.to_dict()): NavigationEnv(c)
            for _ in range(n_envs)
        ]
    )
    env.seed(seed)
    from .candidate_policy import CandidateVecNormalize
    normalizer = CandidateVecNormalize if config.candidate_observation else VecNormalize
    return normalizer(
        env, norm_obs=True, norm_reward=False, gamma=config.gamma, clip_obs=10.0
    )


class Recorder(BaseCallback):
    def __init__(self, run, wall_deadline):
        super().__init__()
        self.run = Path(run)
        self.deadline = wall_deadline
        self.physics = 0
        self.actions = 0
        self.episodes = 0
        self.interrupted = False
        self.reward_parts = {}
        self.returns = []
        self.timings = {"plan": [], "control": [], "qp": [], "control_cycle": []}
        self.episode_ids = {}
        self.seen_episodes = set()

    def _on_step(self):
        for rank, info in enumerate(self.locals["infos"]):
            key = (rank, self.episode_ids.get(rank, 0))
            if key not in self.seen_episodes:
                append_json(
                    self.run / "scenarios.jsonl",
                    {"env_id": rank, "episode_id": key[1], **info["initial_scenario"]},
                )
                self.seen_episodes.add(key)
            self.physics += info["executed_substeps"]
            self.actions += 1
            if info.get("plan"):
                self.timings["plan"].append(info["plan"]["planning_time"])
            for tick in info["trace"]:
                self.timings["control_cycle"].append(tick["control_cycle_seconds"])
                for key, value in tick["reward_components"].items():
                    self.reward_parts[key] = self.reward_parts.get(key, 0) + value
                for key, field in (
                    ("control", "control_seconds"),
                    ("qp", "qp_seconds"),
                ):
                    if field in tick["diagnostics"]:
                        self.timings[key].append(tick["diagnostics"][field])
            append_json(
                self.run / "transitions.jsonl",
                {
                    "env_id": rank,
                    "episode_id": self.episode_ids.get(rank, 0),
                    "action_index": self.actions,
                    "action": self.locals["actions"][rank],
                    "mode": info["mode"],
                    "plan": info["plan"],
                    "trace": info["trace"],
                    "termination_reason": info["termination_reason"],
                },
            )
            if "episode_metrics" in info:
                # Each vector env carries its own terminal metrics. No shared success latch.
                append_json(
                    self.run / "episodes.jsonl",
                    {
                        "env_id": rank,
                        "episode_id": self.episode_ids.get(rank, 0),
                        **info["episode_metrics"],
                    },
                )
                self.episodes += 1
                self.episode_ids[rank] = self.episode_ids.get(rank, 0) + 1
                self.returns.append(info["episode_metrics"])
        if time.perf_counter() >= self.deadline:
            self.interrupted = True
            return False
        return True


def actor_vector(model):
    return torch.cat(
        [
            p.detach().flatten().cpu()
            for name, p in model.policy.named_parameters()
            if "policy_net" in name or "action_net" in name or "log_std" in name
        ]
    )


def train(
    config,
    profile="smoke",
    seed=7,
    root="runs",
    physics_budget=None,
    n_envs=1,
    wall_seconds=None,
):
    config.__post_init__()
    if config.method == "B3":
        raise ValueError("B3 is a rule baseline, not PPO")
    settings = dict(PROFILES[profile])
    budget = int(
        settings["physics_budget"] if physics_budget is None else physics_budget
    )
    wall = float(settings["wall_seconds"] if wall_seconds is None else wall_seconds)
    if budget < 1 or wall <= 0 or n_envs < 1:
        raise ValueError("budgets and env count must be positive")
    torch.set_num_threads(1)
    run = new_run(root, f"train_{config.method}_{profile}_seed{seed}")
    write_json(
        run / "config.json",
        {
            "env": config.to_dict(),
            "profile": profile,
            "seed": seed,
            "n_envs": n_envs,
            "physics_budget": budget,
            "wall_seconds": wall,
            "ppo": settings,
            "signature": signature(config),
        },
    )
    vector = make_vector(config, seed, n_envs)
    steps = max(2, settings["rollout_physics"] // (config.substeps * n_envs))
    model = PPO(
        "MlpPolicy",
        vector,
        n_steps=steps,
        batch_size=min(64, steps * n_envs),
        n_epochs=settings["epochs"],
        learning_rate=3e-4,
        gamma=config.gamma,
        gae_lambda=0.95 ** (config.substeps * config.control_dt),
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.set_logger(configure(str(run / "sb3"), ["csv"]))
    initial = actor_vector(model).clone()
    started = time.perf_counter()
    recorder = Recorder(run, started + wall)
    updates = 0
    while (recorder.physics < budget or updates < 2) and not recorder.interrupted:
        old = model._n_updates
        model.learn(
            total_timesteps=steps * n_envs,
            callback=recorder,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        if model._n_updates > old:
            updates += 1
            statistics = {
                key: float(value)
                for key, value in model.logger.name_to_value.items()
                if key.startswith("train/")
                and np.isscalar(value)
                and np.isfinite(value)
            }
            statistics["train/entropy"] = -statistics["train/entropy_loss"]
            append_json(
                run / "optimization.jsonl",
                {
                    "optimization_phase": updates,
                    "physics_steps": recorder.physics,
                    "actions": recorder.actions,
                    "wall_seconds": time.perf_counter() - started,
                    **statistics,
                },
            )
        if time.perf_counter() - started >= wall:
            recorder.interrupted = True
    training_seconds = time.perf_counter() - started
    delta = float(torch.linalg.vector_norm(actor_vector(model) - initial))
    model.save(run / "model.zip")
    vector.save(run / "normalization.pkl")
    # Check both stochastic-policy distribution parameters and deterministic actions on a batch.
    raw = np.stack([vector.venv.envs[0]._observation()] * 4)
    raw[1, 0] += 0.1
    raw[2, 16] += 0.3
    raw[3, 5] = 0
    normalized = vector.normalize_obs(raw)
    before = model.predict(normalized, deterministic=True)[0]
    loaded, normalizer = load_run(run, config)
    after = loaded.predict(normalizer.normalize_obs(raw), deterministic=True)[0]
    reload_ok = bool(
        np.array_equal(before, after)
        and torch.equal(actor_vector(model), actor_vector(loaded))
    )
    np.savez(run / "reload_probe.npz", observations=raw, before=before, after=after)
    summary = {
        "run_id": run.name,
        "method": config.method,
        "seed": seed,
        "requested_physics_budget": budget,
        "task_deadline_seconds": config.deadline,
        "physics_steps": recorder.physics,
        "budget_overshoot": recorder.physics - budget,
        "actions": recorder.actions,
        "simulation_seconds": recorder.physics * config.control_dt,
        "training_wall_seconds": training_seconds,
        "physics_steps_per_wall_second": recorder.physics / max(training_seconds, 1e-9),
        "optimization_phases": updates,
        "ppo_epoch_updates": model._n_updates,
        "actor_l2_change": delta,
        "reload_exact": reload_ok,
        "wall_budget_exceeded": recorder.interrupted,
        "episodes": recorder.episodes,
        "reward_components": recorder.reward_parts,
        "network_parameters": sum(p.numel() for p in model.policy.parameters()),
        "gamma_low": config.gamma_low,
        "gamma_policy": config.gamma,
        "policy_frequency_hz": 1 / (config.control_dt * config.substeps),
        "gae_lambda_policy": 0.95 ** (config.substeps * config.control_dt),
        "training_pass": updates >= 2
        and delta > 0
        and reload_ok
        and not recorder.interrupted,
        "conclusion": "engineering smoke only; no architecture advantage established",
    }
    for key, values in recorder.timings.items():
        summary[key + "_timing"] = timing_summary(
            values, config.decision_dt if key == "plan" else config.control_dt
        )
    write_json(
        run / "unfinished_episodes.json",
        [
            {"env_id": i, "censored_at_budget_end": True, **e.summary("budget_end")}
            for i, e in enumerate(vector.venv.envs)
            if e.episode["physics_steps"] > 0
        ],
    )
    write_json(run / "training_summary.json", summary)
    normalizer.close()
    vector.close()
    print(
        json.dumps({"run": str(run), **summary}, default=jsonable, ensure_ascii=False),
        flush=True,
    )
    return run, summary


def load_run(run, config=None):
    run = Path(run)
    if not (run / "config.json").exists() or not (run / "normalization.pkl").exists():
        raise ValueError(
            "incompatible legacy checkpoint: physical-space signature and normalization are required; retrain"
        )
    metadata = json.loads((run / "config.json").read_text(encoding="utf-8"))
    saved = EnvConfig.from_dict(metadata["env"])
    config = config or saved
    if metadata.get("signature") != signature(config):
        raise ValueError(
            "incompatible policy/vehicle/action semantics; retrain (explicit transfer protocol required)"
        )
    raw = DummyVecEnv([lambda: NavigationEnv(config)])
    normalizer = VecNormalize.load(run / "normalization.pkl", raw)
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(run / "model.zip", env=normalizer, device="cpu")
    return model, normalizer


def evaluate(config, model_run=None, episodes=4, root="runs", seed=7, policy="learned"):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if policy not in ("learned", "rule", "cycle"):
        raise ValueError("unknown policy")
    if policy == "learned" and not model_run:
        raise ValueError("a trained checkpoint is required")
    if policy == "rule" and config.method != "B3":
        raise ValueError("rule policy is B3 only")
    if policy == "cycle" and config.method == "B1":
        raise ValueError("cycle smoke requires discrete actions")
    run = new_run(root, f"eval_{config.method}_{policy}_{config.split}_seed{seed}")
    write_json(
        run / "config.json",
        {
            "env": config.to_dict(),
            "model_run": str(model_run) if model_run else None,
            "policy": policy,
            "episodes": episodes,
            "seed": seed,
        },
    )
    model = normalizer = None
    if model_run:
        model, normalizer = load_run(model_run, config)
    means_before = normalizer.obs_rms.mean.copy() if normalizer else None
    records = []
    start = time.perf_counter()
    scenarios = (
        ["empty", "static", "narrow", "dynamic"]
        if config.split != "test"
        else ["empty", "unseen", "narrow", "dynamic"]
    )
    for i in range(episodes):
        c = EnvConfig.from_dict(config.to_dict())
        if c.scenario == "mixed":
            c.scenario = scenarios[i % len(scenarios)]
        env = NavigationEnv(c)
        scenario_seed = (
            {"train": 0, "validation": 1000000, "test": 2000000}[c.split]
            + seed * 100
            + i
        )
        obs, _ = env.reset(seed=scenario_seed)
        append_json(
            run / "scenarios.jsonl",
            {
                "episode": i,
                "seed": scenario_seed,
                "name": env.scenario_name,
                "initial": asdict(env.state),
                "goal": env.goal,
                "static": env.world.static,
                "dynamic": [asdict(o) for o in env.world.dynamic],
            },
        )
        episode_started = time.perf_counter()
        action_count = 0
        while True:
            if policy == "learned":
                action = model.predict(
                    normalizer.normalize_obs(obs), deterministic=True
                )[0]
            elif policy == "rule":
                action = rule_action(env)
            else:
                action = 16  # explicitly scripted B2 plumbing smoke, never called a trained PPO
            obs, reward, terminated, truncated, info = env.step(action)
            append_json(
                run / "transitions.jsonl",
                {
                    "episode": i,
                    "action": action,
                    "plan": info["plan"],
                    "trace": info["trace"],
                },
            )
            action_count += 1
            if terminated or truncated:
                record = {
                    "episode": i,
                    "seed": scenario_seed,
                    "policy": policy,
                    **info["episode_metrics"],
                    "wall_seconds": time.perf_counter() - episode_started,
                }
                records.append(record)
                append_json(run / "episodes.jsonl", record)
                break
        env.close()
    summary = {
        "run_id": run.name,
        "method": config.method,
        "policy": policy,
        "split": config.split,
        "episodes": episodes,
        "task_deadline_seconds": config.deadline,
        "success_rate": float(np.mean([r["success"] for r in records])),
        "failure_counts": {
            reason: sum(r["termination_reason"] == reason for r in records)
            for reason in ("collision", "out_of_bounds", "deadline")
        },
        "mean_progress_m": float(np.mean([r["progress"] for r in records])),
        "wall_seconds": time.perf_counter() - start,
        "physics_steps": sum(r["physics_steps"] for r in records),
        "normalization_frozen": (
            bool(np.array_equal(means_before, normalizer.obs_rms.mean))
            if normalizer
            else None
        ),
        "whole_episode_timing": timing_summary(
            [r["wall_seconds"] for r in records], config.deadline
        ),
        "conclusion": "smoke data; failed episodes retained; not evidence of hierarchical superiority",
    }
    if normalizer:
        normalizer.close()
    write_json(run / "evaluation_summary.json", summary)
    print(
        json.dumps({"run": str(run), **summary}, default=jsonable, ensure_ascii=False),
        flush=True,
    )
    return run, summary


def report(root="runs", output="reports"):
    dest = new_run(output, "report")
    training = []
    evaluation = []
    for path in sorted(Path(root).glob("*/training_summary.json")):
        row = {"source": str(path), **json.loads(path.read_text(encoding="utf-8"))}
        row["task_deadline_seconds"] = json.loads(
            (path.parent / "config.json").read_text(encoding="utf-8")
        )["env"]["deadline"]
        training.append(row)
    for path in sorted(Path(root).glob("*/evaluation_summary.json")):
        row = {"source": str(path), **json.loads(path.read_text(encoding="utf-8"))}
        row["task_deadline_seconds"] = json.loads(
            (path.parent / "config.json").read_text(encoding="utf-8")
        )["env"]["deadline"]
        evaluation.append(row)
    write_json(
        dest / "results.json",
        {
            "training": training,
            "evaluation": evaluation,
            "formal_independent_seed_intervals": None,
            "interval_missing_reason": "formal multi-seed experiment not performed; smoke seeds are not evidence",
        },
    )
    lines = [
        "# 实测运行报告",
        "",
        "仅记录工程验证，不据短训练宣称分层优势。所有失败回合保留。",
        "",
        "8 秒早期运行仅用于接口调试（路线无法在该截止时间完成），不与 30 秒任务合并统计。",
        "|run_id / 方法|截止秒|物理步|优化阶段|Actor L2变化|重载一致|训练秒|",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in training:
        lines.append(
            f"|{row['run_id']} / {row['method']}|{row['task_deadline_seconds']}|{row['physics_steps']}|{row['optimization_phases']}|{row['actor_l2_change']:.6f}|{row['reload_exact']}|{row['training_wall_seconds']:.2f}|"
        )
    lines += [
        "",
        "|run_id / 方法/策略|截止秒|划分|回合|成功率|失败原因|",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in evaluation:
        lines.append(
            f"|{row['run_id']} / {row['method']}/{row['policy']}|{row['task_deadline_seconds']}|{row['split']}|{row['episodes']}|{row['success_rate']:.3f}|{row['failure_counts']}|"
        )
    (dest / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(dest)
    return dest

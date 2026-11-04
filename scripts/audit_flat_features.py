"""Measured flat-PPO rollout and nonexecuted feature-sensitivity diagnostics."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint, source, output = (Path(x).resolve() for x in (args.checkpoint, args.source, args.output))
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.experiments import load_run, jsonable
    torch.set_num_threads(1)
    config = EnvConfig.from_dict(json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["env"])
    if config.method != "B1":
        raise ValueError("This diagnostic requires direct continuous controls")
    model, normalizer = load_run(checkpoint)
    mean, var = normalizer.obs_rms.mean.copy(), normalizer.obs_rms.var.copy()
    config.scenario, config.split = "empty", "validation"
    episodes, records = [], []
    with gzip.open(output / "rollout.jsonl.gz", "wt", encoding="utf-8") as stream:
        try:
            for seed in (7301, 7302):
                env = NavigationEnv(config)
                try:
                    obs, _ = env.reset(seed=seed, options={"diagnostic_route_length": 23})
                    while not env.done:
                        normalized = normalizer.normalize_obs(obs[None])
                        tensor = torch.as_tensor(normalized, dtype=torch.float32)
                        with torch.no_grad():
                            distribution = model.policy.get_distribution(tensor).distribution
                            extractor = model.policy.mlp_extractor
                            if hasattr(extractor, "policy_net"):
                                first = torch.tanh(extractor.policy_net[0](tensor))
                                first_layer_scope = "all_observation_features"
                            elif hasattr(extractor, "policy_net_context"):
                                context = torch.cat((tensor[:, :351], tensor[:, 599:]), dim=1)
                                first = torch.tanh(extractor.policy_net_context[0](context))
                                first_layer_scope = "global_and_outcome_context_only"
                            else:
                                raise ValueError("Unsupported actor diagnostic architecture")
                            latent = extractor.forward_actor(tensor)
                            diagnostic = {"seed": seed, "time": env.state.time,
                                "core_clipped_fraction": float(np.mean(np.abs(normalized[0, :351]) >= 9.999)),
                                "first_tanh_saturated_fraction": float((first.abs() > .98).float().mean()),
                                "actor_tanh_saturated_fraction": float((latent.abs() > .98).float().mean()),
                                "raw_action_mean": distribution.mean[0].numpy(),
                                "action_std": distribution.stddev[0].numpy()}
                            if normalized.shape[1] > 351:
                                # Sensitivity probe only: these counterfactual means
                                # never control the plant or enter its observation.
                                probe = tensor.clone()
                                probe[:, 351:] = 0
                                removed = model.policy.get_distribution(probe).distribution.mean
                                diagnostic["unexecuted_extra_zero_action_mean"] = removed[0].numpy()
                                diagnostic["extra_zero_action_mean_l2"] = float(torch.linalg.vector_norm(removed - distribution.mean))
                        action, _ = model.predict(normalized, deterministic=True)
                        obs, _, _, _, info = env.step(action[0])
                        diagnostic["executed_action"] = action[0]
                        stream.write(json.dumps({**diagnostic, "trace": info["trace"]}, default=jsonable) + "\n")
                        records.append(diagnostic)
                    episodes.append({"seed": seed, **info["episode_metrics"]})
                finally:
                    env.close()
            np.testing.assert_array_equal(mean, normalizer.obs_rms.mean)
            np.testing.assert_array_equal(var, normalizer.obs_rms.var)
        finally:
            normalizer.close()
    metrics = {}
    for key in ("core_clipped_fraction", "first_tanh_saturated_fraction", "actor_tanh_saturated_fraction", "extra_zero_action_mean_l2"):
        values = [r[key] for r in records if key in r]
        metrics[key] = {"mean": float(np.mean(values)), "max": float(np.max(values))} if values else None
    means = np.array([r["raw_action_mean"] for r in records])
    result = {"scope": "two declared empty-task development replays; zeroed features are unexecuted sensitivity probes, not trained ablations",
        "checkpoint_name": checkpoint.name, "model_sha256": hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest(),
        "source_files": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                         for name in ("navstack/env.py", "navstack/config.py", "navstack/candidate_policy.py") if (source / name).exists()},
        "observation_size": int(model.observation_space.shape[0]), "flat_hold": config.flat_hold,
        "first_layer_scope": first_layer_scope,
        "normalization_frozen": True, "decisions": len(records), "metrics": metrics,
        "mean_action_clipped_fraction": (np.abs(means) >= 1).mean(axis=0).tolist(),
        "action_std_mean": np.array([r["action_std"] for r in records]).mean(axis=0).tolist(), "episodes": episodes}
    (output / "results.json").write_text(json.dumps(result, indent=2, default=jsonable), encoding="utf-8")
    print(json.dumps({"size": result["observation_size"], "hold": config.flat_hold, "metrics": metrics,
                      "clipped_actions": result["mean_action_clipped_fraction"], "std": result["action_std_mean"],
                      "outcomes": [(e["success"], e["progress"], e["jerk_rms"]) for e in episodes]}))


if __name__ == "__main__":
    main()

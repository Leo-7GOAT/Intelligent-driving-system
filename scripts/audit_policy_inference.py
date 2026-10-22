"""Diagnose training/evaluation mismatch without changing a checkpoint."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import load_run, write_json, jsonable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--route-length", type=float, default=6)
    p.add_argument("--scenario")
    p.add_argument("--deadline", type=float)
    args = p.parse_args()
    torch.set_num_threads(1)
    checkpoint, output = Path(args.checkpoint), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = EnvConfig.from_dict(json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["env"])
    if args.scenario:
        cfg.scenario = args.scenario
    if args.deadline:
        cfg.deadline = args.deadline
    model, normalizer = load_run(checkpoint)
    before = normalizer.obs_rms.mean.copy()
    rows = []
    with gzip.open(output / "trace.jsonl.gz", "wt", encoding="utf-8") as stream:
        for deterministic in (True, False):
            for split in ("train", "validation"):
                cfg.split = split
                for seed in range(7301, 7309):
                    torch.manual_seed(seed)
                    env = NavigationEnv(cfg)
                    obs, _ = env.reset(seed=seed, options={"diagnostic_route_length": args.route_length})
                    initial = env.initial_scenario
                    actions = 0
                    while True:
                        if model.use_sde and actions % 10 == 0:
                            model.policy.reset_noise(1)
                        action, _ = model.predict(normalizer.normalize_obs(obs[None]), deterministic=deterministic)
                        obs, _, done, _, info = env.step(action[0]); actions += 1
                        stream.write(json.dumps({"deterministic": deterministic, "split": split, "seed": seed, "action": action[0], "info": info}, default=jsonable) + "\n")
                        if done:
                            rows.append({"deterministic": deterministic, "split": split, "seed": seed,
                                         "initial": initial, "terminal_state": info["trace"][-1]["state"], **info["episode_metrics"]})
                            break
    np.testing.assert_array_equal(before, normalizer.obs_rms.mean)
    normalizer.close()
    groups = [{"deterministic": d, "split": s, "success": sum(r["success"] for r in rows if r["deterministic"] == d and r["split"] == s), "n": 8}
              for d in (True, False) for s in ("train", "validation")]
    write_json(output / "results.json", {"checkpoint": str(checkpoint), "epoch": time.time(), "groups": groups, "episodes": rows,
               "route_length": args.route_length, "scenario": cfg.scenario, "deadline": cfg.deadline,
               "scope": "development inference diagnostic; stochastic gSDE noise resampled every 10 actions; frozen normalization"})
    print(json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()

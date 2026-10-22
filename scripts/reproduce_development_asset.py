"""Verify an extracted release asset, rebuild its figure and rerun learned policies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full", action="store_true", help="Rerun all10 conditions per model instead of3 representative cases")
    args = parser.parse_args()
    asset, output = Path(args.asset).resolve(), Path(args.output).resolve()
    manifest = json.loads((asset / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (asset / name).resolve()
        if not path.is_relative_to(asset):raise ValueError("Manifest path escapes asset")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:raise ValueError("Hash mismatch: " + name)
    sys.path.insert(0, str(asset / "source"))
    import numpy as np
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.experiments import load_run, jsonable
    output.mkdir(parents=True, exist_ok=False)
    rows, endpoints = [], []
    for entry in manifest["models"]:
        label = entry["label"]
        saved = json.loads((asset / "evidence" / label / "evaluation.json").read_text(encoding="utf-8"))
        expected = saved["episodes"]
        endpoints.append({"label": label, "successes": sum(e["success"] for e in expected), "episodes": len(expected),
                          "progress_mean_m": float(np.mean([e["progress"] for e in expected]))})
        checkpoint = asset / "models" / label
        config = EnvConfig.from_dict(json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["env"])
        model, normalizer = load_run(checkpoint)
        mean, variance = normalizer.obs_rms.mean.copy(), normalizer.obs_rms.var.copy()
        try:
            for case in expected:
                if not args.full and case["case_id"] not in (0, 1, 4):continue
                config.split, config.scenario = "validation", case["scenario"]
                env = NavigationEnv(config)
                obs, _ = env.reset(seed=case["seed"])
                while not env.done:
                    action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
                    obs, _, _, _, info = env.step(action)
                metrics = info["episode_metrics"]
                matched = (metrics["success"] == case["success"] and metrics["termination_reason"] == case["termination_reason"]
                           and abs(metrics["progress"]-case["progress"]) < .02)
                rows.append({"label": label, "case_id": case["case_id"], "scenario": case["scenario"],
                             "matches_recorded_outcome": matched, **metrics})
                env.close()
            np.testing.assert_array_equal(mean, normalizer.obs_rms.mean)
            np.testing.assert_array_equal(variance, normalizer.obs_rms.var)
        finally:normalizer.close()
    (output / "reproduction.json").write_text(json.dumps({"hashes_verified": len(manifest["files"]),
          "normalization_frozen": True, "rows": rows, "saved_raw_endpoints": endpoints}, indent=2, default=jsonable), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([e["label"] for e in endpoints], [e["successes"]/e["episodes"] for e in endpoints])
    ax.set(ylim=(0, 1), ylabel="Safe completion fraction", title="200k development endpoint; one training seed")
    for i, e in enumerate(endpoints):ax.text(i, e["successes"]/e["episodes"]+.02, f"{e['successes']}/{e['episodes']}", ha="center")
    fig.tight_layout()
    fig.savefig(output / "development_endpoint.png", dpi=160)
    plt.close(fig)
    if not all(r["matches_recorded_outcome"] for r in rows):raise RuntimeError("Replay differs; inspect reproduction.json, do not claim verification")
    print(json.dumps({"verified_files": len(manifest["files"]), "replayed_episodes": len(rows), "all_outcomes_match": True}))


if __name__ == "__main__":
    main()

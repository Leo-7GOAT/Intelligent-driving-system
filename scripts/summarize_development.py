"""Rebuild sanity-learning evidence from immutable checkpoints and raw episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import statistics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.experiments import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    args = p.parse_args()
    batch = Path(args.batch)
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    mismatches = [name for name, digest in protocol["source_hashes"].items()
                  if hashlib.sha256((batch / "source" / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError("Changed source: " + repr(mismatches))
    rows = []
    for job in protocol["jobs"]:
        run = batch / "sanity_v6" / job["id"]
        for file in run.glob("validation_*.json"):
            result = json.loads(file.read_text(encoding="utf-8"))
            checkpoint = Path(result["checkpoint"])
            summary = json.loads((checkpoint / "training_summary.json").read_text(encoding="utf-8"))
            actual_hash = hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest()
            if actual_hash != result["model_sha256"]:
                raise RuntimeError("Evaluation model hash mismatch")
            episodes = result["episodes"]
            rows.append({"job": job["id"], "seed": job["seed"], "physical_steps": summary["physics_steps"],
                         "nominal_steps": int(file.stem.split("_")[-1]), "safe_completion": result["safe_completion"],
                         "conditions": len(episodes), "progress_mean_m": statistics.mean(e["progress"] for e in episodes),
                         "events": {reason: sum(e["termination_reason"] == reason for e in episodes)
                                    for reason in ("success", "collision", "out_of_bounds", "deadline")},
                         "optimizer_steps": summary["optimizer_steps"], "model_sha256": actual_hash,
                         "result": str(file.relative_to(batch)), "checkpoint": str(checkpoint.relative_to(batch.resolve()))})
    rows.sort(key=lambda r: (r["job"], r["physical_steps"]))
    analysis = batch / "analysis"
    analysis.mkdir(exist_ok=True)
    write_json(analysis / "sanity_metrics.json", {"scope": "single-seed development; no confidence interval or research acceptance claim",
               "source_hashes_checked": len(protocol["source_hashes"]), "rows": rows})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for job in protocol["jobs"]:
        series = [r for r in rows if r["job"] == job["id"]]
        for ax, metric in zip(axes, ("safe_completion", "progress_mean_m")):
            ax.plot([r["physical_steps"] for r in series], [r[metric] for r in series], "o-", label=job["id"])
    axes[0].set(ylabel="Safe completion fraction", ylim=(-.03, 1.03))
    axes[1].set(ylabel="Mean measured progress (m)")
    for ax in axes:
        ax.set_xlabel("Actual training physical steps")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle("Development sanity: 6 m empty-route parking, one training seed\nNot independent confirmation or a hierarchy comparison", fontsize=10)
    fig.tight_layout()
    fig.savefig(analysis / "sanity_learning.png", dpi=160)
    plt.close(fig)
    lines = ["# Development sanity evidence", "", "Single training seed; no confidence interval or research acceptance claim.", "",
             "| Configuration | Actual physical steps | Completion | Mean progress m |", "|---|---:|---:|---:|"]
    for r in rows:
        lines.append("| %s | %d | %d/%d | %.3f |" % (r["job"], r["physical_steps"], r["events"]["success"], r["conditions"], r["progress_mean_m"]))
    lines += ["", "Model hashes and frozen source hashes verified; raw evaluation traces and optimizer/checkpoint metadata remain in this batch."]
    (analysis / "sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "source_files_verified": len(protocol["source_hashes"])}))


if __name__ == "__main__":
    main()

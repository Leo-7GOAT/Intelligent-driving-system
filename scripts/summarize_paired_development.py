"""Verify source/model provenance and rebuild all completed development milestones."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    batch = Path(args.batch).resolve()
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol["source_hashes"].items():
        assert hashlib.sha256((batch / "source" / name).read_bytes()).hexdigest() == expected, name
    rows = []
    for job in protocol["jobs"]:
        run = batch / protocol["run_group"] / job["id"]
        for result_path in run.glob("validation_*.json"):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = Path(result["checkpoint"])
            assert hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest() == result["model_sha256"]
            training = json.loads((checkpoint / "training_summary.json").read_text(encoding="utf-8"))
            episodes = result["episodes"]
            expected_cases = protocol["development_cases"]
            assert len(episodes) == len(expected_cases)
            for episode, case in zip(episodes, expected_cases):
                assert all(episode[key] == value for key, value in case.items())
            for group in ["all"] + sorted({e["scenario"] + ("_" + str(e["route_m"]) + "m" if "route_m" in e else "") for e in episodes}):
                subset = episodes if group == "all" else [e for e in episodes if e["scenario"] + ("_" + str(e["route_m"]) + "m" if "route_m" in e else "") == group]
                rows.append({"job": job["id"], "nominal_steps": int(result_path.stem.split("_")[-1]),
                             "physics_steps": training["physics_steps"], "group": group, "episodes": len(subset),
                             "successes": sum(e["success"] for e in subset),
                             "progress_m": statistics.mean(e["progress"] for e in subset),
                             "intervention_fraction": statistics.mean(e["intervention_rate"] for e in subset),
                             "failure_reasons": {reason: sum(e["termination_reason"] == reason for e in subset)
                                                 for reason in sorted({e["termination_reason"] for e in subset})},
                             "model_sha256": result["model_sha256"], "result": str(result_path.relative_to(batch))})
    rows.sort(key=lambda r: (r["job"], r["physics_steps"], r["group"]))
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    (output / "development_metrics.json").write_text(json.dumps({"scope": "single-seed development; no independent confirmation or confidence interval", "rows": rows}, indent=2), encoding="utf-8")
    lines = ["# Paired development evidence", "", "Single training seed. All completed milestone cases retained; no confidence interval or superiority claim.", "",
             "| Job | Physical steps | Group | Completion | Progress m | Intervention fraction |", "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append("| {job} | {physics_steps} | {group} | {successes}/{episodes} | {progress_m:.2f} | {intervention_fraction:.3f} |".format(**row))
    (output / "development_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for job in protocol["jobs"]:
        series = [r for r in rows if r["job"] == job["id"] and r["group"] == "all"]
        for ax, values in zip(axes, ([r["successes"]/r["episodes"] for r in series], [r["progress_m"] for r in series], [r["intervention_fraction"] for r in series])):
            ax.plot([r["physics_steps"] for r in series], values, "o-", label=job["id"])
    for ax, label in zip(axes, ("Safe completion", "Mean progress (m)", "Mean intervention fraction")):
        ax.set(xlabel="Actual training physical steps", ylabel=label)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    axes[0].set_ylim(-.03, 1.03)
    axes[2].set_ylim(-.03, 1.03)
    fig.suptitle("Development only: one independent training seed")
    fig.tight_layout()
    fig.savefig(output / "development_learning.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"rows": len(rows), "source_files_verified": len(protocol["source_hashes"])}))


if __name__ == "__main__":
    main()

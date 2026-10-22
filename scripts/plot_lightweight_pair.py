"""Prespecified seed11, first case of each final scenario; measured states only."""

import argparse, gzip, json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load(run):
    run = Path(run)
    paths = {}
    for f in sorted(run.glob("transitions*.jsonl.gz")):
        current = {}
        with gzip.open(f, "rt", encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                current.setdefault(r["case_id"], []).extend(
                    (t["state"]["x"], t["state"]["y"]) for t in r["trace"]
                )
        paths.update(current)
    scenes = {
        r["case_id"]: r
        for line in (run / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        if (r := json.loads(line))
    }
    episodes = {
        r["case_id"]: r
        for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if (r := json.loads(line))
    }
    return paths, scenes, episodes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    args = p.parse_args()
    batch = Path(args.batch)
    markers = [
        json.loads(
            (batch / "formal_v5" / f"{m}_seed11" / "final_primary.json").read_text(
                encoding="utf-8"
            )
        )
        for m in ("H", "B1")
    ]
    if not all(r["complete"] for r in markers):
        raise ValueError("Paired complete test required")
    h, scenes, he = load(markers[0]["run"])
    b, bs, be = load(markers[1]["run"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, ax in enumerate(axes.flat):
        assert scenes[i]["case"] == bs[i]["case"]
        scene = scenes[i]["initial_scenario"]
        for x, y, xx, yy in scene["static"]:
            ax.add_patch(Rectangle((x, y), xx - x, yy - y, color="gray", alpha=0.5))
        for path, name, color, result in (
            (h[i], "H", "#2563eb", he[i]),
            (b[i], "B1", "#dc2626", be[i]),
        ):
            ax.plot(
                *zip(*path),
                color=color,
                label=name + ": " + result["termination_reason"],
                lw=2,
            )
            if name == "H":
                ax.scatter(*path[-1], facecolors="none", edgecolors=color, s=65, linewidths=1.4)
            else:
                ax.scatter(*path[-1], color=color, marker="x", s=30)
        ax.scatter(
            *scene["goal"][:2], marker="*", color="green", s=100, label="Parking goal"
        )
        for obstacle in scene["dynamic"]:
            ax.scatter(obstacle["x"], obstacle["y"], color="black", marker="x")
        ax.set(
            xlim=(0, 34),
            ylim=(0, 24),
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"Case {i}: {scenes[i]['case']['scenario']}",
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Paired 1M PPO, seed 11; prespecified first four final cases, including failures"
    )
    fig.tight_layout()
    fig.savefig(batch / "analysis/lightweight_final_trajectories.png", dpi=170)

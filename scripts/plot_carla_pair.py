"""Plot measured rear-axle states from paired CARLA PPO logs, never reference trajectories."""

import argparse, gzip, json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load(folder):
    folder = Path(folder)
    episodes = [
        json.loads(s)
        for s in (folder / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    attempts = {
        r["case_id"]: r["attempt"] for r in episodes if r.get("status") == "completed"
    }
    paths = {}
    frames = {}
    for file in sorted(folder.glob("transitions*.jsonl.gz")):
        session_paths = {}
        session_frames = {}
        with gzip.open(file, "rt", encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                i = r["case_id"]
                if r["attempt"] != attempts.get(i):
                    continue
                session_paths.setdefault(i, []).extend(
                    (t["state"]["x"], t["state"]["y"]) for t in r["trace"]
                )
                session_frames.setdefault(i, []).extend(
                    t["backend_frame"] for t in r["trace"]
                )
        # A resumed budget-censored case restarts physically; keep its latest complete path.
        paths.update(session_paths)
        frames.update(session_frames)
    scenes = {
        r["case_id"]: r
        for s in (folder / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        if (r := json.loads(s))["attempt"] == attempts.get(r["case_id"])
    }
    return paths, scenes, frames


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h", required=True)
    p.add_argument("--b1", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    hp, scenes, hf = load(args.h)
    bp, other, bf = load(args.b1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    indices = []
    seen = set()
    for i, record in sorted(scenes.items()):
        if record["scene"]["name"] not in seen:
            indices.append(i)
            seen.add(record["scene"]["name"])
    if len(indices) != 3:
        raise ValueError("Three completed scenario types required")
    for i, ax in zip(indices, axes):
        scene = scenes[i]["scene"]
        initial = scenes[i]["initial_state"]
        assert (
            scene["condition"] == other[i]["scene"]["condition"]
            and scene["name"] == other[i]["scene"]["name"]
        )
        for a, b, c, d in scene["static"]:
            ax.add_patch(Rectangle((a, b), c - a, d - b, color="gray", alpha=0.5))
        for path, label, color in (
            (hp.get(i, []), "H PPO measured", "#1769aa"),
            (bp.get(i, []), "B1 PPO measured", "#ef6c00"),
        ):
            if path:
                ax.plot(*zip(*path), label=label, color=color, linewidth=2)
                ax.scatter(*path[-1], color=color, s=32)
        ax.scatter(
            initial["goal"][0],
            initial["goal"][1],
            marker="*",
            s=140,
            color="green",
            label="Parking goal",
        )
        if initial["dynamic"]:
            o = initial["dynamic"][0]
            ax.scatter(
                o["x"],
                o["y"],
                marker="x",
                color="black",
                label="Pedestrian initial position",
            )
        ax.set(
            xlim=(0, scene["width"]),
            ylim=(0, scene["height"]),
            xlabel="Road-local x (m)",
            ylabel="Road-local y (m)",
            title=f"{scene['name']} / condition {scene['condition']} / actual CARLA rear axle",
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")
    transfer = json.loads((Path(args.h) / "config.json").read_text(encoding="utf-8"))[
        "transfer"
    ]
    title = (
        "PPO after matched CARLA fine-tuning"
        if transfer.startswith("carla_finetuned")
        else "Frozen PPO with calibrated low-level adapter"
    )
    split = json.loads((Path(args.h) / "config.json").read_text(encoding="utf-8"))[
        "split"
    ]
    fig.suptitle(title + f"; failures retained ({split})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)

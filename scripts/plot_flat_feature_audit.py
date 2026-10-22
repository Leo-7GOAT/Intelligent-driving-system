"""Rebuild diagnostic plots from every saved actual control tick."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    labels = (("v9_full23", "351-input baseline", "#0072B2"),
              ("v13_hold1", "605 inputs, hold1", "#D55E00"),
              ("v14_hold10", "605 inputs, hold10", "#009E73"))
    for folder, label, color in labels:
        grouped = {}
        with gzip.open(root / folder / "rollout.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                group = grouped.setdefault(record["seed"], {"ticks": [], "decisions": []})
                group["ticks"].extend(record["trace"])
                group["decisions"].append(record)
        for index, group in enumerate(grouped.values()):
            ticks, decisions = group["ticks"], group["decisions"]
            times = [r["state"]["time"] for r in ticks]
            style = "-" if index == 0 else "--"
            options = dict(color=color, linestyle=style, alpha=.8, linewidth=1,
                           label=label if index == 0 else None)
            axes[0, 0].plot(times, [r["state"]["x"]-4 for r in ticks], **options)
            axes[0, 1].plot(times, [r["state"]["v"] for r in ticks], **options)
            axes[1, 0].plot(times, [r["state"]["acceleration"] for r in ticks], **options)
            axes[1, 1].plot([r["time"] for r in decisions],
                            [100*r["first_tanh_saturated_fraction"] for r in decisions], **options)
    for ax, ylabel in zip(axes.ravel(), ("Forward travel x - start (m)", "Actual speed (m/s)",
                                        "Actual acceleration (m/s²)", "First actor tanh saturation (%)")):
        ax.set(xlabel="Actual physical time (s)", ylabel=ylabel)
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("300k development diagnostics: two declared empty-task seeds per model\nEnds are not padded; input-zeroing probes never control the vehicle")
    fig.tight_layout()
    target = root / "flat_control_diagnostics.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()

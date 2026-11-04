"""Plot completed seed-level final outcomes without filling unrun seeds with zeros."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(records, output):
    records = [
        r for r in records if r["backend"] == "lightweight" and r["name"] in ("H", "B1")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    dimensions = (
        ("safe_completion", "Safe completion", (-0.04, 1.04)),
        ("progress_m", "All-case progress (m)", None),
        (
            "physical_step_intervention_rate",
            "Intervention / control tick",
            (-0.04, 1.04),
        ),
    )
    for row_index, label in enumerate(("primary", "generalization")):
        for column, (metric, title, limits) in enumerate(dimensions):
            ax = axes[row_index, column]
            tick_labels = []
            for index, (name, color) in enumerate(
                (("H", "#2563eb"), ("B1", "#dc2626"))
            ):
                group = sorted(
                    (r for r in records if r["label"] == label and r["name"] == name),
                    key=lambda r: r["seed"],
                )
                values = [r["metrics"][metric] for r in group]
                tick_labels.append(f"{name}\nn={len(group)}/5 seeds")
                if not values:
                    ax.text(
                        index,
                        0.5,
                        "Pending",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        color="gray",
                        fontsize=9,
                    )
                    continue
                offsets = (
                    np.linspace(-0.24, 0.24, len(values)) if len(values) > 1 else [0]
                )
                ax.scatter(
                    index + np.asarray(offsets),
                    values,
                    color=color,
                    s=26,
                    alpha=0.75,
                    zorder=3,
                )
                for x, value, record in zip(index + np.asarray(offsets), values, group):
                    ax.annotate(
                        str(record["seed"]),
                        (x, value),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                        color=color,
                    )
                deviation = np.std(values, ddof=1) if len(values) > 1 else 0
                ax.errorbar(
                    index,
                    np.mean(values),
                    yerr=deviation,
                    fmt="_",
                    markersize=20,
                    capsize=5,
                    color="black",
                    lw=1.3,
                    zorder=4,
                )
            ax.set_xticks([0, 1], tick_labels)
            ax.set_xlim(-0.55, 1.55)
            ax.set_title(f"{label.capitalize()}: {title}", fontsize=10)
            if limits:
                ax.set_ylim(*limits)
            else:
                ax.margins(y=0.2)
            ax.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "Completed 1M independent tests: seed values, mean and sample SD\nMissing seeds remain pending; paired confidence intervals are reported separately",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    batch = Path(parser.parse_args().batch)
    plot(
        json.loads(
            (batch / "analysis/measured_final_metrics.json").read_text(encoding="utf-8")
        ),
        batch / "analysis/final_seed_metrics.png",
    )

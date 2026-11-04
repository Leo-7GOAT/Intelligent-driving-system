"""Compare original and repaired candidate geometry on the same development probes."""

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    args = p.parse_args()
    batch = Path(args.batch)
    datasets = {}
    for name, file in (
        ("Original", "learning_audit.json"),
        ("Terminal prefix repair", "action_audit_formal_v4.json"),
    ):
        records = json.loads((batch / file).read_text(encoding="utf-8"))
        counts = defaultdict(Counter)
        for row in records["rows"]:
            if "actions" in row:
                counts[(row["scenario"], row["probe"])].update(
                    a["mode"] for a in row["actions"] if a["action"] != 0
                )
        datasets[name] = counts
    keys = list(datasets["Original"])
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for offset, (name, data) in zip((-0.18, 0.18), datasets.items()):
        ax.bar(
            x + offset,
            [data[k]["reference"] / sum(data[k].values()) for k in keys],
            0.36,
            label=name,
        )
    ax.set_xticks(x, ["/".join(k) for k in keys], rotation=40, ha="right")
    ax.set(ylabel="Plannable non-wait candidate fraction", ylim=(0, 1.05))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.suptitle(
        "31-action development audit: three fixed seeds, all probe cases retained",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(batch / "analysis/action_feasibility_audit.png", dpi=170)
    result = {
        name: {"/".join(k): dict(v) for k, v in data.items()}
        for name, data in datasets.items()
    }
    (batch / "analysis/action_feasibility_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

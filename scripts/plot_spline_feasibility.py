"""Display every diagnostic condition, including unresolved geometry."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def main(batch, diagnostic="posthoc_spline_feasibility"):
    batch = Path(batch)
    output = batch / "analysis" / diagnostic
    data = json.loads((output / "results.json").read_text(encoding="utf-8"))
    marker = json.loads((batch / "formal_v5/H_seed11/final_primary.json").read_text(encoding="utf-8"))
    scenes = {r["case_id"]: r["initial_scenario"] for line in
              (Path(marker["run"]) / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
              if (r := json.loads(line))["case"]["scenario"] == "unseen"}
    traces = {}
    with gzip.open(output / "physical_trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            traces.setdefault(row["case_id"], []).append((row["state"]["x"], row["state"]["y"]))
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)
    for ax, result in zip(axes.flat, data["results"]):
        case = result["case_id"]
        scene = scenes[case]
        for x0,y0,x1,y1 in scene["static"]:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color="gray",alpha=.6))
        path = output / f"path_case{case}.npy"
        if path.exists():
            points = np.load(path)
            ax.plot(points[:,0],points[:,1],"--",color="black",lw=1,label="Validated spline")
        if case in traces:
            points = np.array(traces[case])
            ax.plot(points[:,0],points[:,1],color="#168350",lw=2,label="MPC / actual plant")
            ax.scatter(*points[-1],marker="o",facecolors="none",edgecolors="#168350")
        ax.scatter(scene["initial"]["x"],scene["initial"]["y"],color="#2468ad",s=25,label="Start")
        ax.scatter(*scene["goal"][:2],marker="*",color="#ed9b0e",s=100,label="Goal")
        ax.set(xlim=(0,34),ylim=(0,24),xlabel="x (m)",ylabel="y (m)",aspect="equal")
        label = "physical witness: success" if result.get("outcome")=="success" else "unresolved in bounded search"
        ax.set_title(f"Fixed case {case}: {label}",fontsize=10)
        if case in traces:
            ax.legend(fontsize=7,loc="lower right")
    fig.suptitle("Independent post-test feasibility diagnostic: all 10 fixed unseen layouts\nPrivileged full-route spline + MPC; no PPO result or scenario exclusion",fontsize=13)
    fig.savefig(output / "all_conditions.png",dpi=160)
    plt.close(fig)
    print(json.dumps({"cases":len(data["results"]),"physical_successes":data["successful_physical_witnesses"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",required=True)
    parser.add_argument("--diagnostic",default="posthoc_spline_feasibility")
    args=parser.parse_args()
    main(args.batch,args.diagnostic)

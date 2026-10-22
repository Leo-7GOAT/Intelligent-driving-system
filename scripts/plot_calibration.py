import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder");folder=Path(p.parse_args().folder)
    rows=[json.loads(x) for x in (folder/"responses.jsonl").read_text(encoding="utf-8").splitlines()]
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for phase in sorted({r["phase"] for r in rows}):
        part=[r for r in rows if r["phase"]==phase];t=[r["t"] for r in part]
        for ax,key,label in zip(axes.flat,("v","a","yaw_degrees","steer"),("Speed (m/s)","Reported longitudinal acceleration (m/s2)","Yaw (degrees)","Normalized steering")):
            ax.plot(t,[r[key] for r in part],label=phase);ax.set(xlabel="Physical time (s)",ylabel=label)
    axes[0,0].legend(fontsize=8);fig.suptitle("CARLA 0.9.15 measured calibration (diagnostic, not PPO)")
    fig.tight_layout();fig.savefig(folder/"responses.png",dpi=160)

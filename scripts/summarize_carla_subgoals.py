"""Rebuild all-case CARLA low-level diagnostic comparison; no PPO attribution."""
import argparse,gzip,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from navstack.experiments import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);args=p.parse_args()
    batch=Path(args.batch);out=batch/"analysis";out.mkdir(exist_ok=True)
    folders=[batch/"carla_subgoal_diagnostic",batch/"carla_subgoal_terminal_diagnostic"]
    data=[]
    for folder in folders:
        protocol=json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
        rows=[json.loads(line) for line in (folder/"episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows)==len(protocol["cases"])
        traces=[]
        for r in rows:
            with gzip.open(folder/("trace_"+str(r["case_id"])+".jsonl.gz"),"rt",encoding="utf-8") as stream:
                steps=[json.loads(line) for line in stream]
            traces.append(steps)
        data.append((rows,traces))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,3,figsize=(13,6.7))
    comparisons=[]
    for i,ax in enumerate(axes.flat):
        row_a,steps_a=data[0][0][i],data[0][1][i]
        row_b,steps_b=data[1][0][i],data[1][1][i]
        assert (row_a["scenario"],row_a["condition"])==(row_b["scenario"],row_b["condition"])
        scene=json.loads((folders[0]/("scene_"+str(i)+".json")).read_text(encoding="utf-8"))
        other=json.loads((folders[1]/("scene_"+str(i)+".json")).read_text(encoding="utf-8"))
        original=np.r_[np.array(scene["static"]).ravel(),scene["goal"]]
        alternate=np.r_[np.array(other["static"]).ravel(),other["goal"]]
        error=float(np.max(np.abs(original-alternate)))
        assert error<.01,"Scene mismatch"
        for x0,y0,x1,y1 in scene["static"]:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color="gray",alpha=.4))
        for label,steps,row,color,style in (("v7",steps_a,row_a,"#777777","--"),("terminal envelope",steps_b,row_b,"#0072B2","-")):
            states=[t["state"] for step in steps for t in step["info"]["trace"]]
            ax.plot([s["x"] for s in states],[s["y"] for s in states],style,color=color,label=label+": "+row["termination_reason"])
            ax.plot(states[-1]["x"],states[-1]["y"],"o",color=color,ms=4)
        ax.plot(*scene["goal"][:2],"*",color="#009E73",ms=12,label="goal")
        ax.set(xlim=(0,34),ylim=(0,14),aspect="equal",title=row_a["scenario"]+" / "+str(row_a["condition"]),xlabel="x (m)",ylabel="y (m)")
        ax.legend(fontsize=6);ax.grid(alpha=.15)
        comparisons.append({"scenario":row_a["scenario"],"condition":row_a["condition"],"scene_geometry_max_difference_m":error,
                            "v7":{k:row_a[k] for k in ("success","termination_reason","progress","intervention_rate","physics_steps")},
                            "terminal_envelope":{k:row_b[k] for k in ("success","termination_reason","progress","intervention_rate","physics_steps")}})
    fig.suptitle("Actual CARLA: all six development cases, B3 rule-selected decisions\nLow-level engineering diagnostic; zero PPO updates; no statistical superiority claim",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/"carla_subgoal_comparison.png",dpi=160);plt.close(fig)
    write_json(out/"carla_subgoal_comparison.json",{"scope":"development diagnostic only, all cases retained","comparisons":comparisons})
    print(json.dumps({"cases":len(comparisons),"v7_success":sum(r["v7"]["success"] for r in comparisons),"new_success":sum(r["terminal_envelope"]["success"] for r in comparisons)}))


if __name__=="__main__":main()

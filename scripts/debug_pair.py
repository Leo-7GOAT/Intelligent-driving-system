"""Actual paired 20k diagnostic training using the common 45-second task."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time
from navstack.config import EnvConfig,PlannerConfig
from navstack.experiments import train,evaluate,write_json

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);args=p.parse_args()
    batch=Path(args.batch);deadline=json.loads((batch/"batch.json").read_text(encoding="utf-8"))["deadline_epoch"]
    results=[]
    for method in ("H","B1"):
        cfg=EnvConfig(method=method,deadline=45,planner=PlannerConfig(max_seconds=None))
        run,summary=train(cfg,"debug",11,batch/"debug",physics_budget=20000,
                          wall_seconds=min(1500,max(1,deadline-time.time()-120)))
        cfg.split="validation"
        validation,metrics=evaluate(cfg,run,episodes=12,root=batch/"debug",seed=101)
        results.append({"method":method,"run":str(run),"training":summary,
                        "validation":str(validation),"evaluation":metrics})
        write_json(batch/"debug_pair.json",results)

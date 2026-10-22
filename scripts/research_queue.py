import argparse,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.config import EnvConfig,PlannerConfig
from navstack.research import cumulative_train,atomic_json
from navstack.experiments import evaluate,append_json

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);p.add_argument("--queue",required=True);args=p.parse_args()
    batch=Path(args.batch);deadline=json.loads((batch/"batch.json").read_text(encoding="utf-8"))["deadline_epoch"]
    jobs=json.loads(Path(args.queue).read_text(encoding="utf-8"));results=[]
    for job in jobs:
        if time.time()>=deadline-90 or (batch/"STOP").exists():break
        name=job["name"];run=batch/job.get("version","development_v1")/name
        cfg=EnvConfig(method=job["method"],deadline=45,planner=PlannerConfig(max_seconds=None),
                      flat_hold=job.get("flat_hold",1),shared_shield=job.get("shared_shield",False),
                      terminal_goal_prefix=job.get("terminal_goal_prefix",False),curriculum=job.get("curriculum",False))
        cfg.reward.beta=job.get("reward_beta",1.)
        state={"status":"training","pid":os.getpid(),"parent_pid":os.getppid(),"run":str(run.resolve()),"job":job,"epoch":time.time(),"deadline_epoch":deadline}
        atomic_json(batch/f"queue_status_{os.getpid()}.json",state)
        summary=cumulative_train(run,cfg,job["seed"],job["budget"],deadline-60,ppo_options=job.get("ppo_options"))
        results.append(summary);append_json(batch/"queue_results.jsonl",{"job":job,"training":summary})
        if summary["reason"]!="milestone_reached":break
        evaluation_marker=run/f"validation_{job['budget']}.json"
        if not evaluation_marker.exists() and time.time()<deadline-180:
            state["status"]="evaluating";atomic_json(batch/f"queue_status_{os.getpid()}.json",state)
            cfg.split="validation"
            val,metrics=evaluate(cfg,summary["checkpoint"],episodes=12,root=run,seed=101)
            atomic_json(evaluation_marker,{"evaluation":str(val),"metrics":metrics,"actual_physics_steps":summary["physics_steps"]})
    atomic_json(batch/f"queue_status_{os.getpid()}.json",{"status":"queue_complete" if len(results)==len(jobs) else "budget_or_stop",
                "pid":os.getpid(),"parent_pid":os.getppid(),"epoch":time.time(),"completed_jobs":len(results),"total_jobs":len(jobs)})

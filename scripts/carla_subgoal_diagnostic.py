"""Actual CARLA closed-loop development validation of the versioned low level.

Rule-selected decisions are explicitly labelled B3, never PPO training/results.
"""
import argparse,gzip,hashlib,json,sys,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.carla_bridge import BridgeClient,CarlaNavigationEnv
from navstack.config import EnvConfig,VehicleConfig,PlannerConfig
from navstack.env import rule_action
from navstack.experiments import write_json,append_json,jsonable


def main():
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);p.add_argument("--terminal-envelope",action="store_true");args=p.parse_args()
    batch=Path(args.batch).resolve();output=batch/("carla_subgoal_terminal_diagnostic" if args.terminal_envelope else "carla_subgoal_diagnostic")
    output.mkdir(exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    hashes={}
    for source in [*(root/"navstack").glob("*.py"),root/"scripts/carla_bridge37.py",Path(__file__)]:
        target=output/"source"/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        hashes[source.relative_to(root).as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest()
    cal=json.loads((batch/"calibration/calibration.json").read_text(encoding="utf-8"))
    cfg=EnvConfig(method="B3",deadline=45,vehicle=VehicleConfig(**cal["vehicle"]),subgoal_mode="goal_frame_terminal" if args.terminal_envelope else "goal_frame",
                  planner=PlannerConfig(max_seconds=None,analytic_expansion=True,vectorized_primitives=args.terminal_envelope),terminal_goal_prefix=True)
    cases=[{"scenario":scene,"condition":condition} for condition in (7301,7302) for scene in ("static","curve","dynamic")]
    write_json(output/"protocol.json",{"config":cfg.to_dict(),"cases":cases,"source_hashes":hashes,
               "scope":"B3 rule/low-level development; zero PPO updates; genuine CARLA state feedback",
               "wall_budget_seconds":900,"retry_limit":0,"curve_scope":"curved obstacle route and parking on existing straight roadbed"})
    client=BridgeClient(batch);rows=[];started=time.time()
    try:
        for index,case in enumerate(cases):
            if time.time()-started>900:break
            env=CarlaNavigationEnv(cfg,client)
            env.reset(options={"case":case})
            write_json(output/("scene_"+str(index)+".json"),env.scene_evidence)
            cycles=[]
            with gzip.open(output/("trace_"+str(index)+".jsonl.gz"),"wt",encoding="utf-8") as stream:
                while not env.done:
                    if time.time()-started>900:raise TimeoutError("diagnostic wall budget")
                    before=time.perf_counter();action=rule_action(env)
                    _,_,_,_,info=env.step(action);cycles.append(time.perf_counter()-before)
                    stream.write(json.dumps({"action":action,"full_decision_seconds":cycles[-1],"info":info},default=jsonable)+"\n")
            row={"case_id":index,**case,**info["episode_metrics"],"full_decision_seconds":cycles}
            append_json(output/"episodes.jsonl",row);rows.append(row)
            print(json.dumps({k:row[k] for k in ("scenario","condition","success","termination_reason","progress","intervention_rate")}),flush=True)
        write_json(output/"summary.json",{"status":"complete" if len(rows)==len(cases) else "budget_censored",
                   "success":sum(r["success"] for r in rows),"completed":len(rows),"scheduled":len(cases),"ppo_updates":0,
                   "world_ticks_including_reset":client.total_ticks,"control_ticks":sum(r["physics_steps"] for r in rows),
                   "wall_seconds":time.time()-started,"policy":"B3_not_PPO"})
    except BaseException as exc:
        write_json(output/"failure.json",{"error":repr(exc),"completed":len(rows),"world_ticks_recorded":client.total_ticks});raise
    finally:client.close()


if __name__=="__main__":main()

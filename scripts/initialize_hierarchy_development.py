"""Freeze a paired H/H0 feedback development experiment, not a confirmation."""
import argparse,hashlib,json,shutil,time,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig,PlannerConfig


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--version",choices=("v7","v8"),default="v7");args=parser.parse_args()
    version=args.version
    batch=ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_hierarchy_"+version+"_development")
    batch.mkdir(); hashes={}
    for folder in ("navstack","scripts","configs","tests"):
        for source in (ROOT/folder).rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix in (".py",".ps1",".json"):
                rel=source.relative_to(ROOT);target=batch/"source"/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
                hashes[rel.as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest()
    (batch/"infrastructure").mkdir();(batch/"calibration").mkdir()
    shutil.copy2(ROOT/"runs/20261016T015226Z_research_batch_3377580f/calibration/calibration.json",batch/"calibration/calibration.json")
    jobs=[]
    for label,feedback in (("H","enabled"),("H0","ablated")):
        cfg=EnvConfig(method="H",deadline=45,obstacle_num=0,subgoal_mode="goal_frame_terminal" if version=="v8" else "goal_frame",execution_feedback=feedback,
                      planner=PlannerConfig(max_seconds=None,analytic_expansion=True,vectorized_primitives=version=="v8"),terminal_goal_prefix=True)
        cfg.reward.beta=40
        jobs.append({"id":label+"_seed11","seed":11,"env":cfg.to_dict(),"worker_script":"scripts/hierarchy_development.py","milestones":[0,25000,100000,200000]})
    protocol={"version":"hierarchy_"+version+"_feedback_development","created_epoch":time.time(),"deadline_epoch":time.time()+7200,
              "run_group":"hierarchy_"+version,"jobs":jobs,"source_hashes":hashes,
              "development_cases":[{"scenario":s,"seed":seed} for seed in (7301,7302) for s in ("empty","static","narrow","dynamic","unseen")],
              "purpose":"Isolate common candidate witness + previous execution feedback, same 450-input network, same low level and PPO parameters; H0 receives zeros",
              "not_confirmation":True,"ppo_core":"No action masking, teacher, rule action replacement, or extra supervision",
              "source_selection":"beta40 PBRS retained; both reward variants passed 1M short-route sanity, no evidence dense task-progress is necessary",
              "resource_limits":{"cpu_workers":2,"disk_min_GiB":12,"ram_min_GiB":4,"gpu_stop_temperature_C":85,"no_progress_seconds":300,"automatic_failure_retries":0}}
    (batch/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    (ROOT/"runs/active_development.json").write_text(json.dumps({"batch":str(batch)},indent=2),encoding="utf-8")
    print(batch)

"""PPO feedback ablation in a frozen development version; no final evaluation."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import load_run,jsonable
from navstack.process_lease import ProcessLease
from navstack.research import cumulative_train,atomic_json


def evaluate(run,checkpoint,config,cases,target):
    marker=run/("validation_"+str(target)+".json")
    if marker.exists(): return
    model,normalizer=load_run(checkpoint)
    original=normalizer.obs_rms.mean.copy()
    rows=[]
    with gzip.open(run/("validation_"+str(target)+"_trace.jsonl.gz"),"wt",encoding="utf-8") as stream:
        for case_id,case in enumerate(cases):
            config.scenario=case["scenario"];config.split="validation"
            env=NavigationEnv(config);obs,_=env.reset(seed=case["seed"],options=case.get("options"))
            cycles=[]
            last_heartbeat=0
            while True:
                started=time.perf_counter()
                action,_=model.predict(normalizer.normalize_obs(obs[None]),deterministic=True)
                chosen=action[0] if config.method=="B1" else int(action[0])
                obs,_,done,_,info=env.step(chosen)
                cycles.append(time.perf_counter()-started)
                stream.write(json.dumps({"case_id":case_id,"action":chosen,"full_cycle_seconds":cycles[-1],"info":info},default=jsonable)+"\n")
                if time.time()-last_heartbeat>5:
                    stream.flush();last_heartbeat=time.time()
                    atomic_json(run/"evaluation_heartbeat.json",{"pid":os.getpid(),"epoch":last_heartbeat,"case_id":case_id,"decisions":len(cycles),"target":target})
                if done:
                    rows.append({"case_id":case_id,**case,**info["episode_metrics"],"full_cycle_seconds":cycles})
                    break
            env.close()
    np.testing.assert_array_equal(original,normalizer.obs_rms.mean)
    normalizer.close()
    atomic_json(marker,{"scope":"development only","checkpoint":str(checkpoint),"model_sha256":hashlib.sha256((Path(checkpoint)/"model.zip").read_bytes()).hexdigest(),
                       "safe_completion":sum(r["success"] for r in rows)/len(rows),"episodes":rows})


def main():
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);p.add_argument("--job",required=True);args=p.parse_args()
    batch=Path(args.batch).resolve();protocol=json.loads((batch/"protocol.json").read_text(encoding="utf-8"))
    job=next(j for j in protocol["jobs"] if j["id"]==args.job)
    run=batch/protocol["run_group"]/job["id"];run.mkdir(parents=True,exist_ok=True)
    with ProcessLease(run/"worker.lock"):
        if (run/"completed.json").exists():return
        if (run/"failure.json").exists():raise RuntimeError("Inspect the previous failure before retry")
        config=EnvConfig.from_dict(job["env"])
        try:
            for target in job["milestones"]:
                if time.time()>=protocol["deadline_epoch"]-120 or (batch/"STOP").exists():return
                atomic_json(run/"worker_status.json",{"state":"training","pid":os.getpid(),"target":target,"epoch":time.time()})
                summary=cumulative_train(run,config,job["seed"],target,protocol["deadline_epoch"]-120,ppo_options=job.get("ppo_options",{"target_kl":.03}))
                if summary["physics_steps"]<target or summary["reason"]!="milestone_reached":return
                atomic_json(run/"worker_status.json",{"state":"evaluating","pid":os.getpid(),"target":target,"epoch":time.time()})
                evaluate(run,summary["checkpoint"],EnvConfig.from_dict(config.to_dict()),protocol["development_cases"],target)
            atomic_json(run/"completed.json",{"epoch":time.time(),"summary":summary})
        except BaseException as exc:
            atomic_json(run/"failure.json",{"epoch":time.time(),"error":repr(exc)});raise


if __name__=="__main__":main()

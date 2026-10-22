"""Bounded local process scheduling; independent training seeds, no shared CARLA world."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.research import atomic_json,resources_ok


def main(args):
    batch=Path(args.batch).resolve();jobs=json.loads(Path(args.queue).read_text(encoding="utf-8"))
    root=Path(__file__).resolve().parents[1];deadline=json.loads((batch/"batch.json").read_text(encoding="utf-8"))["deadline_epoch"]
    folder=batch/("scheduler_"+str(time.time_ns()));folder.mkdir()
    completed=[];failed=[];active=[]
    for milestone in sorted({j["budget"] for j in jobs}):
        pending=[j for j in jobs if j["budget"]==milestone]
        while pending or active:
            if time.time()>deadline-90 or (batch/"STOP").exists():
                pending=[]
            ok,reason=resources_ok(batch)
            if not ok:(batch/"STOP").write_text(reason,encoding="utf-8");pending=[]
            while pending and len(active)<args.workers:
                job=pending.pop(0);run=batch/job.get("version","development_v1")/job["name"]
                if (run/f"validation_{job['budget']}.json").exists():
                    completed.append({"job":job,"status":"already_complete"});continue
                file=folder/(job["version"]+"_"+job["name"]+"_"+str(job["budget"])+".json")
                atomic_json(file,[job])
                log=(folder/(file.stem+"_watchdog.log")).open("w",encoding="utf-8")
                command=[sys.executable,str(root/"scripts/batch_watchdog.py"),"--batch",str(batch),"--queue",str(file),"--source-root",str(Path(args.source_root).resolve())]
                process=subprocess.Popen(command,cwd=root,stdout=log,stderr=log,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),
                                         env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
                active.append({"process":process,"log":log,"job":job,"run":run})
            remaining=[]
            for child in active:
                if child["process"].poll() is None:remaining.append(child);continue
                child["log"].close();job=child["job"]
                if (child["run"]/f"validation_{job['budget']}.json").exists():completed.append({"job":job,"status":"completed"})
                else:failed.append({"job":job,"status":"incomplete","watchdog_exit_code":child["process"].returncode})
            active=remaining
            atomic_json(folder/"status.json",{"pid":os.getpid(),"epoch":time.time(),"milestone":milestone,
                        "active":[{"pid":c["process"].pid,"job":c["job"]} for c in active],"pending":pending,
                        "completed":completed,"failed":failed,"deadline_epoch":deadline})
            if active:time.sleep(2)
        # Do not advance an architecture to a larger budget while its paired stage failed.
        if failed or time.time()>deadline-90 or (batch/"STOP").exists():break
    atomic_json(folder/"summary.json",{"completed":completed,"failed":failed,"finished_epoch":time.time(),"pid":os.getpid()})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);p.add_argument("--queue",required=True)
    p.add_argument("--source-root",required=True);p.add_argument("--workers",type=int,choices=(1,2,3,4,5,6),default=2)
    main(p.parse_args())

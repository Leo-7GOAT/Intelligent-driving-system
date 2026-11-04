"""Resume a registered matrix after a running scheduler drains, without duplicate jobs."""
import argparse,json,subprocess,sys,time
from pathlib import Path

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);p.add_argument("--wait-folder",required=True)
    p.add_argument("--queue",required=True);p.add_argument("--source-root",required=True);args=p.parse_args()
    batch=Path(args.batch).resolve();folder=Path(args.wait_folder).resolve()
    deadline=json.loads((batch/"batch.json").read_text(encoding="utf-8"))["deadline_epoch"]
    while not (folder/"summary.json").exists():
        if time.time()>deadline-120 or (batch/"STOP").exists():sys.exit(1)
        time.sleep(5)
    command=[sys.executable,str(Path(__file__).with_name("parallel_research.py")),"--batch",str(batch),"--queue",args.queue,
             "--source-root",args.source_root,"--workers","4"]
    print(json.dumps({"epoch":time.time(),"action":"resume_matrix_after_drain","command":command}),flush=True)
    result=subprocess.run(command,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    sys.exit(result.returncode)

"""Bounded guard for an explicitly identified, already running research worker."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.owned_process import OwnedProcess
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json,resources_ok


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();batch=Path(a.batch).resolve()
    identity=json.loads(Path(a.identity).read_text(encoding='utf-8-sig'))
    protocol=json.loads((batch/'protocol.json').read_text(encoding='utf-8'))
    allowed=('paired_update_stability.py','collect_executed_demonstrations.py','train_demonstration_warmstart.py')
    if str(batch) not in identity['command'] or not any(name in identity['command'] for name in allowed):
        raise ValueError('Adopted command does not identify an allowed registered worker')
    with ProcessLease(batch/'guard.lock'):
        child=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=True)
        reason=None
        while child.poll() is None:
            now=time.time();ok,reason=resources_ok(batch)
            if shutil.disk_usage(batch).free<12*1024**3:ok,reason=False,'disk_below_12GiB'
            if now>=protocol['deadline_epoch']:ok,reason=False,'wall_deadline'
            scopes=[batch]+[batch/b['name'] for b in protocol.get('branches',[])]+[batch/name for name in protocol.get('worker_scopes',[])]
            recent=max([protocol['created_epoch']]+[f.stat().st_mtime for scope in scopes
                for f in scope.glob('*heartbeat.json')]+[f.stat().st_mtime for name in ('progress.json','stdout.log')
                for f in [batch/name] if f.exists()])
            if now-recent>300:ok,reason=False,'no_progress_300s'
            gpu=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if gpu.returncode==0 and max(map(int,gpu.stdout.split()))>=85:ok,reason=False,'gpu_temperature_85C'
            atomic_json(batch/'guard.json',{'pid':os.getpid(),'worker_pid':child.pid,'epoch':now,'last_progress':recent,
                'deadline_epoch':protocol['deadline_epoch'],'healthy':ok,'reason':reason})
            if not ok:
                # Last immutable checkpoint and partial logs stay intact. No
                # automatic retries or termination of any unrelated process.
                child.terminate();child.wait();break
            time.sleep(5)
        atomic_json(batch/'guard_final.json',{'epoch':time.time(),'worker_pid':child.pid,'exit_code':child.poll(),'reason':reason})


if __name__=='__main__':main()

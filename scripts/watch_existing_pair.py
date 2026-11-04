"""Bounded guard for an explicitly identified, already running research worker."""
import argparse
import json
import os
import shlex
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.owned_process import OwnedProcess
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json,resources_ok

ALLOWED_WORKERS = ('paired_update_stability.py','collect_executed_demonstrations.py','train_demonstration_warmstart.py',
    'train_recovery_demonstrations.py','fit_recovery_actor.py','relabel_recovery_teacher.py',
    'train_current_state_aggregation.py','train_aggregated_ppo.py','audit_normalization_transfer.py','native_road_learning.py',
    'evaluate_native_rule.py','evaluate_native_policy_weights.py','native_curve_control.py')


def query_gpu_temperature(runner=None):
    """At most two ten-second probes;unavailable telemetry still fails closed."""
    runner=subprocess.run if runner is None else runner
    attempts=[]
    for _ in range(2):
        try:
            result=runner(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:
                attempts.append({'status':'nonzero_exit','returncode':result.returncode});continue
            values=[int(value) for value in result.stdout.split()]
            if not values or any(not 0<=value<=125 for value in values):
                attempts.append({'status':'invalid_temperature'});continue
            attempts.append({'status':'ok'})
            return {'status':'ok','temperature_c':max(values),'attempts':attempts}
        except subprocess.TimeoutExpired:
            attempts.append({'status':'timeout','timeout_seconds':10})
        except (OSError,ValueError):
            attempts.append({'status':'unavailable_or_malformed'})
    return {'status':'unavailable','temperature_c':None,'attempts':attempts}


def command_identifies_batch(command, batch):
    """Resolve an explicit --batch token; never rely on a name substring."""
    try:
        arguments = [part.strip('"') for part in shlex.split(command,posix=False)]
        if arguments.count('--batch') != 1:
            return False
        value = arguments[arguments.index('--batch')+1]
        return Path(value).resolve() == Path(batch).resolve()
    except (ValueError,IndexError,OSError):
        return False


def command_identifies_worker(command, batch):
    if not command_identifies_batch(command, batch):
        return False
    try:
        arguments = [part.strip('"') for part in shlex.split(command, posix=False)][1:]
        while arguments and arguments[0] in ('-u', '-B'):
            arguments.pop(0)
        # Inspect Python's actual script operand, not an allowed name embedded
        # in another argument, diagnostic text, or a python -c expression.
        return bool(arguments) and Path(arguments[0]).name in ALLOWED_WORKERS
    except (ValueError, OSError):
        return False


def cleanup_owned_server(batch, powershell):
    """Record failed cleanup as evidence; never escalate to an unverified kill."""
    script=Path(__file__).resolve().parent/'stop_owned_carla.ps1'
    receipt={'epoch':time.time(), 'returncode':None,
        'scope':'Stop only recorded owned server after matching-version empty-world probe; failure preserves server for inspection'}
    try:
        result=subprocess.run([powershell,'-NoProfile','-File',str(script),'-Batch',str(batch),'-BridgeDrainSeconds','30'],
            capture_output=True,text=True,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        receipt.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except (subprocess.TimeoutExpired,OSError) as exc:
        receipt['error']=repr(exc)
        # TimeoutExpired may retain bytes even when text=True was requested.
        for name in ('stdout','stderr'):
            value=getattr(exc,name,None)
            receipt[name]=value.decode('utf-8',errors='replace') if isinstance(value,bytes) else value
    atomic_json(Path(batch)/'server_cleanup.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--identity',required=True)
    p.add_argument('--powershell',help='Required only for registered owned-CARLA cleanup')
    a=p.parse_args();batch=Path(a.batch).resolve()
    identity=json.loads(Path(a.identity).read_text(encoding='utf-8-sig'))
    protocol=json.loads((batch/'protocol.json').read_text(encoding='utf-8'))
    if not command_identifies_worker(identity['command'],batch):
        raise ValueError('Adopted command does not identify an allowed registered worker')
    if protocol.get('cleanup_owned_carla') and (not a.powershell or not Path(a.powershell).is_file()):
        raise ValueError('Registered native run requires a verified PowerShell executable for owned-empty-server cleanup')
    with ProcessLease(batch/'guard.lock'):
        child=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=True)
        reason=None
        try:
            while child.poll() is None:
                now=time.time();ok,reason=resources_ok(batch)
                if shutil.disk_usage(batch).free<12*1024**3:ok,reason=False,'disk_below_12GiB'
                if now>=protocol['deadline_epoch']:ok,reason=False,'wall_deadline'
                scopes=[batch]+[batch/b['name'] for b in protocol.get('branches',[])]+[batch/name for name in protocol.get('worker_scopes',[])]+[batch/name for name in protocol.get('methods',[])]
                recent=max([protocol['created_epoch']]+[f.stat().st_mtime for scope in scopes
                    for f in scope.glob('*heartbeat.json')]+[f.stat().st_mtime for name in ('progress.json','stdout.log')
                    for f in [batch/name] if f.exists()])
                if now-recent>300:ok,reason=False,'no_progress_300s'
                gpu=query_gpu_temperature()
                if gpu['status']!='ok':ok,reason=False,'gpu_telemetry_unavailable'
                elif gpu['temperature_c']>=85:ok,reason=False,'gpu_temperature_85C'
                atomic_json(batch/'guard.json',{'pid':os.getpid(),'worker_pid':child.pid,'epoch':now,'last_progress':recent,
                    'deadline_epoch':protocol['deadline_epoch'],'healthy':ok,'reason':reason,'gpu_telemetry':gpu})
                if not ok:
                    # Retained process handle prevents PID reuse from targeting
                    # another task. Bridge EOF cleanup precedes the empty probe.
                    child.terminate();child.wait();break
                time.sleep(5)
        except BaseException as exc:
            reason='guard_exception: '+repr(exc)
            if child.poll() is None:child.terminate();child.wait()
            raise
        finally:
            atomic_json(batch/'guard_final.json',{'epoch':time.time(),'worker_pid':child.pid,'exit_code':child.poll(),'reason':reason})
            if protocol.get('cleanup_owned_carla'):
                cleanup_owned_server(batch,a.powershell)


if __name__=='__main__':main()

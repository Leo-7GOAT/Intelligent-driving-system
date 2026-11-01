"""Run registered illustrative PPO recordings after the owned training queue exits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def verify_training_dependencies(plan,root):
    receipts=plan.get('completed_training_barrier');root=Path(root).resolve()
    if receipts is None:
        if not plan.get('dependency_processes'):
            raise ValueError('Registered live dependencies or completed training receipts required')
        return
    expected={str(Path('runs')/plan[name]/filename).replace('\\','/') for name,filename in
        (('lower_execution','status.json'),('full_execution','status.json'),('full_execution','failure.json'))}
    if not isinstance(receipts,dict) or set(receipts)!=expected or plan.get('dependency_processes'):
        raise ValueError('Explicit completed training receipts cannot mix live dependencies')
    for relative,digest in receipts.items():
        path=root/relative
        if (not path.resolve().is_relative_to(root/'runs')
                or hashlib.sha256(path.read_bytes()).hexdigest()!=digest):
            raise ValueError('Completed training barrier receipt changed')


def require_training_barrier(lower,full,lower_failure,full_failure):
    rows=lower.get('completed',[])
    phases=('ALL_FIVE_BASIC_PPO_COURSES_QUALIFIED_FOR_FULL_TASK_TRAINING',
            'ALL_FIVE_BASIC_PPO_COURSES_COMPLETE_WITH_FAILED_GATES')
    if (lower_failure is not None or lower.get('phase') not in phases
            or [r['seed'] for r in rows]!=[11,22,33,44,55]
            or any(type(r.get('basic_qualified_after_learning')) is not bool for r in rows)):
        raise ValueError('All five native learning extensions must finish without infrastructure failure')
    qualified=all(r['basic_qualified_after_learning'] for r in rows)
    if qualified:
        if (lower['phase']!=phases[0] or full_failure is not None
                or full.get('phase')!='ALL_FIVE_FULL_TASK_PPO_AFTER_BASIC_COURSE_COMPLETE'
                or [r['seed'] for r in full.get('completed',[])]!=[11,22,33,44,55]):
            raise ValueError('Qualified courses require completion of every registered full-task child')
    else:
        expected="ValueError('All five actually trained basic courses must qualify;no seed omission or gate waiver')"
        if (lower['phase']!=phases[1] or full.get('created')!=[] or full.get('completed')!=[]
                or full_failure is None or full_failure.get('error')!=expected
                or full_failure.get('created')!=[] or full_failure.get('completed')!=[]):
            raise ValueError('Only the explicit childless qualification failure may precede recordings')
    return {'all_five_learning_extensions_complete':True,'basic_qualified_all_five':qualified,
            'full_task_skipped_for_unchanged_failed_gate':not qualified}


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;source=execution/'source'
    sys.path.insert(0,str(source))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    plan=read(execution/'plan.json');hashes=read(execution/'source_manifest.json')
    owners=[];completed=[]
    def status(phase,**extra):
        atomic_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time(),'completed':completed,**extra})
    def verify():
        for name,digest in hashes.items():
            if sha(source/name)!=digest:raise ValueError('Frozen camera observer changed:'+name)
        for row in plan['recordings']:
            batch=root/'runs'/row['batch']
            original=batch/'queued_protocol.json' if (batch/'queued_protocol.json').is_file() else batch/'protocol.json'
            if sha(original)!=row['protocol_sha256']:
                raise ValueError('Registered pending camera protocol changed')
    def call(command,label,env=None):
        with (execution/(label+'.log')).open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json(execution/'last_child.json',{'pid':child.pid,'epoch':time.time(),'label':label})
            code=child.wait(timeout=2400)
        if code:raise RuntimeError('Camera stage failed;no automatic retry:'+label)
    try:
        verify()
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'epoch':time.time()},f)
            verify_training_dependencies(plan,root)
            for identity in plan.get('dependency_processes',[]):
                owners.append(OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False))
            deadline=time.monotonic()+12*3600
            while any(owner.poll() is None for owner in owners):
                if time.monotonic()>deadline:raise TimeoutError('Finite training wait expired;no camera launch')
                status('WAITING_FOR_OWNED_TRAINING_QUEUES_TO_EXIT')
                time.sleep(10)
            paths=[root/'runs'/plan[name] for name in ('lower_execution','full_execution')]
            failures=[read(p/'failure.json') if (p/'failure.json').is_file() else None for p in paths]
            states=[read(p/'status.json') for p in paths]
            barrier=require_training_barrier(*states,*failures)
            for state in states:
                for row in state.get('completed',[]):require_finished_parent(root/'runs'/row['batch'])
            atomic_json(execution/'training_barrier.json',barrier)
            for row in plan['recordings']:
                verify();okay,reason=resources_ok(root)
                if not okay:raise RuntimeError(str(reason))
                batch=root/'runs'/row['batch'];method=row['method'];label=method+'_camera'
                status('RECORDING_REAL_NATIVE_PPO',batch=batch.name,method=method)
                call([shutil.which('pwsh'),'-NoProfile','-File',source/'scripts/run_registered_native.ps1',
                    '-Batch',batch,'-Python',root/'.venv/Scripts/python.exe','-CarlaPython',root/'.carla-python37/runtime/python.exe',
                    '-ActivateQueued','-UseCurrentGuard'],label+'_lifecycle')
                require_finished_parent(batch)
                env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'))
                call([sys.executable,batch/'source/scripts/audit_native_method_endpoint.py','--batch',batch,'--method',method],label+'_raw',env)
                call([sys.executable,batch/'source/scripts/audit_native_policy_inference.py','--batch',batch,'--method',method,
                    '--output',batch/method/'measured_start_inference.json'],label+'_inference',env)
                call([sys.executable,batch/'source/scripts/render_carla_recordings.py','--run',batch/method,
                    '--output',batch/'videos'],label+'_encode')
                call([sys.executable,source/'scripts/verify_carla_videos.py','--videos',batch/'videos'],label+'_decode')
                summary=read(batch/method/'summary.json');audit=read(batch/method/'endpoint_raw_audit.json')
                if summary['physics_steps']!=0 or summary['optimizer_steps']!=0:
                    raise ValueError('Frozen recording unexpectedly changed the learner')
                completed.append({'batch':batch.name,'method':method,'raw_controls_verified':audit['evaluation_control_frames_verified'],
                    'video_manifest_sha256':sha(batch/'videos/manifest.json'),
                    'decode_verification_sha256':sha(batch/'videos/decode_verification.json')})
            status('ALL_REGISTERED_NATIVE_PPO_RECORDINGS_AND_FRAME_VERIFICATIONS_COMPLETE')
    except BaseException as error:
        with (execution/'failure.json').open('x',encoding='utf-8') as f:json.dump({'error':repr(error),'epoch':time.time(),'completed':completed},f,indent=2)
        raise
    finally:
        for owner in owners:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True)
    args=parser.parse_args();run(args.execution)

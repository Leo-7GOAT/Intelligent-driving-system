"""Conditional all-five full-task PPO continuation after every basic course qualifies."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;source=execution/'source'
    sys.path.insert(0,str(source))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts import prepare_native_full_task_after_course as preparation
    preparation.ROOT=root
    plan=read(execution/'plan.json');hashes=read(execution/'source_manifest.json');owner=None;completed=[];created=[]
    def verify():
        for name,digest in hashes.items():
            if sha(source/name)!=digest:raise ValueError('Frozen continuation code changed:'+name)
        for name,digest in plan['workspace_bindings'].items():
            if sha(root/name)!=digest:raise ValueError('Bound operational source changed:'+name)
    def status(phase,**extra):atomic_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time(),'created':created,'completed':completed,**extra})
    def call(command,label,timeout,env=None):
        with (execution/(label+'.log')).open('x',encoding='utf-8') as log:
            child=subprocess.Popen(list(map(str,command)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(execution/'last_child.json',{'pid':child.pid,'label':label,'epoch':time.time()})
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:raise RuntimeError('Observer expired;native watchdog retains ownership,no retry')
        if code:raise RuntimeError('Registered full-task child failed:'+label)
    try:
        verify()
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'epoch':time.time()},f)
            dependency=root/'runs'/plan['dependency_execution'];identity=plan['dependency_process']
            owner=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
            until=time.monotonic()+43200
            while True:
                if (dependency/'failure.json').exists():raise RuntimeError('Basic-course infrastructure failed;no full-task starts')
                state=read(dependency/'status.json')
                if state['phase'].startswith('ALL_FIVE_BASIC_PPO_COURSES_'):
                    preparation.require_all_courses(state);break
                if owner.poll() is not None:raise RuntimeError('Basic-course driver exited without all-five qualification')
                if time.monotonic()>until:raise TimeoutError('Finite12hour basic-course wait expired')
                status('WAITING_FOR_ALL_FIVE_TRAINED_BASIC_QUALIFICATIONS',dependency_pid=owner.pid);time.sleep(15)
            if owner.poll() is None:owner.wait(timeout=30)
            preparation.require_all_courses(read(dependency/'status.json'))
            for row in state['completed']:require_finished_parent(root/'runs'/row['batch'])
            atomic_json(execution/'qualified_basic_courses.json',{'state_sha256':sha(dependency/'status.json'),'all_five':state['completed']})
            for row in state['completed']:
                verify();healthy,reason=resources_ok(root)
                if not healthy:raise RuntimeError(str(reason))
                parent=root/'runs'/row['batch'];batch=preparation.prepare(parent);seed=row['seed'];prefix='seed'+str(seed)
                created.append({'seed':seed,'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')})
                status('RUNNING_FULL_TASK_PPO_AFTER_BASIC_COURSE',seed=seed,batch=batch.name)
                call([shutil.which('pwsh'),'-NoProfile','-File',source/'scripts/run_registered_native.ps1',
                    '-Batch',batch,'-Python',root/'.venv/Scripts/python.exe','-CarlaPython',root/'.carla-python37/runtime/python.exe',
                    '-ActivateQueued','-UseCurrentGuard'],prefix+'_lifecycle',7800)
                require_finished_parent(batch);env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'))
                call([sys.executable,source/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method','B1'],prefix+'_raw',300,env)
                call([sys.executable,source/'scripts/audit_native_policy_inference.py','--batch',batch,'--method','B1',
                    '--output',batch/'B1/measured_start_inference.json'],prefix+'_inference',300,env)
                folder=batch/'B1';summary=read(folder/'summary.json');raw=read(folder/'endpoint_raw_audit.json');episodes=read(folder/'evaluation.json')['episodes']
                if (len(episodes)!=12 or sorted(r['case_id'] for r in episodes)!=list(range(12))
                        or summary['physics_steps']!=20480 or summary['lineage_control_steps']!=30720
                        or summary['optimizer_steps']<=0 or len(read(folder/'measured_start_inference.json')['results'])!=12):
                    raise ValueError('Incomplete full-task learning or original12-condition inventory')
                result={'scope':'Actual full-task DEVELOPMENT PPO after disclosed basic curriculum;not equal-budget or final-confirmation evidence',
                    'seed':seed,'batch':batch.name,'parent_basic_course':parent.name,'cases':12,'successes':sum(r['success'] for r in episodes),
                    'termination_counts':dict(Counter(r['termination_reason'] for r in episodes)),
                    'new_full_task_training_controls':summary['physics_steps'],'total_policy_lineage_controls':summary['lineage_control_steps'],
                    'new_full_task_Adam_steps':summary['optimizer_steps'],'raw_controls_verified':raw['total_applied_controls_verified'],
                    'model_sha256':sha(folder/'model.zip'),'normalization_sha256':sha(folder/'normalization.pkl'),'research_acceptance':'UNMET'}
                with (root/'reports'/('native_B1_full_after_basic_seed'+str(seed)+'_endpoint_20261101.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
                completed.append({'seed':seed,'batch':batch.name});status('FULL_TASK_SEED_COMPLETE',seed=seed)
            status('ALL_FIVE_FULL_TASK_PPO_AFTER_BASIC_COURSE_COMPLETE')
    except BaseException as error:
        with (execution/'failure.json').open('x',encoding='utf-8') as f:json.dump({'epoch':time.time(),'error':repr(error),'created':created,'completed':completed},f,indent=2)
        raise
    finally:
        if owner is not None:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',required=True);a=p.parse_args();run(a.execution)

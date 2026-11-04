"""Train every registered basic-motion PPO course after the old all-five gate audit exits."""
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


def require_failed_predecessor(state,failure_present):
    gates=state['gates']
    if (not failure_present or state.get('created')!=[] or [r['seed'] for r in gates]!=[11,22,33,44,55]
            or any(type(r.get('passed')) is not bool for r in gates) or all(r['passed'] for r in gates)):
        raise ValueError('Require the terminated all-five failed initialization audit;no skipped gates or started PPO')


def basic_qualification(episodes):
    if len(episodes)!=4 or sorted(r['case_id'] for r in episodes)!=list(range(4)):
        raise ValueError('Every registered basic road must be evaluated exactly once')
    unsafe=sum(r['termination_reason'] in ('collision','out_of_bounds') for r in episodes)
    return sum(r['success'] for r in episodes)>=3 and unsafe==0


def require_old_processes_absent(identities):
    """Read-only recovery check:PID reuse never grants process-management authority."""
    pids=','.join(str(int(r['pid'])) for r in identities)
    command=f"Get-CimInstance Win32_Process | Where-Object {{$_.ProcessId -in @({pids})}} | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
    result=subprocess.run([shutil.which('pwsh'),'-NoProfile','-Command',command],check=True,capture_output=True,text=True,timeout=30)
    actual=json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(actual,dict):actual=[actual]
    for old in identities:
        if any(r['ProcessId']==old['pid'] and r['Name'].lower()==old['name'].lower() for r in actual):
            raise RuntimeError('Old process identity remains live or ambiguous;no restart')
    return actual


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;source=execution/'source'
    plan=read(execution/'plan.json');hashes=read(execution/'source_manifest.json')
    sys.path.insert(0,str(source))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.audit_native_ppo_configuration import verify as verify_learner
    prepared=read(root/'runs'/plan['preparation']/'status.json')
    if (prepared['phase']!='ALL_FIVE_BASIC_PPO_COURSES_PREPARED_NOT_DRIVEN'
            or [r['seed'] for r in prepared['batches']]!=[11,22,33,44,55]):raise ValueError('All five registered courses required')
    owner=None;completed=[]
    def verify():
        for name,digest in hashes.items():
            if sha(source/name)!=digest:raise ValueError('Frozen course source changed:'+name)
        for name,digest in plan['workspace_bindings'].items():
            if sha(root/name)!=digest:raise ValueError('Bound live operational source changed:'+name)
    def status(phase,**extra):atomic_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time(),'completed':completed,**extra})
    def call(command,label,timeout,env=None):
        with (execution/(label+'.log')).open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json(execution/'last_child.json',{'pid':child.pid,'epoch':time.time(),'label':label})
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:raise RuntimeError('Observer expired;native watchdog retains ownership,no retry')
        if code:raise RuntimeError('Registered course child failed;no automatic retry:'+label)
    try:
        verify()
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'epoch':time.time()},f)
            if plan.get('external_interruption_restart'):
                inventory=require_old_processes_absent(plan['interrupted_processes'])
                atomic_json(execution/'old_process_liveness.json',{'epoch':time.time(),'actual_reused_PID_inventory':inventory,'no_process_killed':True})
            else:
                identity=plan['dependency_process'];owner=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
            until=time.monotonic()+7200
            while owner is not None and owner.poll() is None:
                if time.monotonic()>until:raise TimeoutError('Finite old-gate wait expired;no native course started')
                status('WAITING_FOR_ALL_FIVE_OLD_GATES_AND_OWNED_DRIVER_EXIT',dependency_pid=owner.pid)
                time.sleep(10)
            predecessor=root/'runs'/plan['dependency_execution']
            require_failed_predecessor(read(predecessor/'status.json'),(predecessor/'failure.json').is_file())
            old_preparation=read(root/'runs'/plan['old_preparation']/'status.json')
            for row in old_preparation['batches']:require_finished_parent(root/'runs'/row['batch'])
            atomic_json(execution/'predecessor_failure_bound.json',{'sha256':sha(predecessor/'failure.json'),'all_five_old_gates_retained':True})
            reference=root/'runs'/prepared['batches'][0]['batch']/'resume/B1'
            for row in prepared['batches']:
                verify();healthy,reason=resources_ok(root)
                if not healthy:raise RuntimeError(str(reason))
                batch=root/'runs'/row['batch'];seed=row['seed'];prefix='seed'+str(seed)
                if sha(batch/'protocol.json')!=row['protocol_sha256']:raise ValueError('Course protocol changed')
                verify_learner(batch/'resume/B1',reference)
                status('TRAINING_BASIC_MOTION_WITH_ACTUAL_PPO',seed=seed,batch=batch.name)
                call([shutil.which('pwsh'),'-NoProfile','-File',source/'scripts/run_registered_native.ps1',
                    '-Batch',batch,'-Python',root/'.venv/Scripts/python.exe','-CarlaPython',root/'.carla-python37/runtime/python.exe',
                    '-ActivateQueued','-UseCurrentGuard'],prefix+'_lifecycle',7800)
                require_finished_parent(batch)
                env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'))
                call([sys.executable,source/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method','B1'],prefix+'_raw',300,env)
                call([sys.executable,source/'scripts/audit_native_policy_inference.py','--batch',batch,'--method','B1',
                      '--output',batch/'B1/measured_start_inference.json'],prefix+'_inference',300,env)
                folder=batch/'B1';episodes=read(folder/'evaluation.json')['episodes'];summary=read(folder/'summary.json')
                raw=read(folder/'endpoint_raw_audit.json');inference=read(folder/'measured_start_inference.json')
                if (summary['physics_steps']<10240 or summary['optimizer_steps']<=0 or len(inference['results'])!=4
                        or raw['training_control_frames_verified']!=summary['physics_steps']):raise ValueError('Incomplete basic PPOlearning/audit')
                qualified=basic_qualification(episodes)
                result={'scope':'Actual native PPO basic-motion DEVELOPMENT course;not a qualified full-task B1baseline',
                    'seed':seed,'batch':batch.name,'cases':4,'successes':sum(r['success'] for r in episodes),
                    'termination_counts':dict(Counter(r['termination_reason'] for r in episodes)),
                    'basic_qualified_after_learning':qualified,'new_training_controls':summary['physics_steps'],
                    'new_Adam_steps':summary['optimizer_steps'],'raw_controls_verified':raw['total_applied_controls_verified'],
                    'model_sha256':sha(folder/'model.zip'),'normalization_sha256':sha(folder/'normalization.pkl'),
                    'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'research_acceptance':'UNMET'}
                with (root/'reports'/('native_B1_basic_PPO_seed'+str(seed)+'_endpoint_20261031.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
                completed.append({'seed':seed,'batch':batch.name,'basic_qualified_after_learning':qualified})
                status('BASIC_PPO_COURSE_COMPLETE',seed=seed)
            status('ALL_FIVE_BASIC_PPO_COURSES_QUALIFIED_FOR_FULL_TASK_TRAINING' if all(r['basic_qualified_after_learning'] for r in completed)
                   else 'ALL_FIVE_BASIC_PPO_COURSES_COMPLETE_WITH_FAILED_GATES')
    except BaseException as error:
        with (execution/'failure.json').open('x',encoding='utf-8') as f:json.dump({'error':repr(error),'epoch':time.time(),'completed':completed},f,indent=2)
        raise
    finally:
        if owner is not None:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',required=True);a=p.parse_args();run(a.execution)

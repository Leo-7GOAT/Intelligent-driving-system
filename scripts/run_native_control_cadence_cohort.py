"""Twenty native control-cadence DEVELOPMENT probes after shared-brake closure."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cadence_trace_counts(path,method,hold):
    if method not in ('H','B1') or type(hold) is not int or hold not in (2,5):
        raise ValueError('Registered H/B1cadence required')
    result={'physics_controls':0,'policy_decisions':0,'MPCrecomputations':0,
            'held_MPCframes':0,'MPCfailed_recomputations':0,'held_failed_MPCframes':0,
            'native_nonreference_frames':0}
    timings=[]
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            info=json.loads(line)['info'];ticks=info['trace']
            if not ticks or len(ticks)!=info['executed_substeps']:
                raise ValueError('Complete measured native decision trace required')
            if method=='B1' and (len(ticks)>hold or info['mode']!='flat'):
                raise ValueError('Direct policy must issue its own held action without a planner')
            result['policy_decisions']+=1
            for tick in ticks:
                result['physics_controls']+=1
                if method=='H' and info['mode']=='reference':
                    d=tick['diagnostics'];recomputed=d['controller_recomputed']
                    if type(recomputed) is not bool or d['controller_hold_steps']!=hold:
                        raise ValueError('Recorded MPCcadence differs from registered profile')
                    if recomputed:
                        result['MPCrecomputations']+=1;timings.append(d['control_seconds'])
                        result['MPCfailed_recomputations']+=tick['controller_status']=='failed'
                    else:
                        result['held_MPCframes']+=1
                        result['held_failed_MPCframes']+=tick['controller_status']=='failed'
                        if d['control_seconds']!=0 or d['qp_seconds']!=0:
                            raise ValueError('Held control cannot claim a fresh MPCsolve time')
                elif method=='H':result['native_nonreference_frames']+=1
    result['MPCrecomputation_seconds']=timings
    return result


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;frozen=execution/'source'
    if execution.parent.name!='runs' or not (root/'.git').exists():raise ValueError('Project execution required')
    plan=read(execution/'plan.json');manifest=read(execution/'source_manifest.json')
    if not 60<=plan['wait_seconds']<=86400 or [r['seed'] for r in plan['H_models']]!=[11,22,33,44,55]:
        raise ValueError('Finite all-five paired cadence plan required')
    def verify():
        for name,digest in manifest.items():
            path=(frozen/name).resolve()
            if not path.is_relative_to(frozen) or sha(path)!=digest:raise ValueError('Frozen cadence source changed:'+name)
        for name,digest in plan['workspace_bindings'].items():
            if sha(root/name)!=digest:raise ValueError('Cadence preparation binding changed:'+name)
    verify();sys.path.insert(0,str(frozen))
    from navstack.metrics import timing_summary
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json,resources_ok
    from scripts import prepare_native_control_cadence as preparation
    from scripts.run_native_shared_brake_cohort import b1_parent_map
    from scripts.evaluate_native_rule import require_finished_parent
    preparation.ROOT=root
    completed=[];owner=None
    def status(phase,**extra):atomic_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time(),'completed':completed,**extra})
    def call(command,log,timeout,environment=None):
        with (execution/log).open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json(execution/'last_child.json',{'pid':child.pid,'epoch':time.time(),'log':log})
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:raise RuntimeError('Observer expired;native watchdog retains ownership PID'+str(child.pid))
        if code:raise RuntimeError('Registered cadence child failed;no retry:'+log)
    try:
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x') as stream:json.dump({'pid':os.getpid(),'epoch':time.time(),'script_sha256':sha(Path(__file__)),'plan_sha256':sha(execution/'plan.json')},stream)
            dependency=root/'runs'/plan['dependency_execution'];expected='ALL_TWENTY_PAIRED_SHARED_BRAKE_PROBES_COMPLETE';until=time.monotonic()+plan['wait_seconds']
            while read(dependency/'status.json')['phase']!=expected:
                if (dependency/'failure.json').exists() or time.monotonic()>until:raise RuntimeError('Shared-brake predecessor failed or bounded wait expired')
                if owner is None:
                    i=plan['dependency_process'];owner=OwnedProcess(i['pid'],i['creation_epoch'],i['executable'],allow_terminate=False)
                if owner.poll() is not None:
                    if read(dependency/'status.json')['phase']==expected:break
                    raise RuntimeError('Predecessor exited before its completion marker')
                status('WAITING_FOR_SHARED_BRAKE_COHORT',dependency_pid=owner.pid,dependency_live_via_retained_handle=True);time.sleep(10)
            if (dependency/'failure.json').exists():raise RuntimeError('Retained shared-brake failure')
            prior=read(dependency/'status.json')['completed']
            expected_keys={(seed,method,enabled) for seed in (11,22,33,44,55) for method in ('H','B1') for enabled in (False,True)}
            if len(prior)!=20 or {(r['seed'],r['method'],r['enabled']) for r in prior}!=expected_keys:
                raise ValueError('Every shared-brake predecessor arm is required')
            for row in prior:require_finished_parent(root/'runs'/row['batch'])
            b1=b1_parent_map(read(root/'runs'/plan['B1_execution']/'status.json'),'all_five_matured_v1')
            parents={(r['seed'],'H'):r['batch'] for r in plan['H_models']};parents.update({(s,'B1'):b for s,b in b1.items()})
            reference=read(root/'runs'/plan['H_models'][0]['batch']/'protocol.json')
            for seed in (11,22,33,44,55):
                for method in ('H','B1'):
                    verify();parent=root/'runs'/parents[(seed,method)];p=read(parent/'protocol.json')
                    if p['seed']!=seed or any(p.get(k)!=reference.get(k) for k in ('routes','native_case_manifest','native_validation','calibration_sha256','episode_seconds','road_geometry_version')):
                        raise ValueError('Cadence task identity differs')
                    for name,digest in plan['physical_source_hashes'].items():
                        if sha(parent/'source'/name)!=digest:raise ValueError('Parent physical source differs:'+name)
                    entries=[r for r in plan['preprepared'] if r['seed']==seed and r['method']==method]
                    if not entries:
                        status('PREPARING_CADENCE_PROBES',seed=seed,method=method);entries=preparation.prepare(parent,method)
                    if [r['hold_steps'] for r in entries]!=[2,5]:raise ValueError('Both registered cadences are required')
                    for row in entries:
                        verify();hold=row['hold_steps'];batch=root/'runs'/row['batch']
                        if sha(batch/'protocol.json')!=row['protocol_sha256']:raise ValueError('Prepared cadence changed')
                        healthy,reason=resources_ok(root)
                        if not healthy:raise RuntimeError(str(reason))
                        prefix='seed'+str(seed)+'_'+method+'_hold'+str(hold)
                        status('RUNNING_FROZEN_CADENCE_PROBE',seed=seed,method=method,hold_steps=hold,batch=batch.name)
                        call([shutil.which('pwsh'),'-NoProfile','-File',frozen/'scripts/run_registered_native.ps1','-Batch',batch,
                            '-Python',root/'.venv/Scripts/python.exe','-CarlaPython',root/'.carla-python37/runtime/python.exe',
                            '-ActivateQueued','-UseCurrentGuard'],prefix+'_lifecycle.log',4200)
                        require_finished_parent(batch);environment=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'))
                        call([sys.executable,frozen/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method',method],prefix+'_raw.log',300,environment)
                        inference=batch/method/'measured_start_inference.json'
                        call([sys.executable,frozen/'scripts/audit_native_policy_inference.py','--batch',batch,'--method',method,'--output',inference],prefix+'_inference.log',300,environment)
                        folder=batch/method;summary=read(folder/'summary.json');episodes=read(folder/'evaluation.json')['episodes'];raw=read(folder/'endpoint_raw_audit.json')
                        cfg=read(folder/'config.json');profile=read(batch/'protocol.json')['deployment_control_cadence']
                        if (summary['physics_steps'] or summary['optimizer_steps'] or summary['actor_l2_change']!=0
                                or cfg['deployment_control_cadence']!=profile or profile['hold_steps']!=hold
                                or sha(folder/'model.zip')!=sha(parent/method/'model.zip')
                                or sha(folder/'normalization.pkl')!=sha(parent/method/'normalization.pkl')):
                            raise ValueError('Frozen cadence model/configuration changed')
                        counts=cadence_trace_counts(folder/'evaluation_trace.jsonl.gz',method,hold)
                        count=len(reference['native_case_manifest']['cases'])
                        if (counts['physics_controls']!=raw['evaluation_control_frames_verified'] or len(read(inference)['results'])!=count
                                or sorted(r['case_id'] for r in episodes)!=list(range(count))):
                            raise ValueError('Every cadence frame/condition/action must verify')
                        timings=counts.pop('MPCrecomputation_seconds')
                        result={'scope':'Frozen-policy native control-cadence DEVELOPMENT stress,not retuned controller or retrained policy superiority',
                            'seed':seed,'method':method,'hold_steps':hold,'batch':batch.name,'physics_seconds':.05,
                            'command_update_hz':20/hold,'cases':count,'successes':sum(e['success'] for e in episodes),
                            'termination_counts':dict(Counter(e['termination_reason'] for e in episodes)),
                            'control_counts':counts,'MPCrecompute_only_timing':timing_summary(timings,hold*.05),
                            'baseline_parent':parent.name,'baseline_raw_audit_sha256':sha(parent/method/'endpoint_raw_audit.json'),
                            'baseline_episodes':read(parent/method/'evaluation.json')['episodes'],'episodes':episodes,
                            'model_sha256':sha(folder/'model.zip'),'normalization_sha256':sha(folder/'normalization.pkl'),
                            'new_training_controls':0,'new_optimizer_updates':0,'research_acceptance':'UNMET',
                            'limits':'Baseline20Hz trained endpoint reused. Hretains2Hz PPOand20Hz MPCdesign;B1policy frequency changes with held direct control. All event/safety rules and20Hz CARLAphysics stay fixed. Recomputed-only timings exclude holds;no full-stack real-time certification.'}
                        with (root/'reports'/('native_control_cadence_'+prefix+'_20261031.json')).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
                        completed.append({'seed':seed,'method':method,'hold_steps':hold,'batch':batch.name});status('CADENCE_PROBE_COMPLETE',seed=seed,method=method,hold_steps=hold)
            status('ALL_TWENTY_PAIRED_CONTROL_CADENCE_PROBES_COMPLETE')
    except BaseException as error:
        with (execution/'failure.json').open('x',encoding='utf-8') as stream:json.dump({'epoch':time.time(),'error':repr(error),'completed':completed,'scope':'No automatic experiment retry,seed omission,policy update or foreign kill'},stream,indent=2)
        raise
    finally:
        if owner is not None:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execution',required=True)
    run(parser.parse_args().execution)

"""Finite matched H/H0 replication after the exact owned predecessor finishes."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def require_preparation(state):
    rows=state.get('batches',[])
    if (state.get('phase')!='ALL_FIVE_MATCHED_BROADER_FEEDBACK_PREPARED_NOT_DRIVEN'
            or [r['seed'] for r in rows]!=[11,22,33,44,55]
            or len({r['batch'] for r in rows})!=5
            or state.get('distinct_actor_initializations_each_method')!={'H':5,'H0':5}):
        raise ValueError('All five independently initialized matched H/H0 pairs must be registered')
    return rows


def require_camera_completion(state,failure_present):
    if (failure_present or state.get('phase')!='ALL_REGISTERED_NATIVE_PPO_RECORDINGS_AND_FRAME_VERIFICATIONS_COMPLETE'
            or [r['method'] for r in state.get('completed',[])]!=['H','B1']):
        raise ValueError('Exact owned predecessor must finish its native cleanup and complete recording audits')


def require_corrective_probe_pair(rows):
    if (not isinstance(rows,list) or [r.get('arm') for r in rows]!=['replay','trajectory_correction']
            or len({r.get('batch') for r in rows})!=2
            or len({r.get('model_sha256') for r in rows})!=2
            or len({r.get('normalization_sha256') for r in rows})!=1):
        raise ValueError('Require both distinct matched actor arms in fixed order with shared normalization')
    return rows


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;source=execution/'source'
    sys.path.insert(0,str(source))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.audit_native_ppo_configuration import profile
    from scripts.evaluate_native_rule import require_finished_parent
    plan=read(execution/'plan.json');hashes=read(execution/'source_manifest.json')
    rows=require_preparation(read(root/'runs'/plan['preparation']/'status.json'))
    probes=require_corrective_probe_pair(plan['pretraining_B1_corrective_probe']) if 'pretraining_B1_corrective_probe' in plan else []
    completed=[];owner=None
    def status(phase,**extra):
        atomic_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time(),'completed':completed,**extra})
    def verify():
        for name,digest in hashes.items():
            if sha(source/name)!=digest:raise ValueError('Frozen broader observer changed:'+name)
        reference=None
        for row in rows:
            batch=root/'runs'/row['batch'];path=batch/'queued_protocol.json' if (batch/'queued_protocol.json').is_file() else batch/'protocol.json'
            if sha(path)!=row['protocol_sha256']:raise ValueError('Original registered protocol changed')
            p=read(batch/'protocol.json')
            if (p['seed']!=row['seed'] or p['methods']!=['H','H0'] or p['method_budgets']!={'H':20480,'H0':20480}
                    or p.get('evaluation_only') or not p.get('freeze_observation_statistics')
                    or len(p['native_case_manifest']['cases'])!=36 or len(p['native_validation']['manifest']['cases'])!=24
                    or any(p['recovery'][m]['lineage_controls'] or p['recovery'][m]['retained_optimizer_steps'] for m in ('H','H0'))):
                raise ValueError('Matched recipe,pristine lineage or partition inventory differs')
            if reference is None:reference=p['source_hashes']
            if p['source_hashes']!=reference:raise ValueError('All five pairs must use exactly the same complete source tree')
            for name,digest in p['source_hashes'].items():
                if sha(batch/'source'/name)!=digest:raise ValueError('Frozen broader worker changed:'+name)
        probe_source=None;probe_env=None
        for row in probes:
            batch=root/'runs'/row['batch'];path=batch/'queued_protocol.json' if (batch/'queued_protocol.json').is_file() else batch/'protocol.json'
            if sha(path)!=row['protocol_sha256']:raise ValueError('Registered corrective probe protocol changed')
            p=read(batch/'protocol.json');origin=batch/'resume/B1'
            if (p['methods']!=['B1'] or p.get('evaluation_only') is not True or p['method_budgets']!={'B1':0}
                    or len(p['native_case_manifest']['cases'])!=4 or p.get('native_validation')
                    or p['trained_actor_correction']['arm']!=row['arm']
                    or p['trained_actor_correction']['additional_supervised_steps']!=512
                    or sha(origin/'model.zip')!=row['model_sha256']
                    or sha(origin/'normalization.pkl')!=row['normalization_sha256']):
                raise ValueError('Matched corrective probe semantics or artifacts changed')
            env=read(origin/'config.json')['env']
            if probe_source is None:probe_source=p['source_hashes'];probe_env=env
            if p['source_hashes']!=probe_source or env!=probe_env:raise ValueError('Corrective arms change execution or task')
            for name,digest in p['source_hashes'].items():
                if sha(batch/'source'/name)!=digest:raise ValueError('Frozen corrective worker changed:'+name)
    def call(command,label,timeout,env=None):
        with (execution/(label+'.log')).open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json(execution/'last_child.json',{'pid':child.pid,'epoch':time.time(),'label':label})
            code=child.wait(timeout=timeout)
        if code:raise RuntimeError('Registered broader child failed;no automatic retry:'+label)
    try:
        verify()
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'epoch':time.time(),'plan_sha256':sha(execution/'plan.json')},f)
            expected={m:profile(root/'runs'/rows[0]['batch']/'resume'/m) for m in ('H','H0')}
            for row in rows:
                for method in ('H','H0'):
                    if profile(root/'runs'/row['batch']/'resume'/method)!=expected[method]:
                        raise ValueError('Actual saved pristine learner recipe differs')
            atomic_json(execution/'all_pristine_saved_profiles_verified.json',{'seeds':[11,22,33,44,55],'methods':['H','H0'],'zero_PPO_and_empty_Adam_required':True})
            identity=plan['dependency_process'];owner=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
            until=time.monotonic()+12*3600
            while owner.poll() is None:
                if time.monotonic()>until:raise TimeoutError('Finite owned predecessor wait exhausted')
                status('WAITING_FOR_OWNED_PREDECESSOR_NATIVE_QUEUE',dependency_pid=owner.pid)
                time.sleep(10)
            predecessor=root/'runs'/plan['dependency_execution']
            prior=read(predecessor/'status.json');require_camera_completion(prior,(predecessor/'failure.json').is_file())
            for row in prior['completed']:require_finished_parent(root/'runs'/row['batch'])
            probe_results=[]
            for row in probes:
                verify();okay,reason=resources_ok(root)
                if not okay:raise RuntimeError(str(reason))
                batch=root/'runs'/row['batch'];prefix='corrective_'+row['arm']
                status('EVALUATING_BOTH_MATCHED_TRAINED_B1_ACTOR_ARMS',arm=row['arm'],batch=batch.name)
                call([shutil.which('pwsh'),'-NoProfile','-File',source/'scripts/run_registered_native.ps1',
                    '-Batch',batch,'-Python',root/'.venv/Scripts/python.exe','-CarlaPython',root/'.carla-python37/runtime/python.exe',
                    '-ActivateQueued','-UseCurrentGuard'],prefix+'_lifecycle',2400)
                require_finished_parent(batch);env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'));folder=batch/'B1'
                call([sys.executable,batch/'source/scripts/audit_native_method_endpoint.py','--batch',batch,'--method','B1'],prefix+'_raw',300,env)
                call([sys.executable,batch/'source/scripts/audit_native_policy_inference.py','--batch',batch,'--method','B1',
                    '--output',folder/'measured_start_inference.json'],prefix+'_inference',300,env)
                summary=read(folder/'summary.json');evaluation=read(folder/'evaluation.json');audit=read(folder/'endpoint_raw_audit.json')
                episodes=evaluation['episodes']
                if (summary['physics_steps'] or summary['optimizer_steps']
                        or len(episodes)!=4 or sorted(e['case_id'] for e in episodes)!=list(range(4))
                        or len(read(folder/'measured_start_inference.json')['results'])!=4
                        or evaluation['model_sha256']!=row['model_sha256']
                        or evaluation['normalization_sha256']!=row['normalization_sha256']):
                    raise ValueError('Incomplete frozen four-case corrective diagnostic')
                probe_results.append({'arm':row['arm'],'batch':batch.name,'cases':4,'successes':sum(e['success'] for e in episodes),
                    'termination_counts':dict(Counter(e['termination_reason'] for e in episodes)),
                    'raw_controls_verified':audit['total_applied_controls_verified'],'PPO_updates':0,
                    'additional_supervised_updates':512,'model_sha256':evaluation['model_sha256'],
                    'normalization_sha256':evaluation['normalization_sha256'],'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json')})
                atomic_json(execution/'completed_corrective_probe_arms.json',probe_results)
            if probes:
                with (root/'reports/native_B1_matched_trained_actor_correction_20261101.json').open('x',encoding='utf-8') as f:
                    json.dump({'scope':'All4known basic cases after two matched512step trained actor fits;one development seed,not final or healthy full baseline',
                        'arms':probe_results,'new_PPO_updates':0,'research_acceptance':'UNMET'},f,indent=2)
            for row in rows:
                verify();okay,reason=resources_ok(root)
                if not okay:raise RuntimeError(str(reason))
                batch=root/'runs'/row['batch'];prefix='seed'+str(row['seed'])
                status('TRAINING_MATCHED_BROADER_H_H0',seed=row['seed'],batch=batch.name)
                call([shutil.which('pwsh'),'-NoProfile','-File',source/'scripts/run_registered_native.ps1',
                    '-Batch',batch,'-Python',root/'.venv/Scripts/python.exe','-CarlaPython',root/'.carla-python37/runtime/python.exe',
                    '-ActivateQueued','-UseCurrentGuard'],prefix+'_lifecycle',7800)
                require_finished_parent(batch);results={}
                for method in ('H','H0'):
                    env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'));folder=batch/method
                    call([sys.executable,batch/'source/scripts/audit_native_method_endpoint.py','--batch',batch,'--method',method],prefix+'_'+method+'_raw',300,env)
                    call([sys.executable,batch/'source/scripts/audit_native_policy_inference.py','--batch',batch,'--method',method,
                        '--output',folder/'measured_start_inference.json'],prefix+'_'+method+'_inference',300,env)
                    summary=read(folder/'summary.json');evaluation=read(folder/'evaluation.json');audit=read(folder/'endpoint_raw_audit.json')
                    episodes=evaluation['episodes']
                    if (summary['physics_steps']<20480 or summary['optimizer_steps']<=0
                            or audit['training_control_frames_verified']!=summary['physics_steps']
                            or len(episodes)!=24 or sorted(e['case_id'] for e in episodes)!=list(range(24))
                            or len(read(folder/'measured_start_inference.json')['results'])!=24):
                        raise ValueError('Incomplete actual PPO or24condition evaluation')
                    results[method]={'cases':24,'successes':sum(e['success'] for e in episodes),
                        'termination_counts':dict(Counter(e['termination_reason'] for e in episodes)),
                        'training_controls':summary['physics_steps'],'actual_Adam_steps':summary['optimizer_steps'],
                        'raw_controls_verified':audit['total_applied_controls_verified'],
                        'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
                        'model_sha256':evaluation['model_sha256'],'normalization_sha256':evaluation['normalization_sha256']}
                result={'scope':'Matched H/H0PPO on36TRAIN/24known development-validation conditions;not final confirmation',
                    'seed':row['seed'],'batch':batch.name,'methods':results,'research_acceptance':'UNMET'}
                with (root/'reports'/('native_matched_broader_feedback_seed'+str(row['seed'])+'_endpoint_20261101.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
                completed.append({'seed':row['seed'],'batch':batch.name});status('MATCHED_BROADER_SEED_COMPLETE',seed=row['seed'])
            status('ALL_FIVE_MATCHED_BROADER_FEEDBACK_PPO_PAIRS_COMPLETE')
    except BaseException as error:
        with (execution/'failure.json').open('x',encoding='utf-8') as f:json.dump({'error':repr(error),'epoch':time.time(),'completed':completed},f,indent=2)
        raise
    finally:
        if owner is not None:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True)
    args=parser.parse_args();run(args.execution)

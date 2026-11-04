"""Run all five teacher-only controls after the exact owned matched-PPO cohort."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def require_pristine_preparation(state):
    rows=state.get('batches',[])
    if (state.get('phase')!='ALL_FIVE_PRISTINE_H_BASELINES_PREPARED_NOT_DRIVEN'
            or [r.get('seed') for r in rows]!=[11,22,33,44,55]
            or len({r.get('batch') for r in rows})!=5
            or len({r.get('paired_training_batch') for r in rows})!=5
            or len({r.get('actor_parameter_sha256') for r in rows})!=5
            or state.get('distinct_pristine_actor_initializations')!=5):
        raise ValueError('All five distinct pristine matched controls are required')
    return rows


def require_training_completion(state,failure_present):
    if (failure_present or state.get('phase')!='ALL_FIVE_MATCHED_BROADER_FEEDBACK_PPO_PAIRS_COMPLETE'
            or [r.get('seed') for r in state.get('completed',[])]!=[11,22,33,44,55]):
        raise ValueError('Every registered matched PPO pair and owned cleanup must complete first')


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;source=execution/'source';sys.path.insert(0,str(source))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.audit_native_ppo_configuration import profile
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.analyze_native_pristine_baselines import rule_compatibility,analyze
    plan=read(execution/'plan.json');hashes=read(execution/'source_manifest.json')
    rows=require_pristine_preparation(read(root/'runs'/plan['preparation']/'status.json'));owner=None;completed=[]
    def status(phase,**extra):
        atomic_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time(),'completed':completed,**extra})
    def verify():
        for name,digest in hashes.items():
            if sha(source/name)!=digest:raise ValueError('Frozen pristine observer changed')
        common_source=None
        for row in rows:
            batch=root/'runs'/row['batch'];original=batch/'queued_protocol.json' if (batch/'queued_protocol.json').exists() else batch/'protocol.json'
            if sha(original)!=row['protocol_sha256']:raise ValueError('Registered pristine protocol changed')
            p=read(batch/'protocol.json');origin=batch/'resume/H'
            if (p['methods']!=['H'] or p.get('evaluation_only') is not True or p['method_budgets']!={'H':0}
                    or p['recovery']['H']['lineage_controls'] or p['recovery']['H']['retained_optimizer_steps']
                    or len(p['native_validation']['manifest']['cases'])!=24
                    or p['pristine_policy_baseline']['paired_training_batch']!=row['paired_training_batch']
                    or sha(origin/'model.zip')!=row['model_sha256']
                    or sha(origin/'normalization.pkl')!=row['normalization_sha256']):
                raise ValueError('Pristine lineage,all-condition inventory or artifact differs')
            if common_source is None:common_source=p['source_hashes']
            if p['source_hashes']!=common_source:raise ValueError('Pristine worker source differs by seed')
            for name,digest in p['source_hashes'].items():
                if sha(batch/'source'/name)!=digest:raise ValueError('Frozen pristine worker changed')
    def call(command,label,timeout,env=None):
        with (execution/(label+'.log')).open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json(execution/'last_child.json',{'pid':child.pid,'label':label,'epoch':time.time()})
            code=child.wait(timeout=timeout)
        if code:raise RuntimeError('Pristine control child failed;no retry:'+label)
    try:
        verify()
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'epoch':time.time()},stream)
            expected=profile(root/'runs'/rows[0]['batch']/'resume/H')
            for row in rows:
                if profile(root/'runs'/row['batch']/'resume/H')!=expected:raise ValueError('Actual zero-PPO saved recipes differ')
            proof=rule_compatibility(root/'runs'/rows[0]['paired_training_batch'],root/'runs'/plan['strong_rule_batch'])
            atomic_json(execution/'reused_strong_rule_compatibility.json',proof)
            identity=plan['dependency_process'];owner=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
            until=time.monotonic()+43200
            while owner.poll() is None:
                if time.monotonic()>until:raise TimeoutError('Finite owned-cohort wait exhausted')
                status('WAITING_FOR_EXACT_OWNED_MATCHED_PPO_COHORT',dependency_pid=owner.pid);time.sleep(10)
            dependency=root/'runs'/plan['dependency_execution']
            require_training_completion(read(dependency/'status.json'),(dependency/'failure.json').is_file())
            for row in rows:require_finished_parent(root/'runs'/row['paired_training_batch'])
            for row in rows:
                verify();okay,reason=resources_ok(root)
                if not okay:raise RuntimeError(str(reason))
                batch=root/'runs'/row['batch'];prefix='seed'+str(row['seed'])
                status('EVALUATING_PRISTINE_TEACHER_ONLY_H',seed=row['seed'],batch=batch.name)
                call([shutil.which('pwsh'),'-NoProfile','-File',source/'scripts/run_registered_native.ps1',
                    '-Batch',batch,'-Python',root/'.venv/Scripts/python.exe','-CarlaPython',root/'.carla-python37/runtime/python.exe',
                    '-ActivateQueued','-UseCurrentGuard'],prefix+'_lifecycle',2400)
                require_finished_parent(batch);env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'));folder=batch/'H'
                call([sys.executable,batch/'source/scripts/audit_native_method_endpoint.py','--batch',batch,'--method','H'],prefix+'_raw',300,env)
                call([sys.executable,batch/'source/scripts/audit_native_policy_inference.py','--batch',batch,'--method','H',
                    '--output',folder/'measured_start_inference.json'],prefix+'_inference',300,env)
                summary=read(folder/'summary.json');evaluation=read(folder/'evaluation.json')
                if (summary['physics_steps'] or summary['optimizer_steps'] or summary['lineage_control_steps']
                        or len(evaluation['episodes'])!=24 or len(read(folder/'measured_start_inference.json')['results'])!=24
                        or evaluation['model_sha256']!=row['model_sha256'] or evaluation['normalization_sha256']!=row['normalization_sha256']):
                    raise ValueError('Incomplete teacher-only evaluation or unexpected learning')
                completed.append({'seed':row['seed'],'batch':batch.name});status('PRISTINE_TEACHER_ONLY_SEED_COMPLETE',seed=row['seed'])
            result=analyze(root,rows,plan['strong_rule_batch'],root/'reports/native_broader_PPO_teacher_rule_attribution_20261101.json')
            atomic_json(execution/'analysis_receipt.json',{'report':'native_broader_PPO_teacher_rule_attribution_20261101.json',
                'report_sha256':sha(root/'reports/native_broader_PPO_teacher_rule_attribution_20261101.json'),
                'unique_conditions':result['unique_conditions'],'independently_PPO_trained_policies':5})
            status('ALL_FIVE_PRISTINE_BASELINES_AND_PPO_ATTRIBUTION_COMPLETE')
    except BaseException as error:
        with (execution/'failure.json').open('x',encoding='utf-8') as stream:json.dump({'error':repr(error),'completed':completed,'epoch':time.time()},stream,indent=2)
        raise
    finally:
        if owner is not None:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True)
    args=parser.parse_args();run(args.execution)

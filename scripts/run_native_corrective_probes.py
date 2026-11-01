"""Finite matched B1 diagnostic after the exact owned teacher-control queue."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def require_teacher_completion(state,failure_present):
    if (failure_present or state.get('phase')!='ALL_FIVE_PRISTINE_BASELINES_AND_PPO_ATTRIBUTION_COMPLETE'
            or [r.get('seed') for r in state.get('completed',[])]!=[11,22,33,44,55]):
        raise ValueError('All five teacher controls and owned cleanup must finish first')


def summarize_probe(evaluation,summary,audit,expected):
    episodes=evaluation['episodes']
    if (sorted(e['case_id'] for e in episodes)!=[0,1,2,3]
            or summary['physics_steps'] or summary['optimizer_steps']
            or audit['training_control_frames_verified']
            or evaluation['model_sha256']!=expected['model_sha256']
            or evaluation['normalization_sha256']!=expected['normalization_sha256']):
        raise ValueError('Every original case, unchanged endpoint and zero new PPO are required')
    unsafe=sum(e['termination_reason'] in ('collision','out_of_bounds') for e in episodes)
    return {**expected,'cases':4,'successes':sum(e['success'] for e in episodes),
        'termination_counts':dict(Counter(e['termination_reason'] for e in episodes)),
        'case_results':[{'case_id':e['case_id'],'success':e['success'],
            'termination_reason':e['termination_reason'],'controls':e['physics_steps']} for e in episodes],
        'unsafe_cases':unsafe,'basic_gate_passed':sum(e['success'] for e in episodes)>=3 and unsafe==0,
        'raw_controls_verified':audit['total_applied_controls_verified'],'new_PPO_updates':0,
        'additional_supervised_updates_before_evaluation':512}


def verify_completed_reuse(root,row):
    """Reuse a completed deterministic control only when full saved state matches."""
    from navstack.corrective_imitation import _exact
    from navstack.independent_clip_ppo import load_ppo
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    batch=root/'runs'/row['batch'];require_finished_parent(batch)
    equivalent=root/row['equivalent_prepared_checkpoint']
    original=batch/'B1/model.zip'
    if sha(equivalent)!=row['equivalent_prepared_model_sha256']:raise ValueError('Equivalent replay endpoint changed')
    left=load_ppo(original,device='cpu');right=load_ppo(equivalent,device='cpu')
    if (not _exact(left.policy.state_dict(),right.policy.state_dict())
            or not _exact(left.policy.optimizer.state_dict(),right.policy.optimizer.state_dict())
            or (left.num_timesteps,left._n_updates)!=(right.num_timesteps,right._n_updates)
            or sha(equivalent.with_name('normalization.pkl'))!=row['normalization_sha256']):
        raise ValueError('Completed replay has different policy, PPO state, counters or normalization')
    for name,digest in row['reused_result_hashes'].items():
        if sha(batch/'B1'/name)!=digest:raise ValueError('Completed replay evidence changed')
    return {'policy_and_PPO_optimizer_tensors_exact':True,'normalization_bytes_exact':True,
            'completed_batch':batch.name,'new_replay_driving_controls':0}


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;source=execution/'source'
    sys.path.insert(0,str(source))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.run_native_broader_feedback_cohort import require_corrective_probe_pair
    plan=read(execution/'plan.json');hashes=read(execution/'source_manifest.json')
    rows=require_corrective_probe_pair(plan['batches']);completed=[];owner=None
    def status(phase,**extra):
        atomic_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time(),'completed':completed,**extra})
    def verify():
        for name,digest in hashes.items():
            if sha(source/name)!=digest:raise ValueError('Frozen observer changed:'+name)
        previous=None;environment=None
        for row in rows:
            batch=root/'runs'/row['batch']
            original=batch/'queued_protocol.json' if (batch/'queued_protocol.json').exists() else batch/'protocol.json'
            if sha(original)!=row['protocol_sha256']:raise ValueError('Registered diagnostic protocol changed')
            p=read(batch/'protocol.json');origin=batch/'resume/B1'
            if (p['methods']!=['B1'] or p.get('evaluation_only') is not True or p['method_budgets']!={'B1':0}
                    or len(p['native_case_manifest']['cases'])!=4 or p.get('native_validation')
                    or p['trained_actor_correction']['arm']!=row['arm']
                    or p['trained_actor_correction']['additional_supervised_steps']!=512
                    or sha(origin/'model.zip')!=row['model_sha256']
                    or sha(origin/'normalization.pkl')!=row['normalization_sha256']):
                raise ValueError('Matched diagnostic semantics or endpoint changed')
            env=read(origin/'config.json')['env']
            if previous is None:previous=p['source_hashes'];environment=env
            if previous!=p['source_hashes'] or environment!=env:raise ValueError('Arms change native execution or task')
            for name,digest in p['source_hashes'].items():
                if sha(batch/'source'/name)!=digest:raise ValueError('Frozen native source changed:'+name)
            if row.get('reuse_completed'):
                equivalent=root/row['equivalent_prepared_checkpoint']
                if sha(equivalent)!=row['equivalent_prepared_model_sha256']:raise ValueError('Equivalent replay model changed')
                for name,digest in row['reused_result_hashes'].items():
                    if sha(batch/'B1'/name)!=digest:raise ValueError('Reused replay evidence changed')
    def call(command,label,timeout,env=None):
        with (execution/(label+'.log')).open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json(execution/'last_child.json',{'pid':child.pid,'label':label,'epoch':time.time()})
            code=child.wait(timeout=timeout)
        if code:raise RuntimeError('Diagnostic child failed; no automatic retry:'+label)
    try:
        verify()
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x',encoding='utf-8') as stream:
                json.dump({'pid':os.getpid(),'epoch':time.time(),'plan_sha256':sha(execution/'plan.json')},stream)
            for row in rows:
                if row.get('reuse_completed'):
                    atomic_json(execution/'completed_replay_equivalence.json',verify_completed_reuse(root,row))
            identity=plan['dependency_process']
            owner=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
            until=time.monotonic()+43200
            while owner.poll() is None:
                if time.monotonic()>until:raise TimeoutError('Finite owned predecessor wait exhausted')
                status('WAITING_FOR_EXACT_OWNED_TEACHER_CONTROL_QUEUE',dependency_pid=owner.pid);time.sleep(10)
            dependency=root/'runs'/plan['dependency_execution']
            prior=read(dependency/'status.json');require_teacher_completion(prior,(dependency/'failure.json').exists())
            for row in prior['completed']:require_finished_parent(root/'runs'/row['batch'])
            for row in rows:
                verify();okay,reason=resources_ok(root)
                if not okay:raise RuntimeError(str(reason))
                batch=root/'runs'/row['batch'];prefix=row['arm']
                status('EVALUATING_MATCHED_B1_TRAINING_COVERAGE_PROBE',arm=prefix,batch=batch.name)
                if not row.get('reuse_completed'):
                    call([shutil.which('pwsh'),'-NoProfile','-File',source/'scripts/run_registered_native.ps1',
                        '-Batch',batch,'-Python',root/'.venv/Scripts/python.exe',
                        '-CarlaPython',root/'.carla-python37/runtime/python.exe','-ActivateQueued','-UseCurrentGuard'],prefix+'_lifecycle',2400)
                require_finished_parent(batch);folder=batch/'B1'
                env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'))
                if not row.get('reuse_completed'):
                    call([sys.executable,batch/'source/scripts/audit_native_method_endpoint.py','--batch',batch,'--method','B1'],prefix+'_raw',300,env)
                    call([sys.executable,batch/'source/scripts/audit_native_policy_inference.py','--batch',batch,'--method','B1',
                        '--output',folder/'measured_start_inference.json'],prefix+'_inference',300,env)
                result=summarize_probe(read(folder/'evaluation.json'),read(folder/'summary.json'),read(folder/'endpoint_raw_audit.json'),row)
                if len(read(folder/'measured_start_inference.json')['results'])!=4:raise ValueError('Every original start must reproduce')
                result['raw_audit_sha256']=sha(folder/'endpoint_raw_audit.json');completed.append(result)
                status('MATCHED_B1_PROBE_ARM_COMPLETE',arm=prefix)
            result={'scope':plan['scope'],'arms':completed,'new_PPO_updates':0,'research_acceptance':'UNMET;one known development seed,not a qualified five-seed baseline'}
            with (root/'reports'/plan['output_report']).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
            status('BOTH_MATCHED_B1_TRAINING_COVERAGE_PROBES_COMPLETE',report=plan['output_report'])
    except BaseException as error:
        atomic_json(execution/'failure.json',{'error':repr(error),'completed':completed,'epoch':time.time()});raise
    finally:
        if owner is not None:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True)
    run(parser.parse_args().execution)

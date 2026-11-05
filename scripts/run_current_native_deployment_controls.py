"""Execute eight frozen development probes after the conditional pilot releases2120."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent


def job_inventory(plan):
    from scripts.prepare_current_native_deployment_controls import VARIANTS,variant_label
    expected={variant_label(method,kind,value) for method in ('H','B1') for kind,value in VARIANTS}
    jobs=plan.get('jobs',[])
    if (len(jobs)!=8 or {job['label'] for job in jobs}!=expected
            or any(variant_label(j['method'],j['kind'],j['value'])!=j['label'] for j in jobs)):
        raise ValueError('All eight distinct H/B1deployment variants are required')
    return jobs


def shared_brake_counts(path):
    counts=dict(physics_controls=0,policy_decisions=0,common_brake_controls=0,
                all_intervention_controls=0,parking_assist_controls=0)
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            info=json.loads(line)['info'];ticks=info['trace']
            if len(ticks)!=info['executed_substeps'] or not ticks:raise ValueError('Incomplete shared-brake trace')
            counts['policy_decisions']+=1
            for tick in ticks:
                common=tick.get('shared_shield_intervention')
                if type(common) is not bool:raise ValueError('Every common brake frame must be explicit')
                counts['physics_controls']+=1
                counts['common_brake_controls']+=common
                counts['all_intervention_controls']+=bool(tick['intervention'])
                counts['parking_assist_controls']+=bool(tick.get('goal_parking_intervention',False))
    return counts


def validate(output):
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    output=Path(output).resolve();plan=read(output/'registration.json')
    for relative,digest in plan['driver_source_hashes'].items():
        path=(output/'source'/relative).resolve()
        if not path.is_relative_to(output/'source') or sha(path)!=digest:raise ValueError('Frozen deployment driver changed')
    for job in job_inventory(plan):
        if Path(job['batch']).name!=job['batch']:raise ValueError('Job must be a sibling batch')
        batch=output.parent/job['batch'];p=read(batch/'protocol.json');method=job['method']
        if (sha(batch/'protocol.json')!=job['queued_protocol_sha256']
                or read(batch/'prepared.json')['protocol_sha256']!=job['queued_protocol_sha256']
                or p['methods']!=[method] or not p['evaluation_only']
                or p['method_budgets']!={method:0} or p['carla_rpc_port_span']!=[2120,2121,2122]):
            raise ValueError('Frozen deployment protocol changed')
        for relative,digest in p['source_hashes'].items():
            path=(batch/'source'/relative).resolve()
            if not path.is_relative_to(batch/'source') or sha(path)!=digest:raise ValueError('Frozen deployment worker changed')
        meta=verified_checkpoint_metadata(batch/'resume'/method)
        if meta['model_sha256']!=job['model_sha256'] or meta['normalization_sha256']!=job['normalization_sha256']:
            raise ValueError('Frozen deployment learning bytes changed')
    return plan


def dependency_closed(pilot):
    pilot=Path(pilot);receipt=pilot/'execution/complete.json'
    if not receipt.exists():return False
    plan=read(pilot/'registration.json');result=read(receipt);rows=result.get('results',[])
    if len(rows)!=2 or {r['arm'] for r in rows}!={'Hc','Hc0'}:raise ValueError('Incomplete conditional pilot result')
    for registered in plan['arms']:
        row=next(r for r in rows if r['arm']==registered['arm'])
        if row['training']!=registered['batch']:raise ValueError('Pilot training identity changed')
        training=pilot.parent/row['training'];endpoint=pilot.parent/row['endpoint']
        for batch in (training,endpoint):require_finished_parent(batch)
        activation=read(training/'activation.json')
        if (activation['original_protocol_sha256']!=registered['queued_protocol_sha256']
                or activation['activated_protocol_sha256']!=sha(training/'protocol.json')
                or read(endpoint/'protocol.json')['fixed_lowlevel_endpoint']['parent_protocol_sha256']!=sha(training/'protocol.json')
                or row['model_sha256']!=sha(endpoint/'H/model.zip')):
            raise ValueError('Pilot lifecycle/protocol/model evidence changed')
    return True


def run(output):
    from navstack.owned_process import OwnedProcess
    from scripts.run_independent_outcome_cohort import resource_admission
    from scripts.run_native_control_cadence_cohort import cadence_trace_counts
    output=Path(output).resolve();plan=validate(output);execution=output/'execution';execution.mkdir()
    pilot=Path(plan['dependency_cohort'])
    if sha(pilot/'registration.json')!=plan['dependency_sha256']:raise ValueError('Pilot dependency changed')
    def status(phase,**fields):atomic_json(execution/'status.json',dict(phase=phase,pid=os.getpid(),epoch=time.time(),**fields))
    def call(arguments,name,env=None,timeout=1800):
        with (execution/(name+'.log')).open('xb') as log:
            subprocess.run(list(map(str,arguments)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    atomic_json(execution/'registration.json',dict(plan_sha256=sha(output/'registration.json'),
        driver_sha256=sha(Path(__file__)),maximum_native_jobs=8,automatic_retries=0))
    try:
        if not dependency_closed(pilot):
            owner=OwnedProcess(**plan['dependency_identity'],allow_terminate=False);until=time.monotonic()+plan['maximum_wait_seconds']
            while not dependency_closed(pilot):
                if owner.poll() is not None or (pilot/'execution/failure.json').exists() or time.monotonic()>=until:
                    raise RuntimeError('Pilot dependency incomplete/stopped or bounded wait expired;no deployment launch')
                status('WAITING_FOR_CONDITIONAL_PILOT',dependency_pid=plan['dependency_identity']['pid'],new_native_jobs=0)
                time.sleep(30)
        atomic_json(execution/'dependency_complete.json',dict(receipt_sha256=sha(pilot/'execution/complete.json'),closed_native_jobs=4))
        results=[]
        for job in job_inventory(plan):
            batch=output.parent/job['batch'];method=job['method'];label=job['label'];p=read(batch/'protocol.json')
            status(label+'_NATIVE',completed_jobs=len(results));atomic_json(execution/(label+'_resources.json'),resource_admission(output.parent))
            call([shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1',
                '-Batch',batch,'-Python',sys.executable,'-CarlaPython',plan['runtime_carla_python'],
                '-UseCurrentGuard','-ActivateQueued'],label+'_native',timeout=15000)
            require_finished_parent(batch)
            env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
            call([sys.executable,ROOT/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method',method],label+'_raw',env)
            call([sys.executable,ROOT/'scripts/audit_native_control_delivery.py','--batch',batch,'--output',batch/'strict_delivery.json'],label+'_engine',env)
            if not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']:raise ValueError('Deployment engine delivery differs')
            inference=batch/method/('all_recorded_action_inference.json' if method=='H' else 'measured_start_inference.json')
            auditor='audit_native_reference_inference.py' if method=='H' else 'audit_native_policy_inference.py'
            arguments=[sys.executable,ROOT/'scripts'/auditor,'--batch',batch,'--method',method,'--output',inference]
            if method=='H':arguments+=['--all-actions']
            call(arguments,label+'_inference',env,timeout=3600)
            summary=read(batch/method/'summary.json');raw=read(batch/method/'endpoint_raw_audit.json');evaluation=read(batch/method/'evaluation.json');episodes=evaluation['episodes']
            if (summary['physics_steps'] or summary['optimizer_steps'] or summary['actor_l2_change']!=0
                    or len(episodes)!=24 or {r['case_id'] for r in episodes}!=set(range(24))
                    or sha(batch/method/'model.zip')!=job['model_sha256']
                    or sha(batch/method/'normalization.pkl')!=job['normalization_sha256']):
                raise ValueError('Deployment changed model/RMS/learning counters or omitted conditions')
            if job['kind']=='cadence':
                counts=cadence_trace_counts(batch/method/'evaluation_trace.jsonl.gz',method,job['value'],
                    reference_centered_effort=True if method=='H' else None)
                from navstack.metrics import timing_summary
                recompute_timing=timing_summary(counts.pop('MPCrecomputation_seconds'),job['value']*.05)
            else:
                counts=shared_brake_counts(batch/method/'evaluation_trace.jsonl.gz');recompute_timing=None
                if job['value'] is False and counts['common_brake_controls']!=0:raise ValueError('OFFarm applied common brake')
            if counts['physics_controls']!=raw['evaluation_control_frames_verified']:raise ValueError('Deployment frame accounting differs')
            result=dict(**job,protocol_sha256=sha(batch/'protocol.json'),success=sum(r['success'] for r in episodes),
                termination_reasons=dict(Counter(r['termination_reason'] for r in episodes)),episodes=episodes,
                control_counts=counts,MPC_recompute_only_timing=recompute_timing,
                baseline_episodes=read(output.parent/job['baseline']/method/'evaluation.json')['episodes'],
                new_training_controls=0,new_optimizer_updates=0,new_final_conditions=0,
                limits='One selected seed11deployment study;no retraining/retuned controller/five-seed or real-time certification claim')
            atomic_json(execution/(label+'_complete.json'),result);results.append(result)
        atomic_json(execution/'complete.json',dict(results=results,research_acceptance='UNMET',unique_known_conditions=24,
            scope='Eight frozen-policy deployment sensitivity studies;no final confirmation or independent new trained seeds'))
        status('COMPLETE',completed_jobs=8)
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if args.validate_only:validate(args.output);print('PASS: eight frozen deployment probes validated;no native launch')
    else:run(args.output)

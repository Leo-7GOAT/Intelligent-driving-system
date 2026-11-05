"""Wait on the authenticated flat queue, then run the two frozen pilot arms."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha


def dependency_inventory(result):
    expected={(seed,phase) for seed in (11,22,33,44,55)
              for phase in ('zero_PPO_basic','TRAIN48_PPO30720','basic','known24')}
    rows=result.get('results',[])
    if (result.get('seeds')!=[11,22,33,44,55] or len(rows)!=20
            or {(r['seed'],r['phase']) for r in rows}!=expected):
        raise ValueError('All20distinct matched-flat phases are required')
    return rows


def validate(output):
    from navstack.contextual_feedback_registration import registered
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    output=Path(output).resolve();plan=read(output/'registration.json')
    if [row['arm'] for row in plan['arms']]!=['Hc','Hc0'] or plan['controls_per_arm']!=10240:
        raise ValueError('Require the complete paired pilot')
    for relative,digest in plan['source_hashes'].items():
        path=(output/'source'/relative).resolve()
        if not path.is_relative_to(output/'source') or sha(path)!=digest:raise ValueError('Pilot source changed:'+relative)
    for row in plan['arms']:
        if Path(row['batch']).name!=row['batch']:raise ValueError('Arm must be a sibling batch')
        batch=output.parent/row['batch'];p=read(batch/'protocol.json')
        if sha(batch/'protocol.json')!=row['queued_protocol_sha256'] or read(batch/'prepared.json')['protocol_sha256']!=row['queued_protocol_sha256']:
            raise ValueError('Queued arm protocol changed')
        if (not registered(p) or p['carla_rpc_port_span']!=[2120,2121,2122]
                or p['contextual_feedback_registration']['ablated']!=(row['arm']=='Hc0')):
            raise ValueError('Pilot admission or transport changed')
        for relative,digest in plan['source_hashes'].items():
            if sha(batch/'source'/relative)!=digest:raise ValueError('Arm source changed:'+relative)
        meta=verified_checkpoint_metadata(batch/'resume/H')
        if meta['model_sha256']!=row['initializer_model_sha256']:raise ValueError('Initializer changed')
    return plan


def run(output):
    from navstack.owned_process import OwnedProcess
    from scripts.evaluate_native_rule import require_finished_parent
    output=Path(output).resolve();plan=validate(output);execution=output/'execution';execution.mkdir()
    flat=Path(plan['dependency_cohort']);dependency=flat/plan['dependency_execution']
    if sha(flat/'registration.json')!=plan['dependency_registration_sha256']:
        raise ValueError('Flat dependency registration changed')
    def status(phase,**kwargs):
        atomic_json(execution/'status.json',dict(phase=phase,pid=os.getpid(),epoch=time.time(),**kwargs))
    atomic_json(execution/'registration.json',dict(pilot_sha256=sha(output/'registration.json'),
        driver_sha256=sha(Path(__file__)),maximum_wait_seconds=plan['maximum_wait_seconds'],
        automatic_retries=0,maximum_native_jobs=4,maximum_simultaneous_servers=2))
    try:
        until=time.monotonic()+plan['maximum_wait_seconds']
        if not (dependency/'complete.json').exists():
            owner=OwnedProcess(**plan['dependency_identity'],allow_terminate=False)
            while not (dependency/'complete.json').exists():
                if (owner.poll() is not None or (dependency/'failure.json').exists()
                        or time.monotonic()>=until):
                    raise RuntimeError('Matched-flat dependency stopped/incomplete or bounded wait expired;no pilot launch')
                status('WAITING_FOR_MATCHED_FLAT',dependency_pid=plan['dependency_identity']['pid'],new_native_jobs=0)
                time.sleep(30)
        for row in dependency_inventory(read(dependency/'complete.json')):
            parent=flat.parent/row['batch'];require_finished_parent(parent)
            if sha(parent/'protocol.json')!=row['protocol_sha256']:raise ValueError('Closed dependency protocol changed')
        atomic_json(execution/'dependency_complete.json',dict(receipt_sha256=sha(dependency/'complete.json'),
            jobs=20,scope='All2120matched native phases closed;2110H cohort may still be live'))
        results=[]
        for row in plan['arms']:
            batch=output.parent/row['batch'];phase=row['arm']+'_TRAIN_AND_KNOWN24'
            status(phase,new_native_jobs_started=len(results)*2)
            env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with (execution/(phase+'.log')).open('xb') as log:
                subprocess.run([sys.executable,ROOT/'scripts/run_fixed_lowlevel_adaptation.py',
                    '--batch',batch,'--execution',batch/'execution'],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=36000)
            complete=read(batch/'execution/complete.json');endpoint=batch.parent/complete['endpoint']
            episodes=read(endpoint/'H/evaluation.json')['episodes']
            if len(episodes)!=24:raise ValueError('Pilot endpoint omitted conditions')
            result=dict(arm=row['arm'],training=batch.name,endpoint=endpoint.name,
                success=sum(e['success'] for e in episodes),
                termination_reasons=dict(Counter(e['termination_reason'] for e in episodes)),
                physics_steps=sum(e['physics_steps'] for e in episodes),
                interventions=sum(e['interventions'] for e in episodes),
                plan_failures=sum(e['plan_failures'] for e in episodes),
                model_sha256=sha(endpoint/'H/model.zip'))
            results.append(result);atomic_json(execution/(row['arm']+'_complete.json'),result)
        atomic_json(execution/'complete.json',dict(results=results,research_acceptance='UNMET',
            scope='One selected exploratory warm-started pair;no independent-seed or final-confirmation claim'))
        status('COMPLETE',new_native_jobs_started=4)
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.validate_only:
        validate(args.output);print('PASS: frozen paired pilot validated;no native launch')
    else:run(args.output)

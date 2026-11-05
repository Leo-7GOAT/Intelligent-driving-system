"""Execute and audit the prepared Grid transfer after the matched native queue.

--validate-only never connects to CARLA or activates a batch. The driving path
requires all20 matched-flat jobs to be closed, preventing resource competition.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from navstack.spatial_planner_transfer import settings
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata

PHASES=('zero_PPO_basic','TRAIN48_PPO30720','basic','known24')


def verify_dependency_inventory(result):
    expected={(seed,phase) for seed in (11,22,33,44,55) for phase in PHASES}
    rows=result.get('results',[])
    if (result.get('seeds')!=[11,22,33,44,55] or len(rows)!=20
            or {(row['seed'],row['phase']) for row in rows}!=expected):
        raise ValueError('All20registered matched-flat jobs must close before Grid driving')
    return rows


def validate(batch):
    batch=Path(batch).resolve();p=read(batch/'protocol.json')
    if read(batch/'prepared.json')['protocol_sha256']!=sha(batch/'protocol.json'):
        raise ValueError('Prepared protocol changed')
    if sha(batch/'calibration/calibration.json')!=p['calibration_sha256']:
        raise ValueError('Frozen physical calibration changed')
    if not settings(p):raise ValueError('Explicit frozen Grid transfer required')
    registration=p.get('shared_grid_transfer',{})
    if (registration.get('label')!='B2' or registration.get('new_PPO_updates')!=0
            or registration.get('new_final_conditions')!=0):raise ValueError('Wrong Grid transfer label or budget')
    name=registration.get('parent','')
    if not name or Path(name).name!=name:raise ValueError('Parent must be a sibling archive batch')
    parent=batch.parent/name;require_finished_parent(parent);original=read(parent/'protocol.json')
    if sha(parent/'protocol.json')!=registration['parent_protocol_sha256']:raise ValueError('Parent protocol changed')
    for key in ('routes','native_case_manifest','native_validation'):
        if p.get(key)!=original.get(key):raise ValueError('Grid transfer changed registered conditions:'+key)
    if len(read(parent/'H/evaluation.json')['episodes'])!=24:raise ValueError('Complete24case parent required')
    for relative,digest in p['source_hashes'].items():
        path=(batch/'source'/relative).resolve()
        if not path.is_relative_to((batch/'source').resolve()) or sha(path)!=digest:raise ValueError('Frozen Grid source changed:'+relative)
    meta=verified_checkpoint_metadata(batch/'resume/H')
    for key,filename in (('model_sha256','model.zip'),('normalization_sha256','normalization.pkl')):
        if meta[key]!=registration[key] or sha(parent/'H'/filename)!=meta[key]:raise ValueError('Frozen Grid actor/normalizer changed')
    if read(batch/'resume/H/config.json')['env']!=read(parent/'H/config.json')['env']:
        raise ValueError('Saved physical/observation/action configuration changed')
    from navstack.config import VehicleConfig
    from scripts.native_road_learning import training_config
    vehicle=VehicleConfig(**read(batch/'calibration/calibration.json')['vehicle'])
    if training_config(p,'H',vehicle).to_dict()!=read(batch/'resume/H/config.json')['env']:
        raise ValueError('Worker runtime configuration differs from the frozen actor')
    return p


def run(batch,execution,after_flat):
    batch,execution,after_flat=map(lambda x:Path(x).resolve(),(batch,execution,after_flat))
    p=validate(batch)
    receipts=[after_flat/name/'complete.json' for name in ('execution','execution_resume_20261105','execution_parallel_20261105')]
    receipts=[path for path in receipts if path.is_file()]
    if len(receipts)!=1:raise ValueError('Exactly one complete matched-flat execution required')
    for row in verify_dependency_inventory(read(receipts[0])):
        parent=after_flat.parent/row['batch'];require_finished_parent(parent)
        if sha(parent/'protocol.json')!=row['protocol_sha256']:raise ValueError('Closed flat dependency changed')
    from scripts.run_matched_flat_cohort import dependency_complete
    flat_plan=read(after_flat/'registration.json');h=Path(flat_plan['h_cohort'])
    if sha(h/'registration.json')!=flat_plan['h_registration_sha256'] or not dependency_complete(h):
        raise ValueError('All25H/H0phases must close before Grid uses the2110span')
    from scripts.run_independent_outcome_cohort import resource_admission
    resources=resource_admission(batch.parent)
    execution.mkdir()
    atomic_json(execution/'registration.json',dict(batch=batch.name,queued_protocol_sha256=sha(batch/'protocol.json'),
        driver_sha256=sha(Path(__file__)),dependency_receipt_sha256=sha(receipts[0]),resource_admission=resources,
        automatic_retries=0,new_PPO_updates=0,new_final_conditions=0))
    def call(args,label,timeout=1800,env=None):
        atomic_json(execution/'status.json',dict(phase=label,pid=os.getpid(),epoch=time.time()))
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,args)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,check=True,timeout=timeout)
    try:
        client=os.environ.get('NAVSTACK_CARLA_PYTHON',str(ROOT/'.carla-python37/runtime/python.exe'))
        call([shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1','-Batch',batch,
            '-Python',sys.executable,'-CarlaPython',client,'-ActivateQueued'],'native',15000)
        require_finished_parent(batch)
        activated=read(batch/'protocol.json')
        if {k:v for k,v in p.items() if k not in ('created_epoch','deadline_epoch')}!={k:v for k,v in activated.items() if k not in ('created_epoch','deadline_epoch')}:
            raise ValueError('Queue activation changed more than the wall window')
        env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
        scripts=batch/'source/scripts'
        call([sys.executable,scripts/'audit_native_method_endpoint.py','--batch',batch,'--method','H'],'raw',env=env)
        for flags,name in (([],'measured_start_inference.json'),(['--all-static-actions'],'static_action_inference.json')):
            call([sys.executable,scripts/'audit_native_reference_inference.py','--batch',batch,'--method','H',*flags,'--output',batch/'H'/name],name,env=env)
        call([sys.executable,scripts/'audit_native_control_delivery.py','--batch',batch,'--output',batch/'strict_delivery.json'],'engine',env=env)
        evaluation=read(batch/'H/evaluation.json');rows=evaluation['episodes'];raw=read(batch/'H/endpoint_raw_audit.json');engine=read(batch/'strict_delivery.json')
        if (len(rows)!=24 or evaluation['model_sha256']!=p['shared_grid_transfer']['model_sha256']
                or not engine['strict_delivery_gate_passed'] or engine['counts']['controls']!=raw['evaluation_control_frames_verified']):
            raise ValueError('Grid native inventory/model/control audit failed')
        atomic_json(execution/'complete.json',dict(label='B2',runtime_checkpoint_schema='H',cases=24,
            successes=sum(row['success'] for row in rows),termination_counts=dict(Counter(row['termination_reason'] for row in rows)),
            actual_controls=raw['evaluation_control_frames_verified'],episodes=rows,new_PPO_updates=0,new_final_conditions=0,
            protocol_sha256=sha(batch/'protocol.json'),raw_audit_sha256=sha(batch/'H/endpoint_raw_audit.json'),
            research_acceptance='UNMET',limits='One frozen-policy known-condition planner deployment transfer;not trained B2 or final confirmation;moving NN sequence beyond starts not reconstructed'))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True)
    parser.add_argument('--execution');parser.add_argument('--after-flat');parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if args.validate_only:
        p=validate(args.batch);print(dict(status='INPUTS_VALIDATED_NO_NATIVE',source_files=len(p['source_hashes'])),flush=True)
    elif not args.execution or not args.after_flat:parser.error('Driving requires --execution and --after-flat')
    else:run(args.batch,args.execution,args.after_flat)

"""Freeze current trained endpoints for bounded cadence/common-brake probes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata

VARIANTS=(('cadence',2),('cadence',5),('shared_brake',False),('shared_brake',True))


def closed_endpoint(parent,method):
    parent=Path(parent).resolve();require_finished_parent(parent)
    p=read(parent/'protocol.json');folder=parent/method
    raw=read(folder/'endpoint_raw_audit.json');evaluation=read(folder/'evaluation.json')
    if (p.get('methods')!=[method] or p.get('evaluation_only') is not True
            or p.get('deployment_shared_brake') or p.get('deployment_control_cadence')
            or raw['protocol_sha256']!=sha(parent/'protocol.json')
            or not read(parent/'strict_delivery.json')['strict_delivery_gate_passed']
            or len(evaluation['episodes'])!=24
            or {row['case_id'] for row in evaluation['episodes']}!=set(range(24))
            or sum(row['physics_steps'] for row in evaluation['episodes'])!=raw['evaluation_control_frames_verified']
            or raw['optimizer_steps_total_lower_bound']<=0):
        raise ValueError('Require an unmodified fully audited trained24condition endpoint')
    if method=='H' and (p.get('reference_context')!={'version':'action_aligned_outcome_context_v1'}
            or p.get('mpc_reference_effort')!={'version':'reference_centered_mpc_effort_v1'}):
        raise ValueError('Require the current signed-context/reference-centered H pipeline')
    for name,key in [('config.json','config_sha256'),('summary.json','summary_sha256'),
                     ('evaluation.json','evaluation_sha256'),('evaluation_trace.jsonl.gz','evaluation_trace_sha256')]:
        if sha(folder/name)!=raw[key]:raise ValueError('Closed endpoint changed:'+name)
    checkpoint=folder/read(folder/'latest_checkpoint.json')['folder'];meta=verified_checkpoint_metadata(checkpoint)
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
        if sha(folder/name)!=meta[key] or evaluation[key]!=meta[key]:raise ValueError('Final checkpoint differs:'+name)
    if meta['lineage_control_steps']<=0:raise ValueError('Untrained initializer cannot be a deployment parent')
    for name,digest in p['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Frozen parent source changed:'+name)
    return p,raw,checkpoint,meta


def variant_label(method,kind,value):
    if method not in ('H','B1'):raise ValueError('Only H and direct B1 are registered')
    if kind=='cadence' and type(value) is int and value in (2,5):return method+'_hold'+str(value)
    if kind=='shared_brake' and type(value) is bool:return method+'_brake_'+('ON' if value else 'OFF')
    raise ValueError('Explicit cadence2/5or brakeOFF/ON required')


def prepare(output,h_parent,flat_parent,pilot,owner_pid):
    from navstack.config import EnvConfig,VehicleConfig
    from navstack.experiments import signature
    from navstack.owned_process import OwnedProcess
    from navstack.control_cadence import configure_cadence
    from scripts.native_road_learning import training_config
    from scripts.prepare_native_control_cadence import (cadence_record,environment_overlay as cadence_env,
        worker_overlay as cadence_worker)
    from scripts.prepare_native_shared_brake import (deployment_record,environment_overlay as brake_env,
        worker_overlay as brake_worker)
    from scripts.prepare_contextual_outcome_pilot import live_owner,available_memory,canonical_digest
    output,pilot=Path(output).resolve(),Path(pilot).resolve()
    labels=[variant_label(method,kind,value) for method in ('H','B1') for kind,value in VARIANTS]
    if output.exists() or any((output.parent/(output.name+'_'+label)).exists() for label in labels):
        raise FileExistsError('Preserve all deployment preparations')
    if output.drive.lower()!='d:' or shutil.disk_usage(output.parent).free<24*1024**3 or available_memory()<4:
        raise ValueError('D archive24GiB disk and4GiB RAM required for CPU preparation')
    identity,command=live_owner(owner_pid)
    if (command.count('--output')!=1 or Path(command[command.index('--output')+1]).resolve()!=pilot
            or not any(Path(arg).name=='run_contextual_outcome_pilot.py' for arg in command)):
        raise ValueError('Live dependency does not identify the registered pilot')
    handle=OwnedProcess(**identity,allow_terminate=False)
    if handle.poll() is not None:raise ValueError('Pilot dependency is not live')
    pp=read(pilot/'registration.json')
    if [row['arm'] for row in pp['arms']]!=['Hc','Hc0']:raise ValueError('Complete paired pilot dependency required')
    parents={'H':Path(h_parent).resolve(),'B1':Path(flat_parent).resolve()}
    originals={method:closed_endpoint(parent,method) for method,parent in parents.items()}
    hp,bp=originals['H'][0],originals['B1'][0]
    if (hp['seed']!=bp['seed'] or hp['seed']!=11
            or hp['native_validation']!=bp['native_validation']
            or hp['calibration_sha256']!=bp['calibration_sha256']):
        raise ValueError('Require the first registered seed11pair,same24validation and calibration')
    for name in ('mpc.py','trajectory.py','vehicle.py','shield.py','world.py','carla_bridge.py'):
        if sha(parents['H']/'source/navstack'/name)!=sha(parents['B1']/'source/navstack'/name):
            raise ValueError('H/B1physical module differs:'+name)
    output.mkdir();rows=[]
    for method,parent in parents.items():
        original,raw,checkpoint,meta=originals[method]
        base_record=read(checkpoint/'config.json')
        if signature(EnvConfig.from_dict(base_record['env']))!=base_record['signature']:
            raise ValueError('Saved parent signature differs from the current compatible config schema')
        if base_record['env']['shared_shield']:raise ValueError('Unwrapped native parent required')
        for kind,value in VARIANTS:
            label=variant_label(method,kind,value);batch=output.parent/(output.name+'_'+label)
            batch.mkdir();(batch/'infrastructure').mkdir()
            shutil.copytree(parent/'source/navstack',batch/'source/navstack',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            shutil.copytree(ROOT/'scripts',batch/'source/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            shutil.copy2(parent/'source/scripts/carla_bridge37.py',batch/'source/scripts/carla_bridge37.py')
            frozen_env=(parent/'source/navstack/env.py').read_text()
            frozen_worker=(parent/'source/scripts/native_road_learning.py').read_text()
            p=deepcopy(original)
            if kind=='cadence':
                record=cadence_record(base_record,method,value)
                (batch/'source/navstack/env.py').write_text(cadence_env(frozen_env),encoding='utf-8')
                (batch/'source/scripts/native_road_learning.py').write_text(cadence_worker(frozen_worker),encoding='utf-8')
                shutil.copy2(ROOT/'navstack/control_cadence.py',batch/'source/navstack/control_cadence.py')
                p['deployment_control_cadence']={**record['deployment_control_cadence'],
                    'baseline_parent':parent.name,'baseline_reused_without_new_driving':True,
                    'reference_centered_effort_retained':method=='H' and original.get('mpc_reference_effort')=={'version':'reference_centered_mpc_effort_v1'}}
                record['deployment_control_cadence']=deepcopy(p['deployment_control_cadence'])
                if method=='B1':p['flat_hold']=value
            else:
                record=deployment_record(base_record,value)
                (batch/'source/navstack/env.py').write_text(brake_env(frozen_env),encoding='utf-8')
                (batch/'source/scripts/native_road_learning.py').write_text(brake_worker(frozen_worker),encoding='utf-8')
                p['deployment_shared_brake']=dict(version='additive_shared_brake_v1',enabled=value,
                    native_reference_safety_gate_retained=True,new_training_controls=0,new_optimizer_updates=0)
            shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(checkpoint,batch/'resume'/method)
            atomic_json(batch/'resume'/method/'config_before_deployment.json',base_record)
            atomic_json(batch/'resume'/method/'config.json',record)
            prior=deepcopy(original['recovery'][method])
            prior.update(checkpoint='resume/'+method,parent_batch=parent.name,parent_checkpoint=checkpoint.name,
                lineage_controls=meta['lineage_control_steps'],retained_optimizer_steps=raw['optimizer_steps_total_lower_bound'],
                consumed_controls_lower_bound=raw['charged_controls_lower_bound']+raw['evaluation_control_frames_verified'],
                unknown_unflushed_parent_tail=raw['unknown_unflushed_ancestor_tail'],
                model_sha256=meta['model_sha256'],normalization_sha256=meta['normalization_sha256'])
            for name in ('fixed_lowlevel_training','freeze_observation_statistics','training_case_schedule','lineage_targets','jobs'):
                p.pop(name,None)
            now=time.time();p.update(methods=[method],evaluation_only=True,controlled_budget_per_method=0,
                method_budgets={method:0},checkpoint_thresholds=[],recovery={method:prior},
                carla_rpc_port=2120,carla_rpc_port_span=[2120,2121,2122],
                created_epoch=now,deadline_epoch=now+14400,
                resource_limits={**p['resource_limits'],'wall_seconds':14400,'automatic_retries':0},
                source_hashes={q.relative_to(batch/'source').as_posix():sha(q) for q in (batch/'source').rglob('*') if q.is_file()},
                current_deployment_control=dict(version='current_native_deployment_controls_v1',seed=11,kind=kind,value=value,
                    parent=parent.name,parent_protocol_sha256=sha(parent/'protocol.json'),
                    parent_raw_audit_sha256=sha(parent/method/'endpoint_raw_audit.json'),new_training_controls=0,new_PPO_updates=0,
                    new_final_conditions=0,known_unique_conditions=24),
                scope='One selected seed11frozen-policy deployment sensitivity;all24known conditions,no retraining or final confirmation')
            vehicle=VehicleConfig(**read(batch/'calibration/calibration.json')['vehicle']);cfg=training_config(p,method,vehicle)
            if kind=='cadence':configure_cadence(cfg,p,method)
            else:cfg.shared_shield=value
            if cfg.to_dict()!=record['env'] or signature(cfg)!=record['signature']:
                raise ValueError('Runtime physical config differs from explicit deployment record')
            verified_checkpoint_metadata(batch/'resume'/method)
            for name in ('model.zip','normalization.pkl','resume_state.pkl'):
                if sha(checkpoint/name)!=sha(batch/'resume'/method/name):raise ValueError('Frozen learning bytes changed')
            allowed={'navstack/env.py'}|({'navstack/control_cadence.py'} if kind=='cadence' else set())
            for name,digest in original['source_hashes'].items():
                if name.startswith('navstack/') and name not in allowed:
                    if sha(batch/'source'/name)!=digest:raise ValueError('Unrelated physical module changed:'+name)
            if sha(batch/'source/scripts/carla_bridge37.py')!=sha(parent/'source/scripts/carla_bridge37.py'):
                raise ValueError('Physical CARLA bridge changed')
            atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',dict(status='QUEUED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
            rows.append(dict(label=label,method=method,kind=kind,value=value,batch=batch.name,
                queued_protocol_sha256=sha(batch/'protocol.json'),model_sha256=meta['model_sha256'],
                normalization_sha256=meta['normalization_sha256'],baseline=parent.name))
    shutil.copytree(ROOT/'scripts',output/'source/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(parents['H']/'source/navstack',output/'source/navstack',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'navstack/control_cadence.py',output/'source/navstack/control_cadence.py')
    plan=dict(version='current_native_deployment_controls_v1',seed=11,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        jobs=rows,driver_source_hashes={q.relative_to(output/'source').as_posix():sha(q) for q in (output/'source').rglob('*') if q.is_file()},
        dependency_cohort=str(pilot),dependency_sha256=sha(pilot/'registration.json'),dependency_identity=identity,
        maximum_wait_seconds=86400,maximum_native_jobs=8,maximum_evaluation_controls=8*24*900,
        maximum_simultaneous_servers=2,maximum_wall_seconds_per_job=14400,automatic_retries=0,
        runtime_carla_python=pp.get('runtime_carla_python') or read(pilot.parent/pp['arms'][0]['batch']/'protocol.json')['contextual_feedback_pilot']['carla_python'],
        validation_tasks_sha256=canonical_digest(hp['native_validation']),new_training_controls=0,new_PPO_updates=0,
        new_final_conditions=0,unique_known_conditions=24,
        scope='One first-registered seed11pair,eight deployment-only probes;not independent five-seed ablations or retrained shared-safety evidence')
    atomic_json(output/'registration.json',plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','h-parent','flat-parent','pilot'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--owner-pid',type=int,required=True)
    args=parser.parse_args();p=prepare(args.output,args.h_parent,args.flat_parent,args.pilot,args.owner_pid)
    print(json.dumps(dict(status='PREPARED_NOT_STARTED',jobs=len(p['jobs']))))

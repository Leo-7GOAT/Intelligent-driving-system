"""Frozen H/H0/teacher diagnostic on all12 previously registered novel static TRAIN cases."""
from copy import deepcopy
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig,VehicleConfig
from navstack.experiments import signature
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare(execution,pair=None,wall_seconds=5400):
    if type(wall_seconds) is not int or not 1800<=wall_seconds<=10800:
        raise ValueError('Frozen diagnostic requires a finite1800..10800second budget')
    teacher=ROOT/'runs/20261102T101413887816Z_native_diverse_TRAIN36_teacher'
    pair=(ROOT/'runs/20261101T011945179519Z_native_matched_broader_feedback' if pair is None else Path(pair).resolve())
    initial=ROOT/'runs/20261101T020243559283Z_native_pristine_H_baseline'
    for parent in (teacher,pair,initial):require_finished_parent(parent)
    p=deepcopy(read(pair/'protocol.json'));reference=read(teacher/'protocol.json')
    calibration=read(teacher/'calibration/calibration.json');oldcal=read(pair/'calibration/calibration.json')
    comparison=deepcopy(calibration);delivery=comparison['recommended_adapter'].pop('control_delivery',None)
    already_common=calibration==oldcal
    if not already_common and (delivery!='batch_sync_v1' or comparison!=oldcal):
        raise ValueError('Only the declared command-delivery transfer is allowed')
    sources=[('H_PPO','H',pair),('H0_PPO','H0',pair),('H_teacher','H',initial)]
    p.update(methods=[name for name,_,_ in sources],jobs=[dict(name=name,method=method) for name,method,_ in sources],
        evaluation_only=True,controlled_budget_per_method=0,method_budgets={name:0 for name,_,_ in sources},
        deterministic_training_route_evaluation=True,checkpoint_thresholds=[],goal_parking_assist=True,
        routes=deepcopy(reference['routes']),native_case_manifest=deepcopy(reference['native_case_manifest']),
        frozen_policy_comparison=dict(version='named_frozen_native_v1',data_role='KNOWN_TRAIN_DIAGNOSTIC',
            source_teacher_batch=teacher.name,original_teacher_case_ids=list(range(12,24)),
            original_B3_successes=7,original_B3_cases=12,original_B3_raw_audit_sha256=sha(teacher/'raw_audit.json'),
            new_PPO_updates=0,new_final_conditions=0,
            limits='One canonical seed,all12registered static TRAINcases including all failures. Not independent final confirmation or new training. Every policy uses the same synchronous delivery and common parking aid;already-adapted policies retain their training lowlevel,others are explicitly frozen transfers. No gain/geometry changes.'))
    p['native_case_manifest']['cases']=deepcopy(reference['native_case_manifest']['cases'][12:24])
    for name in ('native_validation','freeze_observation_statistics','own_success_replay','native_recovery_collection',
                 'gradient_clipping_experiment','rollout_length_experiment','flat_distribution_experiment','actor_head_experiment'):
        p.pop(name,None)
    vehicle=VehicleConfig(**calibration['vehicle']);common=EnvConfig.from_dict(read(teacher/'source_config.json')['env'])
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<16*1024**3:raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_static_H_H0_teacher_transfer')
    batch.mkdir();(batch/'infrastructure').mkdir();(batch/'calibration').mkdir();(batch/'transfer_evidence').mkdir()
    shutil.copy2(teacher/'calibration/calibration.json',batch/'calibration/calibration.json')
    hashes={}
    for folder in ('navstack','scripts'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.ps1'):
                relative=path.relative_to(ROOT);destination=batch/'source'/relative
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination);hashes[relative.as_posix()]=sha(destination)
    p['source_hashes']=hashes;p['calibration_sha256']=sha(batch/'calibration/calibration.json');p['recovery']={};proofs=[]
    for name,method,parent in sources:
        folder=parent/method;origin=folder/read(folder/'latest_checkpoint.json')['folder'];metadata=verified_checkpoint_metadata(origin)
        before=EnvConfig.from_dict(read(origin/'config.json')['env'])
        converted=EnvConfig.from_dict({**before.to_dict(),'goal_parking_assist':True})
        cfg=training_config(p,method,vehicle)
        if converted.to_dict()!=common.to_dict() or signature(cfg)!=signature(converted):
            raise ValueError('Common physical/interface config differs beyond declared parking transfer:'+name)
        destination=batch/'resume'/name;shutil.copytree(origin,destination)
        shutil.copy2(origin/'config.json',batch/'transfer_evidence'/(name+'_original_config.json'))
        atomic_json(destination/'config.json',dict(env=cfg.to_dict(),signature=signature(cfg),seed=22,backend='CARLA_0.9.15',
            explicit_transfer='Frozen neural policy;common synchronous command delivery and parking aid. Already adapted policies retain their training lowlevel. Original learned tensors/Adam/RMS/RNG files retained.'))
        for filename in ('model.zip','normalization.pkl','resume_state.pkl'):
            if (origin/filename).exists() and sha(origin/filename)!=sha(destination/filename):raise ValueError('Transfer changed saved learning bytes')
        previous=read(parent/'protocol.json').get('recovery',{}).get(method,{})
        p['recovery'][name]=dict(checkpoint='resume/'+name,parent_batch=parent.name,parent_checkpoint=method+'/'+origin.name,
            lineage_controls=metadata.get('lineage_control_steps',0),
            consumed_controls_lower_bound=metadata.get('charged_control_steps_lower_bound',0),
            retained_optimizer_steps=previous.get('retained_optimizer_steps',0)+metadata.get('optimizer_steps',0),
            model_sha256=metadata['model_sha256'],normalization_sha256=metadata['normalization_sha256'])
        proofs.append(dict(name=name,method=method,parent=parent.name,model_sha256=metadata['model_sha256'],
            normalization_sha256=metadata['normalization_sha256'],source_config_sha256=sha(origin/'config.json'),
            transferred_config_sha256=sha(destination/'config.json'),all_saved_learning_bytes_retained=True,
            parking_assist_changed=before.goal_parking_assist!=cfg.goal_parking_assist,
            original_calibration_sha256=sha(parent/'calibration/calibration.json')))
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+wall_seconds,
        resource_limits={**p['resource_limits'],'wall_seconds':wall_seconds,'automatic_retries':0},
        scope='All12novel static TRAINcases;fixed canonical H/H0/teacher policies under common corrected delivery/parking. Already adapted policies retain their training lowlevel. Zero newPPO,not final or five-seed evidence.')
    atomic_json(batch/'transfer_evidence/registration.json',dict(models=proofs,original_calibration_sha256=sha(pair/'calibration/calibration.json'),
        common_calibration_sha256=p['calibration_sha256'],pair_already_uses_common_calibration=already_common,
        only_adapter_change='none_for_adapted_pair;legacy_teacher_transfer_to_batch_sync_v1' if already_common else 'control_delivery:legacy_actor_async_to_batch_sync_v1',
        geometry_model_case14_disconnection_retained=True,task_selection='Every originally registered static case;no exclusions'))
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
    atomic_json(execution/'registration.json',dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json'),models=proofs))
    return batch


def run(execution=None,pair=None,wall_seconds=5400):
    execution=(ROOT/'runs/H_H0_teacher_static12_transfer_execution_20261102' if execution is None else Path(execution).resolve())
    if execution.exists():raise FileExistsError('Preserve every frozen comparison')
    execution.mkdir();batch=prepare(execution,pair,wall_seconds)
    def call(args,label,timeout=900,env=None):
        atomic_json(execution/'status.json',dict(phase=label,pid=os.getpid(),batch=batch.name,epoch=time.time()))
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,args)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,env=env,check=True)
    try:
        call([shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1','-Batch',batch,'-UseCurrentGuard'],'frozen_native',wall_seconds+600)
        require_finished_parent(batch);env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
        results={}
        for name in ('H_PPO','H0_PPO','H_teacher'):
            call([sys.executable,ROOT/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method',name],name+'_raw',env=env)
            call([sys.executable,ROOT/'scripts/audit_native_policy_inference.py','--batch',batch,'--method',name,
                '--output',batch/name/'measured_start_inference.json'],name+'_starts',env=env)
            episodes=read(batch/name/'evaluation.json')['episodes']
            results[name]=dict(successes=sum(r['success'] for r in episodes),cases=len(episodes),new_PPO_updates=0)
        call([sys.executable,ROOT/'scripts/audit_native_control_delivery.py','--batch',batch,'--output',batch/'strict_delivery.json'],'strict_delivery')
        if not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']:raise ValueError('Strict engine control delivery failed')
        atomic_json(execution/'complete.json',dict(batch=batch.name,results=results,research_acceptance='UNMET',scope='One-seed knownTRAINtransfer diagnostic only'))
        atomic_json(execution/'status.json',dict(phase='FROZEN_STATIC_CONTRAST_COMPLETE',batch=batch.name,results=results))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution');parser.add_argument('--pair')
    parser.add_argument('--wall-seconds',type=int,default=5400)
    args=parser.parse_args();run(args.execution,args.pair,args.wall_seconds)

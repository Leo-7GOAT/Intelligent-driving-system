"""Explicit frozen-policy engineering transfer to a method-independent parking aid."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig,VehicleConfig
from navstack.experiments import signature
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_control_delivery_probe import overlays,replace_once
from scripts.native_road_learning import training_config


def prepare(method):
    if method not in ('B1','H'):raise ValueError('Explicit engineering transfer supports B1 or H')
    pointer=ROOT/'runs'/('native_shared_parking_v2_'+method+'_registration_20261101.json')
    if pointer.exists():raise FileExistsError('Preserve registered shared-wrapper transfer')
    b=ROOT/'runs/20261101T100704463570Z_native_reflection_feedback_probe'
    parent=b if method=='B1' else ROOT/'runs/20261101T011945179519Z_native_matched_broader_feedback'
    require_finished_parent(parent);require_finished_parent(b)
    old=read(parent/'protocol.json');bp=read(b/'protocol.json')
    origin=(ROOT/'runs/B1_seed22_reflection_feedback_extension_20261101/checkpoint' if method=='B1'
        else parent/'H'/read(parent/'H/latest_checkpoint.json')['folder'])
    metadata=verified_checkpoint_metadata(origin);saved=read(origin/'config.json')
    p=deepcopy(old)
    for key in ('native_reflection_feedback','native_validation','freeze_observation_statistics','jobs','own_success_replay','native_camera_evaluation'):
        p.pop(key,None)
    cal=read(b/'calibration/calibration.json');oldcal=read(parent/'calibration/calibration.json')
    plain=deepcopy(cal);plain['recommended_adapter'].pop('control_delivery',None)
    oldplain=deepcopy(oldcal);oldplain['recommended_adapter'].pop('control_delivery',None)
    if plain!=oldplain:raise ValueError('Shared parking transfer must not change vehicle gains or geometry')
    now=time.time();p.update(methods=[method],evaluation_only=True,controlled_budget_per_method=0,method_budgets={method:0},checkpoint_thresholds=[],
        goal_parking_assist=True,deterministic_training_route_evaluation=True,routes=deepcopy(bp['routes']),native_case_manifest=deepcopy(bp['native_case_manifest']),
        created_epoch=now,deadline_epoch=now+1800,resource_limits={**old['resource_limits'],'wall_seconds':1800,'automatic_retries':0},
        scope='Frozen-policy shared parking-assist engineering transfer,all4original TRAINroads. Physical full braking only after reaching unchanged goal position/yaw gates. No goal/state changes,new PPOlearning or native-unassisted qualification.',
        shared_parking_transfer={'version':'inside_goal_pose_overspeed_brake_v2','method':method,'same_capability_for':['H','H0','B1','B2','B3'],
            'source_model_sha256':metadata['model_sha256'],'normalization_sha256':metadata['normalization_sha256'],
            'new_PPO_updates':0,'new_supervised_updates':0,'steering_override':False,
            'limits':'Deploy-time common aid,not trained under this wrapper. Must retrain related methods before a shared-wrapper training comparison. Original unassisted failures retained.',
            'source_delivery':oldcal['recommended_adapter'].get('control_delivery','actor_async_legacy'),'target_delivery':'batch_sync_v1'})
    if method=='B1':p['flat_hold']=1
    config=training_config(p,method,VehicleConfig(**cal['vehicle']))
    unassisted=EnvConfig.from_dict({**config.to_dict(),'goal_parking_assist':False})
    if signature(unassisted)!=saved['signature']:
        raise ValueError('Unrelated saved-policy semantics differ from the proposed wrapper transfer')
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_shared_parking_v2_'+method)
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(b/'calibration',batch/'calibration');shutil.copytree(origin,batch/'resume'/method)
    for name in ('navstack/env.py','navstack/config.py','navstack/experiments.py','navstack/parking_assist.py'):
        shutil.copy2(ROOT/name,batch/'source'/name)
    # Preserve each parent's worker implementation; add only the common flag.
    worker='scripts/native_road_learning.py';text=(parent/'source'/worker).read_text(encoding='utf-8')
    marker='    config = EnvConfig(method="H" if method == "H0" else method, vehicle=vehicle, road_observation=True,'
    text=replace_once(text,marker,marker+"\n        goal_parking_assist=protocol.get('goal_parking_assist',False),")
    (batch/'source'/worker).write_text(text,encoding='utf-8')
    changes=['navstack/env.py','navstack/config.py','navstack/experiments.py','navstack/parking_assist.py',worker]
    if method=='H':
        for name,text in overlays(parent).items():
            (batch/'source'/name).write_text(text,encoding='utf-8');changes.append(name)
    shutil.copy2(__file__,batch/'source/scripts/prepare_native_shared_parking.py');changes.append('scripts/prepare_native_shared_parking.py')
    if (parent/'measured_demonstrations.npz').exists():shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    destination=batch/'resume'/method;shutil.copy2(destination/'config.json',destination/'config_before_shared_parking.json')
    atomic_json(destination/'config.json',{**saved,'env':config.to_dict(),'signature':signature(config),
        'explicit_deployment_intervention':'inside_goal_pose_overspeed_brake_v2;weights,optimizer,RMS and RNG bytes unchanged. Not training under the new wrapper.'})
    prior_raw=read(parent/method/'endpoint_raw_audit.json');summary=read(parent/method/'summary.json')
    inherited=(old.get('recovery',{}).get(method,{}) or {})
    p['recovery']={method:{'checkpoint':'resume/'+method,'parent_batch':parent.name,'parent_checkpoint':str(origin.relative_to(ROOT/'runs')),
        'lineage_controls':summary['lineage_control_steps'],
        'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound']+prior_raw['evaluation_control_frames_verified'],
        'retained_optimizer_steps':inherited.get('retained_optimizer_steps',0)+summary['optimizer_steps'],
        'model_sha256':metadata['model_sha256'],'normalization_sha256':metadata['normalization_sha256']}}
    p['calibration_sha256']=sha(batch/'calibration/calibration.json')
    p['shared_parking_transfer'].update(checkpoint_config_before_sha256=sha(destination/'config_before_shared_parking.json'),
        checkpoint_config_after_sha256=sha(destination/'config.json'),declared_source_overlays=changes)
    for name in changes:p['source_hashes'][name]=sha(batch/'source'/name)
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
    atomic_json(pointer,{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')});print(batch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['B1','H'],required=True);a=p.parse_args();prepare(a.method)

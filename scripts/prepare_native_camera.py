"""Stage a frozen native PPO camera evaluation; never change or start the parent."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.native_road_learning import camera_evaluation_mode, training_config
from navstack.config import VehicleConfig
from navstack.experiments import signature


def selected_manifest(manifest, ids):
    if (not isinstance(ids,list) or not 1<=len(ids)<=4 or any(type(i) is not int for i in ids)
            or len(set(ids))!=len(ids) or any(i<0 or i>=len(manifest['cases']) for i in ids)):
        raise ValueError('Select one to four unique existing development case IDs')
    if manifest.get('split')!='development':
        raise ValueError('Camera illustrations must not activate final confirmation conditions')
    return {**deepcopy(manifest),'cases':[deepcopy(manifest['cases'][i]) for i in ids]}


def prepare(parent,method,case_ids):
    parent=Path(parent).resolve();require_finished_parent(parent)
    old=read(parent/'protocol.json');folder=parent/method
    if method not in ('H','H0','B1') or method not in old['methods']:
        raise ValueError('Require an existing native trained policy')
    manifest=selected_manifest(old['native_case_manifest'],case_ids)
    audit=read(folder/'endpoint_raw_audit.json');evaluation=read(folder/'evaluation.json')
    for name,key in (('config.json','config_sha256'),('evaluation.json','evaluation_sha256'),('summary.json','summary_sha256')):
        if sha(folder/name)!=audit[key]:raise ValueError('Parent endpoint changed after raw audit')
    if sha(parent/'protocol.json')!=audit['protocol_sha256']:raise ValueError('Parent protocol changed')
    summary=read(folder/'summary.json')
    if summary['optimizer_steps']<=0 or summary['stopped_reason'] is not None:
        raise ValueError('Require completed actual PPOtraining,not a rule or untrained actor')
    latest=read(folder/'latest_checkpoint.json')['folder']
    if Path(latest).name!=latest:raise ValueError('Checkpoint must be a direct child')
    origin=folder/latest;metadata=verified_checkpoint_metadata(origin)
    for name,key in (('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')):
        if not sha(origin/name)==sha(folder/name)==evaluation[key]:
            raise ValueError('Latest retained checkpoint is not the evaluated final policy')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed')
    if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:
        raise ValueError('Parent physical calibration changed')
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<13*1024**3:raise RuntimeError('Recording resource guard:'+str(reason))
    p=deepcopy(old)
    for key in ('jobs','actor_head_experiment','gradient_clipping_experiment','rollout_length_experiment',
                'fresh_cadence_experiment','freeze_observation_statistics','own_success_replay','native_validation'):
        p.pop(key,None)
    p.update(scope='Illustrative actual RGB evaluation of a trained native PPOpolicy on explicitly selected known DEVELOPMENT cases;zero updates,not final confirmation or model selection',
        methods=[method],evaluation_only=True,controlled_budget_per_method=0,method_budgets={method:0},
        checkpoint_thresholds=[],deterministic_training_route_evaluation=True,native_case_manifest=manifest,
        native_camera_evaluation={'version':'frozen_native_RGB_v1','original_case_ids':case_ids,
            'model_sha256':evaluation['model_sha256'],'normalization_sha256':evaluation['normalization_sha256'],
            'parent_batch':parent.name,'parent_raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
            'selection':'Prespecified illustrative cases from known results;all new outcomes retained,no success/failure matching enforced',
            'rendering':'Actual640x360RGB at20Hz;unchanged physical and policy code;rendering enabled with frame fences,not a no-camera timing benchmark'},
        recovery={method:{'checkpoint':'resume/'+method,'parent_batch':parent.name,'parent_checkpoint':latest,
            'lineage_controls':summary['lineage_control_steps'],
            'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound']+audit['evaluation_control_frames_verified'],
            'retained_optimizer_steps':old.get('recovery',{}).get(method,{}).get('retained_optimizer_steps',0)+summary['optimizer_steps']}},
        resource_limits={**old['resource_limits'],'wall_seconds':1800,'automatic_retries':0})
    camera_evaluation_mode(p)
    vehicle=VehicleConfig(**read(parent/'calibration/calibration.json')['vehicle'])
    if signature(training_config(p,method,vehicle))!=read(origin/'config.json')['signature']:
        raise ValueError('Camera evaluation changes saved policy/task semantics')
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_PPO_camera')
    batch.mkdir(parents=True);(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',batch/'calibration')
    (batch/'resume').mkdir();shutil.copytree(origin,batch/'resume'/method)
    if verified_checkpoint_metadata(batch/'resume'/method)!=metadata:raise ValueError('Staged checkpoint changed')
    for name in ('native_road_learning.py','prepare_native_camera.py','render_carla_recordings.py'):
        shutil.copy2(ROOT/'scripts'/name,batch/'source/scripts'/name)
    write_json(batch/'parent_protocol.json',old)
    write_json(batch/'parent_raw_audit.json',audit)
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+1800)
    p['source_hashes']={path.relative_to(batch/'source').as_posix():sha(path)
        for path in (batch/'source').rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    write_json(batch/'protocol.json',p)
    write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
    print(batch,flush=True)
    return batch


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True);parser.add_argument('--method',required=True)
    parser.add_argument('--case-ids',required=True,type=int,nargs='+')
    args=parser.parse_args();prepare(args.parent,args.method,args.case_ids)

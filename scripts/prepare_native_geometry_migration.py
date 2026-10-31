"""Stage one complete endpoint on explicit v4 geometry; no policy updates."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import write_json, signature
from navstack.native_route import REFINED_GRID_VERSION
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import audit,read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config
from scripts.prepare_native_evaluation_recovery import evaluation_lineage
from scripts.prepare_native_recovery import verified_checkpoint_metadata


OVERRIDES=('navstack/config.py','navstack/native_route.py','navstack/road_grid.py',
    'navstack/road_grid_batch.py','navstack/grid_smoothing.py','navstack/planner_memo.py',
    'scripts/carla_bridge37.py','scripts/prepare_native_geometry_migration.py')


def migrated_config(saved,protocol,vehicle,method):
    original=EnvConfig.from_dict(saved['env'])
    if saved['signature']!=signature(original):raise ValueError('Parent configuration signature changed')
    changed=training_config(protocol,method,vehicle)
    expected=original.to_dict();expected['road_geometry_version']=REFINED_GRID_VERSION
    if changed.to_dict()!=expected:raise ValueError('Migration changes fields beyond road geometry')
    return {**saved,'env':changed.to_dict(),'signature':signature(changed)}


def prepare(parent,method='B1'):
    parent=Path(parent).resolve();require_finished_parent(parent)
    old=read(parent/'protocol.json')
    if method not in old['methods'] or old.get('native_validation'):
        raise ValueError('Require one existing complete known-condition development endpoint')
    if old['road_geometry_version']!='carla_driving_indexed_union_64_v3':
        raise ValueError('Migration source must be registered v3 geometry')
    parent_audit=audit(parent,method)
    if parent_audit!=read(parent/method/'endpoint_raw_audit.json'):raise ValueError('Parent raw evidence differs')
    folder=parent/method;summary=read(folder/'summary.json');name=read(folder/'latest_checkpoint.json')['folder']
    if Path(name).name!=name or summary['stopped_reason'] is not None or not summary['reload_exact']:
        raise ValueError('Require complete reload-exact final endpoint')
    origin=folder/name;metadata=verified_checkpoint_metadata(origin)
    if any(metadata[k]!=summary[k] for k in ('physics_steps','actions','optimizer_steps')):
        raise ValueError('Checkpoint is not the final endpoint')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed')
    if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:raise ValueError('Calibration changed')
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    with ProcessLease(parent/'geometry_migration_prepare.lock'):
        registration=parent/(method+'_geometry_v4_migration_registration.json')
        if registration.exists():raise FileExistsError('Preserve registered geometry migration;no retries')
        protocol=deepcopy(old);now=time.time()
        for key in ('jobs','freeze_observation_statistics','own_success_replay'):
            protocol.pop(key,None)
        protocol.update(road_geometry_version=REFINED_GRID_VERSION,evaluation_only=True,
            methods=[method],controlled_budget_per_method=0,method_budgets={method:0},checkpoint_thresholds=[],
            created_epoch=now,deadline_epoch=now+3600,parent_batch=parent.name,
            resource_limits={**old['resource_limits'],'wall_seconds':3600},
            scope='Frozen checkpoint on new256collision geometry;zero PPO updates;known development conditions;deployment migration only,not trained-v4 or final confirmation',
            geometry_migration={'from':old['road_geometry_version'],'to':REFINED_GRID_VERSION,
                'policy_observation':'same64x64 samples and network shape for all methods',
                'collision_geometry':'256x256 full-body occupied-cell SAT;shared planning,risk,bounds;no lane dilation',
                'limitations':'Conservative relative to sampled points,not continuous mesh proof;old outcomes retained;PPO contribution not inferred'},
            recovery={method:evaluation_lineage(old.get('recovery',{}).get(method,{}),summary,
                parent.name,origin.name,method,parent_audit['evaluation_control_frames_verified'])})
        from navstack.config import VehicleConfig
        vehicle=VehicleConfig(**read(parent/'calibration/calibration.json')['vehicle'])
        changed=migrated_config(read(origin/'config.json'),protocol,vehicle,method)
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_geometry_migration')
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent/'calibration',batch/'calibration')
        destination=batch/'resume'/method;shutil.copytree(origin,destination)
        write_json(destination/'config.json',changed)
        if verified_checkpoint_metadata(destination)!=metadata:raise ValueError('Migration changed checkpoint bytes')
        for relative in OVERRIDES:shutil.copy2(ROOT/relative,batch/'source'/relative)
        if old.get('extra_supervision'):
            dataset=parent/'measured_demonstrations.npz'
            if sha(dataset)!=old['extra_supervision']['dataset_sha256']:raise ValueError('Teacher provenance changed')
            shutil.copy2(dataset,batch/dataset.name)
        write_json(batch/'parent_protocol.json',old);write_json(batch/'parent_endpoint_raw_audit.json',parent_audit)
        write_json(batch/'parent_training_summary.json',summary)
        protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p)
            for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        protocol['geometry_migration']['source_overrides']={name:protocol['source_hashes'][name] for name in OVERRIDES}
        write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        write_json(registration,{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')})
        print(batch,flush=True);return batch


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--method',default='B1')
    a=p.parse_args();prepare(a.parent,a.method)

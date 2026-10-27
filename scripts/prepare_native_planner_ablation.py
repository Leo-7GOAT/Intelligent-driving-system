"""Frozen trained H with the existing Grid A* backend; not a retrained B2 claim."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time
import ast

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import signature,write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import audit,read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_supervised_ppo import copy_cleanup_harness


def planner_replacement_record(record,grid_postprocess='legacy'):
    if grid_postprocess not in ('legacy','body_spline_v1'):raise ValueError('Unknown Grid A* postprocessing version')
    original=EnvConfig.from_dict(record['env'])
    if original.method!='H':raise ValueError('Frozen planner replacement requires the hierarchical H action interface')
    env=EnvConfig.from_dict({**original.to_dict(),'method':'B2'})
    before,after=original.to_dict(),env.to_dict()
    if {k:(before.get(k),after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)}!={'method':('H','B2')}:
        raise ValueError('Planner replacement changes unrelated environment settings')
    env.planner.grid_postprocess=grid_postprocess
    return {**record,'env':env.to_dict(),'signature':signature(env),
            'scope':'Same frozen H actor with existing2DGridAStarPlanner and postprocessing; zero retraining. Deployment ablation,not a from-scratch B2result.'}


def copy_planning_config_harness(archived,current,destination):
    def stripped(path):
        tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='training_config']
        return ast.dump(tree,include_attributes=False)
    if stripped(archived)!=stripped(current):raise ValueError('Planning configuration override changed other runtime code')
    shutil.copy2(current,destination)
    return {'parent_sha256':sha(archived),'initialized_sha256':sha(current),
            'scope':'training_config only; explicit Grid A* postprocessor, no plant/training-loop/evaluation code change'}


def prepare(parent,wall_seconds=1800,grid_postprocess='legacy'):
    parent=Path(parent).resolve();require_finished_parent(parent)
    if type(wall_seconds) is not int or not 900<=wall_seconds<=3600:raise ValueError('Invalid bounded ablation window')
    protocol=read(parent/'protocol.json');method='H'
    if method not in protocol.get('methods',[]) or protocol.get('evaluation_only'):
        raise ValueError('Require an actually trained H parent')
    verified=audit(parent,method)
    if verified!=read(parent/method/'endpoint_raw_audit.json'):raise ValueError('Trained parent raw audit changed')
    summary=read(parent/method/'summary.json')
    origin=parent/method/read(parent/method/'latest_checkpoint.json')['folder']
    metadata=verified_checkpoint_metadata(origin)
    if (not summary['reload_exact'] or summary['stopped_reason'] is not None
            or summary['optimizer_steps']<=0 or summary['physics_steps']<=0
            or any(metadata[k]!=summary[k] for k in ('physics_steps','actions','optimizer_steps'))
            or metadata['model_sha256']!=sha(parent/method/'model.zip')):
        raise ValueError('Require the final complete trained checkpoint')
    for name,digest in protocol['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent frozen source changed')
    record=planner_replacement_record(read(origin/'config.json'),grid_postprocess)
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    with ProcessLease(parent/'frozen_B2_prepare.lock'):
        registration=parent/('frozen_H_B2_grid_v1'+('' if grid_postprocess=='legacy' else '_'+grid_postprocess)+'_registration.json')
        if registration.exists():raise FileExistsError('Preserve the registered planner ablation')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_frozen_B2')
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent/'calibration',batch/'calibration')
        destination=batch/'resume/B2';destination.parent.mkdir();shutil.copytree(origin,destination)
        write_json(destination/'config.json',record)
        if sha(destination/'model.zip')!=metadata['model_sha256'] or sha(destination/'normalization.pkl')!=metadata['normalization_sha256']:
            raise ValueError('Frozen actor/normalizer bytes changed')
        verified_checkpoint_metadata(destination)
        cleanup=copy_cleanup_harness(parent/'source/scripts/carla_bridge37.py',ROOT/'scripts/carla_bridge37.py',batch/'source/scripts/carla_bridge37.py')
        config_harness=None
        if grid_postprocess!='legacy':
            config_harness=copy_planning_config_harness(parent/'source/scripts/native_road_learning.py',ROOT/'scripts/native_road_learning.py',batch/'source/scripts/native_road_learning.py')
        shutil.copy2(Path(__file__),batch/'source/scripts'/Path(__file__).name)
        dataset=parent/'measured_demonstrations.npz'
        if protocol.get('extra_supervision'):
            if not dataset.is_file() or sha(dataset)!=protocol['extra_supervision']['dataset_sha256']:raise ValueError('Parent teacher dataset changed')
            shutil.copy2(dataset,batch/'measured_demonstrations.npz')
        write_json(batch/'parent_H_training_summary.json',summary);write_json(batch/'parent_protocol.json',protocol)
        now=time.time();old=deepcopy(protocol)
        for key in ('jobs','own_success_replay','freeze_observation_statistics','supervised_ppo_intervention','supervised_representation_gates','initialization_gate_profile'):
            protocol.pop(key,None)
        protocol.update(methods=['B2'],evaluation_only=True,controlled_budget_per_method=0,method_budgets={'B2':0},
            checkpoint_thresholds=[],created_epoch=now,deadline_epoch=now+wall_seconds,parent_batch=parent.name,
            scope='Frozen H-to-B2planning deployment ablation on all registered development conditions; no new training,not independent confirmation or retrained B2performance',
            planner_ablation={'version':'frozen_H_grid_astar_v1','parent_method':'H','runtime_method':'B2',
                'environment_changes':{'method':['H','B2'],**({'planner.grid_postprocess':['legacy',grid_postprocess]} if grid_postprocess!='legacy' else {})},'model_sha256':metadata['model_sha256'],
                'grid_postprocess':grid_postprocess,'configuration_harness':config_harness,
                'normalization_sha256':metadata['normalization_sha256'],'cleanup_harness':cleanup,
                'scope':'Existing2DGridAStarPlanner with its existing geometric smoothing/feasibility checks. Same H weights,observations,trajectory,MPC,vehicle,reward and scene list; feedback outcomes naturally change with planner execution.'},
            recovery={'B2':{'checkpoint':'resume/B2','parent_batch':parent.name,'parent_checkpoint':origin.name,
                'lineage_controls':summary['lineage_control_steps'],
                'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound']+verified['evaluation_control_frames_verified'],
                'retained_optimizer_steps':old.get('recovery',{}).get('H',{}).get('retained_optimizer_steps',0)+summary['optimizer_steps'],
                'unknown_unflushed_parent_tail':False}},resource_limits={**old['resource_limits'],'wall_seconds':wall_seconds})
        if grid_postprocess!='legacy':protocol['grid_postprocess']=grid_postprocess
        generated=training_config(protocol,'B2',EnvConfig.from_dict(record['env']).vehicle)
        if generated.to_dict()!=record['env']:raise ValueError('Runtime B2config differs from the registered replacement')
        protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        write_json(batch/'protocol.json',protocol);write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        write_json(registration,{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')})
        print(batch,flush=True);return batch


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--wall-seconds',type=int,default=1800)
    p.add_argument('--grid-postprocess',choices=['legacy','body_spline_v1'],default='legacy')
    a=p.parse_args();prepare(a.parent,a.wall_seconds,a.grid_postprocess)

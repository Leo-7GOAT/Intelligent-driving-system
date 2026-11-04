"""Register5120 actual PPOcontrols after the prespecified frozen-option pilot."""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.fixed_lowlevel_training import REGISTRATION,REGISTRATION_POINTS,registered
from navstack.reference_option_commitment import settings as option_settings
from navstack.reference_road_refinement import settings as road_settings
from navstack.steering_transition import settings as steering_settings
from navstack.config import VehicleConfig
from navstack.experiments import signature
from navstack.research import atomic_json
from scripts.native_road_learning import training_config
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def without_settings(path):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='settings']
    if path.name=='fixed_lowlevel_training.py':
        tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='registered')
            and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('VERSION_POINTS','REGISTRATION_POINTS') for t in n.targets))]
    return ast.dump(tree,include_attributes=False)


def prepare(parent,output):
    parent,output=Path(parent).resolve(),Path(output).resolve();require_finished_parent(parent)
    old=read(parent/'protocol.json');raw=read(parent/'H/endpoint_raw_audit.json')
    point_course=old.get('exact_point_pilot_registration',{}).get('original_case_ids')==[21,27]
    ids=[21,27] if point_course else [0,1,4]
    if (old['methods']!=['H'] or not old.get('evaluation_only')
            or not point_course and old.get('refinement_transfer_registration',{}).get('original_case_ids')!=[0,1,4]
            or raw['protocol_sha256']!=sha(parent/'protocol.json')
            or not read(parent/'strict_delivery.json')['strict_delivery_gate_passed']
            or len(read(parent/'H/evaluation.json')['episodes'])!=len(ids)):
        raise ValueError('Require complete audited prespecified frozen-option pilot,regardless of its outcomes')
    road_settings(old);steering_settings(old)
    if option_settings(old) is None:raise ValueError('Require the same explicit repeated-option interface')
    registration_only={'navstack/reference_road_refinement.py','navstack/steering_transition.py','navstack/reference_option_commitment.py',
        'navstack/steering_retiming.py','navstack/fixed_lowlevel_training.py'}
    for name,digest in old['source_hashes'].items():
        frozen=parent/'source'/name
        if sha(frozen)!=digest:raise ValueError('Pilot source changed:'+name)
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:
            if name not in registration_only or without_settings(ROOT/name)!=without_settings(frozen):
                raise ValueError('Lowlevel must remain frozen;only registration gates may change:'+name)
    if output.exists():raise FileExistsError('Preserve each registered adaptation attempt')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.drive.lower()!='d:' or shutil.disk_usage(output.parent).free<16*1024**3:
        raise RuntimeError('Use D staging with16GiB headroom')
    folder=parent/'H';checkpoint=folder/read(folder/'latest_checkpoint.json')['folder'];meta=verified_checkpoint_metadata(checkpoint)
    for name in ['model.zip','normalization.pkl']:
        if sha(checkpoint/name)!=sha(folder/name):raise ValueError('Pilot initializer differs from its evaluated bytes')
    p=deepcopy(old);p.pop('reference_context_endpoint',None);p.pop('refinement_transfer_registration',None)
    p.pop('exact_point_pilot_registration',None);p.pop('exact_compute_recovery',None)
    p.pop('own_success_replay',None)
    p.update(evaluation_only=False,method_budgets={'H':5120},controlled_budget_per_method=5120,
        deterministic_training_route_evaluation=False,freeze_observation_statistics=True,record_initial_parameters=True,
        fixed_lowlevel_training=deepcopy(REGISTRATION_POINTS if point_course else REGISTRATION),checkpoint_thresholds=list(range(320,5121,320)))
    p['reference_learning_registration'].update(requested_controls_each=5120,physical_models_and_actions_unchanged=False,
        physical_actuation_unchanged=True,planner_and_option_semantics_changed=True,lowlevel_fixed_for_this_course=True,
        initializer_proof_sha256=sha(checkpoint/'metadata.json'),selection='Final fixed requested budget endpoint only')
    prior=old['recovery']['H']
    p['recovery']={'H':dict(checkpoint='resume/H',parent_batch=parent.name,parent_checkpoint='H/'+checkpoint.name,
        lineage_controls=meta['lineage_control_steps'],consumed_controls_lower_bound=meta['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
        retained_optimizer_steps=prior['retained_optimizer_steps']+meta['optimizer_steps'],unknown_unflushed_parent_tail=False,
        model_sha256=meta['model_sha256'],normalization_sha256=meta['normalization_sha256'])}
    p['lineage_targets']={'H':meta['lineage_control_steps']+5120}
    p['fixed_lowlevel_adaptation_registration']=dict(parent_pilot=parent.name,parent_protocol_sha256=sha(parent/'protocol.json'),
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        parent_raw_audit_sha256=sha(parent/'H/endpoint_raw_audit.json'),original_known_development_case_ids=ids,
        fixed_requested_controls=5120,maximum_rollout_overshoot=319,new_supervision=0,new_final_conditions=0,
        selection='All prespecified cases retained,regardless of pilot outcome;no seed/model selection',
        limits='One continued policy and known development conditions only;new lowlevel/action semantics,not a matched from-scratch comparison,independent seed or final acceptance.')
    p['scope']='Single continued native PPOpolicy adapting to fixed32000-node slew planning and explicit repeated-option persistence;known development only'
    output.mkdir();(output/'infrastructure').mkdir();shutil.copytree(parent/'calibration',output/'calibration')
    shutil.copytree(checkpoint,output/'resume/H')
    vehicle=VehicleConfig(**read(output/'calibration/calibration.json')['vehicle'])
    if signature(training_config(p,'H',vehicle))!=read(checkpoint/'config.json')['signature']:
        raise ValueError('Actuation/observation dimensions unexpectedly changed;do not restore this model')
    for directory in ('navstack','scripts'):
        for source in (ROOT/directory).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix in ('.py','.ps1'):
                destination=output/'source'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    p['source_hashes']={f.relative_to(output/'source').as_posix():sha(f) for f in (output/'source').rglob('*') if f.is_file()}
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+14400,resource_limits={**p['resource_limits'],'wall_seconds':14400,'automatic_retries':0})
    if not registered(p):raise ValueError('Adaptation registration incomplete')
    road_settings(p);steering_settings(p);option_settings(p)
    atomic_json(output/'protocol.json',p);atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.parent,a.output)

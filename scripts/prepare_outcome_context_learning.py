"""Register one10240-control course for action-aligned outcome feedback."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.fixed_lowlevel_training import REGISTRATION_OUTCOME,registered
from navstack.outcome_context_policy import VERSION,FAILURE_GAIN_SCALE
from navstack.reference_road_refinement import settings as road_settings
from navstack.steering_transition import settings as steering_settings
from navstack.mpc_reference_effort import settings as mpc_settings
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_fixed_lowlevel_adaptation import without_settings
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare(parent,initializer,equivalence,output,rpc_port=2100):
    parent,initializer,equivalence,output=map(lambda p:Path(p).resolve(),[parent,initializer,equivalence,output])
    require_finished_parent(parent);old=read(parent/'protocol.json');raw=read(parent/'H/endpoint_raw_audit.json')
    receipt=read(initializer/'outcome_migration.json');proof=read(equivalence);meta=verified_checkpoint_metadata(initializer)
    if (output.exists() or output.drive.lower()!='d:' or shutil.disk_usage(output.parent).free<16*1024**3):
        raise ValueError('Use a fresh D-drive course directory with16GiB headroom')
    if (old.get('mpc_effort_transfer_registration',{}).get('original_case_ids')!=[21,27]
            or raw['protocol_sha256']!=sha(parent/'protocol.json')
            or not read(parent/'strict_delivery.json')['strict_delivery_gate_passed']):
        raise ValueError('Require the complete current-controller two-condition baseline')
    if (receipt['old_model_sha256']!=sha(parent/'H/model.zip') or receipt['new_model_sha256']!=meta['model_sha256']
            or receipt['policy_source_sha256']!=sha(ROOT/'navstack/outcome_context_policy.py')
            or proof['model_sha256']!=meta['model_sha256'] or proof['raw_audit_sha256']!=sha(parent/'H/endpoint_raw_audit.json')
            or proof['normalization_sha256']!=meta['normalization_sha256']
            or not proof['outcome_migration_probabilities_and_values_exact'] or not proof['all_recorded_actions_match']
            or sum(x['decisions'] for x in proof['results'])!=147):
        raise ValueError('Require source-bound zero-effect migration on all147actual baseline decisions')
    for name,digest in old['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:
            if name not in ('navstack/fixed_lowlevel_training.py','navstack/mpc_reference_effort.py') or without_settings(ROOT/name)!=without_settings(parent/'source'/name):
                raise ValueError('Freeze every lowlevel function;only registration gates may change:'+name)
    p=deepcopy(old)
    for key in ['exact_compute_recovery','mpc_effort_transfer_registration','fixed_lowlevel_endpoint']:
        p.pop(key,None)
    p.update(reference_context={'version':VERSION},evaluation_only=False,controlled_budget_per_method=10240,
        method_budgets={'H':10240},fixed_lowlevel_training=deepcopy(REGISTRATION_OUTCOME),
        freeze_observation_statistics=True,record_initial_parameters=True,deterministic_training_route_evaluation=False,
        checkpoint_thresholds=list(range(320,10241,320)),
        outcome_context_registration=dict(version=VERSION,failure_gain_scale=FAILURE_GAIN_SCALE,parameter_states=34,new_supervision=0,
            failure_feedback_ablated=receipt['failure_feedback_ablated']))
    if type(rpc_port) is not int or rpc_port not in (2100,2110):raise ValueError('Use the two owned experimental RPC spans')
    p.update(carla_rpc_port=rpc_port,carla_rpc_port_span=list(range(rpc_port,rpc_port+3)))
    p['reference_learning_registration'].update(requested_controls_each=10240,selection='Final fixed-budget checkpoint only')
    prior=old['recovery']['H']
    p['recovery']={'H':dict(checkpoint='resume/H',parent_batch=parent.name,parent_checkpoint='H/'+read(parent/'H/latest_checkpoint.json')['folder'],
        initializer_schema_transform=VERSION,
        lineage_controls=meta['lineage_control_steps'],consumed_controls_lower_bound=meta['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
        retained_optimizer_steps=prior['retained_optimizer_steps'],unknown_unflushed_parent_tail=False,
        model_sha256=meta['model_sha256'],normalization_sha256=meta['normalization_sha256'])}
    p['lineage_targets']={'H':meta['lineage_control_steps']+10240}
    p['fixed_lowlevel_adaptation_registration']=dict(parent_pilot=parent.name,parent_protocol_sha256=sha(parent/'protocol.json'),
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        parent_raw_audit_sha256=sha(parent/'H/endpoint_raw_audit.json'),original_known_development_case_ids=[21,27],
        fixed_requested_controls=10240,maximum_rollout_overshoot=319,new_supervision=0,new_final_conditions=0,
        comparison_label='H0_failure_feedback_off' if receipt['failure_feedback_ablated'] else 'H_outcome_feedback',
        selection='Final fixed requested budget;one continued seed22,all known conditions retained',
        limits='New structured feedback with twozero-initialized learning terms;not new physical information,an action override,independent seed or final confirmation.')
    p['outcome_initialization']=dict(receipt='resume/H/outcome_migration.json',equivalence='outcome_initialization_equivalence.json',
        equivalence_sha256=sha(equivalence),original32states_exact=True,new2states_zero=True)
    p['scope']='Real PPO adaptation of action-aligned outcome feedback with fixed point-constrained planning and reference-centered MPC'
    output.mkdir();(output/'infrastructure').mkdir();shutil.copytree(parent/'calibration',output/'calibration')
    shutil.copytree(initializer,output/'resume/H');shutil.copy2(equivalence,output/'outcome_initialization_equivalence.json')
    for directory in ['navstack','scripts']:
        for f in (ROOT/directory).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.ps1'):
                target=output/'source'/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    p['source_hashes']={f.relative_to(output/'source').as_posix():sha(f) for f in (output/'source').rglob('*') if f.is_file()}
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+14400,resource_limits={**p['resource_limits'],'wall_seconds':14400,'automatic_retries':0})
    if not registered(p):raise ValueError('Outcome learning course registration incomplete')
    road_settings(p);steering_settings(p);mpc_settings(p)
    atomic_json(output/'protocol.json',p);atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','initializer','equivalence','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--rpc-port',type=int,default=2100)
    a=p.parse_args();prepare(a.parent,a.initializer,a.equivalence,a.output,a.rpc_port)

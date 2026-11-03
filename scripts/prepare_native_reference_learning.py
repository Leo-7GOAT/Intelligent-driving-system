"""Register a bounded native course for the explicit reference-context schema."""
import argparse
import json
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import VehicleConfig
from navstack.experiments import signature
from navstack.research import atomic_json
from navstack.reference_context_policy import VERSION
from scripts.audit_native_method_endpoint import read,sha
from scripts.native_road_learning import training_config
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_reference_context import validate_entropy
from navstack.planner_memo import VERSION as MEMO_VERSION


def prepare(output,initializers=None,exact_failure_memo=False):
    if type(exact_failure_memo) is not bool:raise ValueError('Explicit boolean memo opt-in required')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve registered training budgets')
    initial=ROOT/'runs/native_reference_context_initializers_20261103' if initializers is None else Path(initializers).resolve()
    proof=read(initial/'complete.json')
    entropy=validate_entropy(proof.get('entropy_coefficient'))
    methods=list(proof['results'])
    if proof['status']!='INITIALIZERS_PREPARED_NOT_NATIVE_TRAINED' or not methods or not set(methods)<=set(('H','H0')):
        raise ValueError('Require fully verified selected H/H0 initializers')
    parent=ROOT/'runs/20261103T043242Z_native_recovery';old=read(parent/'protocol.json');p=deepcopy(old)
    for name in ('native_validation','explicit_recovery_method_selection','recovery_budget_interpretation'):
        p.pop(name,None)
    p.pop('planner_failure_memo',None)
    if exact_failure_memo:p['planner_failure_memo']=MEMO_VERSION
    p.update(methods=methods,method_budgets={m:10240 for m in methods},controlled_budget_per_method=10240,
        evaluation_only=False,deterministic_training_route_evaluation=False,freeze_observation_statistics=True,
        record_initial_parameters=True,reference_context={'version':VERSION},
        checkpoint_thresholds=list(range(320,10241,320)),
        scope='One canonical seed versioned4736-input selected '+','.join(methods)+' actual TRAIN48course. New learned reference context,H0when selected hides old feedback and reference context. Fixed current lowlevel,reward and categorical actions;zero new supervision. Not isolated context-only training ablation or final confirmation.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<16*1024**3:raise RuntimeError('Staging drive requires16GiBheadroom')
    output.mkdir();(output/'infrastructure').mkdir()
    shutil.copytree(parent/'calibration',output/'calibration')
    shutil.copy2(parent/'measured_demonstrations.npz',output/'measured_demonstrations.npz')
    p['recovery']={};p['lineage_targets']={};vehicle=VehicleConfig(**read(output/'calibration/calibration.json')['vehicle'])
    for method in p['methods']:
        source=initial/method;metadata=verified_checkpoint_metadata(source)
        if entropy is not None:
            change=proof['results'][method].get('entropy_regularization')
            if not isinstance(change,dict) or change.get('version')!='fixed_entropy_v1' or change.get('coefficient')!=entropy:
                raise ValueError('Initializer entropy registration differs')
            with zipfile.ZipFile(source/'model.zip') as archive:
                if json.loads(archive.read('data')).get('ent_coef')!=entropy:
                    raise ValueError('Initializer saved PPO entropy differs')
        if signature(training_config(p,method,vehicle))!=read(source/'config.json')['signature']:
            raise ValueError('Reference context must retain the fixed base physical configuration')
        shutil.copytree(source,output/'resume'/method)
        origin=ROOT/'runs'/proof['results'][method]['parent'];prior=read(origin/'protocol.json')['recovery'][method]
        raw=read(origin/method/'endpoint_raw_audit.json')
        p['recovery'][method]=dict(checkpoint='resume/'+method,parent_batch=origin.name,
            parent_checkpoint=method+'/'+proof['results'][method]['parent_checkpoint'],
            lineage_controls=metadata['lineage_control_steps'],
            consumed_controls_lower_bound=metadata['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
            retained_optimizer_steps=prior.get('retained_optimizer_steps',0)+metadata['optimizer_steps'],
            unknown_unflushed_parent_tail=method=='H0',model_sha256=metadata['model_sha256'],
            normalization_sha256=metadata['normalization_sha256'])
        p['lineage_targets'][method]=metadata['lineage_control_steps']+10240
    for directory in ('navstack','scripts'):
        for source in (ROOT/directory).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix in ('.py','.ps1'):
                destination=output/'source'/source.relative_to(ROOT)
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    p['source_hashes']={f.relative_to(output/'source').as_posix():sha(f) for f in (output/'source').rglob('*') if f.is_file()}
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+14400,
        resource_limits={**p['resource_limits'],'wall_seconds':14400,'automatic_retries':0},
        reference_learning_registration=dict(version=VERSION,requested_controls_each=10240,
            gain_scale=proof.get('gain_scale',1.0),
            entropy_coefficient=entropy,planner_failure_memo=p.get('planner_failure_memo'),
            maximum_rollout_overshoot_each=319,new_auxiliary_updates=0,new_final_conditions=0,
            initializer_proof_sha256=sha(initial/'complete.json'),
            physical_models_and_actions_unchanged=True,
            planner_computation_wrapper_changed=exact_failure_memo,
            selection='Final fixed requested budget endpoint only',
            limits='H0ablates all executable feedback,so this pair does not isolate the incremental reference term alone. All prior failed/consumed costs remain separately retained.'))
    atomic_json(output/'protocol.json',p)
    atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
    print(str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--initializers');p.add_argument('--exact-failure-memo',action='store_true')
    a=p.parse_args();prepare(a.output,a.initializers,a.exact_failure_memo)

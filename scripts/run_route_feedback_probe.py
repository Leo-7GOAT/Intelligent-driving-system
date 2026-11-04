"""Freeze and measure the single TRAIN-only route-feedback endpoint in real CARLA."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.run_broader_native_recovery import basic_gate


def prepare(fit,execution):
    recipe=ROOT/'protocols/native_B1_route_feedback_actor_20261102.json'
    registration=read(fit/'registration.json');result=read(fit/'complete.json')
    metadata=verified_checkpoint_metadata(fit/'checkpoint')
    if (registration['recipe_sha256']!=sha(recipe) or result['actual_Adam_step_each']!=[4096]
            or result['auxiliary_parameter_states']!=5 or result['new_PPO_updates']!=0
            or not result['all_original_actor_critic_tensors_exact']
            or result['model_sha256']!=metadata['model_sha256']):
        raise ValueError('Require the fixed completed residual fit and retained original tensors')
    for name,digest in registration['source_copies'].items():
        if sha(fit/'executed_source'/name)!=digest or sha(ROOT/name)!=digest:
            raise ValueError('Executed source differs:'+name)
    parent=ROOT/'runs/20261102T105643912157Z_native_measured_actor_validation22'
    require_finished_parent(parent);old=read(parent/'protocol.json')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed:'+name)
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<18*1024**3:raise RuntimeError(str(reason))
    p=deepcopy(old)
    for name in ('native_validation','measured_actor_validation_probe'):
        p.pop(name,None)
    now=time.time()
    p.update(routes=deepcopy(old['routes'][:4]),flat_route_observation=True,
        native_case_manifest=dict(version='native_development_cases_v1',split='development',
            cases=[dict(route_index=i,hazard=None) for i in range(4)]),
        deterministic_training_route_evaluation=True,evaluation_only=True,
        checkpoint_thresholds=[],controlled_budget_per_method=0,method_budgets={'B1':0},
        parent_batch=parent.name,created_epoch=now,deadline_epoch=now+1800,
        resource_limits={**old['resource_limits'],'wall_seconds':1800,'automatic_retries':0},
        scope='Fixed canonical22 learned direct route-error residual after extra TRAIN supervision. Known map and measured ego only;no runtime teacher,new PPO or final conditions.')
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_route_feedback_basic22')
    batch.mkdir();(batch/'infrastructure').mkdir()
    for name in ('source','calibration'):
        shutil.copytree(parent/name,batch/name,ignore=shutil.ignore_patterns('__pycache__'))
    overlays=['navstack/config.py','navstack/env.py','navstack/experiments.py',
        'navstack/route_error.py','navstack/route_feedback_policy.py']
    for name in overlays:shutil.copy2(ROOT/name,batch/'source'/name)
    worker=batch/'source/scripts/native_road_learning.py'
    value=worker.read_text(encoding='utf-8')
    marker='config = EnvConfig(method="H" if method == "H0" else method, vehicle=vehicle, road_observation=True,'
    if value.count(marker)!=1 or 'flat_route_observation=' in value:
        raise ValueError('Require original constructor for the explicit observation-only overlay')
    worker.write_text(value.replace(marker,marker+"\n        flat_route_observation=protocol.get('flat_route_observation',False),"),encoding='utf-8')
    overlays.append('scripts/native_road_learning.py')
    for name in overlays:p['source_hashes'][name]=sha(batch/'source'/name)
    shutil.copytree(fit/'checkpoint',batch/'resume/B1')
    shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    evidence=batch/'measured_actor_evidence';evidence.mkdir()
    for name in ('registration.json','complete.json','auxiliary_Adam.pt','sampler_state.json'):
        shutil.copy2(fit/name,evidence/name)
    shutil.copy2(recipe,evidence/'recipe.json')
    p['extra_supervision']['supervised_optimizer_steps']+=4096
    summary=read(parent/'B1/summary.json');raw=read(parent/'B1/endpoint_raw_audit.json')
    p['recovery']['B1'].update(parent_batch=parent.name,parent_checkpoint='route_feedback_actor/checkpoint',
        consumed_controls_lower_bound=summary['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
        model_sha256=metadata['model_sha256'],normalization_sha256=metadata['normalization_sha256'])
    p['measured_actor_basic_probe']=dict(version='known_route_feedback_residual_v5',
        fit_complete_sha256=sha(fit/'complete.json'),recipe_sha256=sha(recipe),source_overlay=overlays,
        newly_performed_actor_steps=4096,cumulative_measured_actor_steps=21504,
        new_learned_parameters=2758,native_teacher_runtime=False,new_PPO_updates=0,new_final_conditions=0,
        gate='At least3/4successANDzero collision/bounds;known24VALonly after gate;no automatic PPO.')
    atomic_json(batch/'protocol.json',p)
    atomic_json(batch/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
    atomic_json(execution/'basic_registration.json',dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json')))
    return batch


def run(execution):
    execution=Path(execution).resolve()
    if (execution/'probe_started.json').exists():raise FileExistsError('Preserve the launched experiment;no repeat')
    atomic_json(execution/'probe_started.json',dict(pid=os.getpid(),epoch=time.time(),automatic_retries=0))
    def call(args,label,timeout=900,env=None):
        atomic_json(execution/'status.json',dict(phase=label,pid=os.getpid(),epoch=time.time()))
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,args)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                timeout=timeout,env=env,check=True)
    def measure(batch,label):
        call([shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1',
            '-Batch',batch,'-UseCurrentGuard'],label+'_native',4200)
        require_finished_parent(batch)
        env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
        call([sys.executable,ROOT/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method','B1'],label+'_raw',env=env)
        call([sys.executable,ROOT/'scripts/audit_native_policy_inference.py','--batch',batch,'--method','B1',
            '--output',batch/'B1/measured_start_inference.json'],label+'_starts',env=env)
        call([sys.executable,ROOT/'scripts/audit_native_control_delivery.py','--batch',batch,
            '--output',batch/'B1/strict_delivery.json'],label+'_delivery')
        if not read(batch/'B1/strict_delivery.json')['strict_delivery_gate_passed']:raise ValueError('Engine delivery failed')
    try:
        basic=prepare(execution/'canonical22_fit',execution);measure(basic,'basic')
        call([sys.executable,ROOT/'scripts/audit_native_flat_sequence.py','--batch',basic,
            '--output',basic/'B1/whole_sequence_inference.json'],'basic_sequence')
        passed=basic_gate(read(basic/'B1/evaluation.json'))
        result=dict(basic_batch=basic.name,basic_gate_passed=passed,new_PPO_updates=0,final_conditions=0,research_acceptance='UNMET')
        if passed:
            output=execution/'validation_probe'
            call([sys.executable,ROOT/'scripts/prepare_measured_actor_validation_probe.py','--parent',basic,'--output',output],'prepare_validation')
            validation=ROOT/'runs'/read(output/'registration.json')['batch'];measure(validation,'known_validation')
            episodes=read(validation/'B1/evaluation.json')['episodes']
            result.update(validation_batch=validation.name,successes=sum(e['success'] for e in episodes),cases=len(episodes))
        atomic_json(execution/'complete.json',result)
        atomic_json(execution/'status.json',dict(phase='DEVELOPMENT_PROBE_COMPLETE',**result))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True)
    run(parser.parse_args().execution)

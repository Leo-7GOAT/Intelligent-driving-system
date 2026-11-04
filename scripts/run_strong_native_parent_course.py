"""Adapt the fixed earlier canonical native policy on TRAIN48; no environment rollback."""
from copy import deepcopy
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import VehicleConfig
from navstack.experiments import signature
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.queue_route_feedback_ppo import training_inventory


def run(rehearsal=False):
    execution=ROOT/('runs/B1_canonical_native_parent_rehearsal_PPO10240_execution_20261102' if rehearsal else 'runs/B1_canonical_native_parent_PPO10240_execution_20261102')
    if execution.exists():raise FileExistsError('Preserve every registered native training course')
    parent=ROOT/'runs/20261101T112758733226Z_native_B1_shared_parking_PPO'
    template=ROOT/('runs/20261102T123912910706Z_native_canonical_parent_TRAIN48_PPO10240' if rehearsal else 'runs/20261102T114745271816Z_native_route_feedback_PPO5120')
    rejected=ROOT/('runs/20261102T125727158887Z_native_route_feedback_PPO_basic' if rehearsal else 'runs/20261102T122150154243Z_native_route_feedback_PPO_known_validation')
    for batch in (parent,template,rejected):require_finished_parent(batch)
    old=read(parent/'protocol.json');physical=read(template/'protocol.json')
    origin=parent/'B1'/read(parent/'B1/latest_checkpoint.json')['folder'];meta=verified_checkpoint_metadata(origin)
    if (meta['model_sha256']!='29efb9c3567c347a194717715570bbe4573699a8af72383e1bf349bd3692c753'
            or meta['lineage_control_steps']!=40960 or old['seed']!=22
            or not all(e['success'] for e in read(parent/'B1/evaluation.json')['episodes'])):
        raise ValueError('Require the fixed actually-trained canonical parent and its basic gate')
    if sha(parent/'calibration/calibration.json')!=sha(template/'calibration/calibration.json'):
        raise ValueError('Do not change physical calibration')
    for name,digest in physical['source_hashes'].items():
        if sha(template/'source'/name)!=digest:raise ValueError('Frozen physical implementation changed:'+name)
    reference=read(ROOT/'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO/protocol.json')
    inventory=read(ROOT/'experiments/native_diverse_TRAIN36_geometry_v2_20261102.json')
    routes,manifest=training_inventory(reference,inventory);p=deepcopy(old)
    p.update(routes=routes,native_case_manifest=manifest,flat_route_observation=False,
        evaluation_only=False,deterministic_training_route_evaluation=False,
        method_budgets={'B1':10240},controlled_budget_per_method=10240,
        checkpoint_thresholds=list(range(1280,10241,1280)),source_hashes=deepcopy(physical['source_hashes']))
    p.pop('shared_parking_PPO',None)
    if rehearsal:
        from navstack.native_cases import registered_native_cases
        from navstack.native_case_schedule import rehearsal_schedule
        p['training_case_schedule']=rehearsal_schedule(registered_native_cases(p))
    vehicle=VehicleConfig(**read(parent/'calibration/calibration.json')['vehicle'])
    if signature(training_config(p,'B1',vehicle))!=read(origin/'config.json')['signature']:
        raise ValueError('Earlier policy is incompatible with the current physical/observation semantics')
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<18*1024**3:raise RuntimeError(str(reason))
    execution.mkdir();shutil.copy2(__file__,execution/'executed_preparation.py')
    registration=dict(version='canonical22_native_parent_TRAIN48_v1',new_actual_PPO_controls=10240,
        basic_rehearsal_schedule=rehearsal,
        source_model_sha256=meta['model_sha256'],source_normalization_sha256=meta['normalization_sha256'],
        script_sha256=sha(Path(__file__)),new_auxiliary_steps=0,seed=22,automatic_retries=0,final_conditions=0,
        rationale='Recent route-error supervised residuals and matched short PPO rates remain unhealthy. Reject those branches and adapt the earlier real canonical native policy;do not restore an old environment or select a best seed.',
        limits='Earlier14/24is mixed-initialization historical development evidence,not a current baseline score. Inherited teacher/pose/reflection supervision remains disclosed in the original protocol;zero new supervised updates. One fixed final endpoint,not five-seed or final confirmation.')
    atomic_json(execution/'registration.json',registration)
    if rehearsal:
        if (sha(template/'resume/B1/model.zip')!=meta['model_sha256']
                or sha(template/'resume/B1/normalization.pkl')!=meta['normalization_sha256']
                or sha(template/'resume/B1/resume_state.pkl')!=sha(origin/'resume_state.pkl')):
            raise ValueError('Matched curriculum contrast must keep initial model/RMS/RNG bytes exact')
        atomic_json(execution/'matched_curriculum_registration.json',dict(
            matched_batch=template.name,unique_conditions=48,training_exposures_per_cycle=64,
            schedule=p['training_case_schedule'],new_actual_control_budget=10240,
            initial_model_RMS_Adam_RNG_exact=True,only_intervention='Training reset order/repeated familiar basic exposures;not new scene conditions,labels,weights,reward or physical control.',
            baseline_basic_outcomes='3success/1bounds;all27TRAINepisodes were on novel roads,none original4basic roads.',
            original_and_post_training_basic_initial_states_exact=True))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_canonical_parent_TRAIN48_PPO10240')
    batch.mkdir();(batch/'infrastructure').mkdir()
    for name in ('source','calibration'):shutil.copytree(template/name,batch/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(origin,batch/'resume/B1')
    shutil.copy2(template/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    overlays=['navstack/env.py','navstack/road_grid_batch.py']
    for name in overlays:
        shutil.copy2(ROOT/name,batch/'source'/name);p['source_hashes'][name]=sha(batch/'source'/name)
    if rehearsal:
        from navstack.native_case_schedule import install_schedule_dispatch
        name='navstack/native_case_schedule.py';shutil.copy2(ROOT/name,batch/'source'/name)
        p['source_hashes'][name]=sha(batch/'source'/name);overlays.append(name)
        worker=batch/'source/scripts/native_road_learning.py'
        worker.write_text(install_schedule_dispatch(worker.read_text(encoding='utf-8')),encoding='utf-8')
        p['source_hashes']['scripts/native_road_learning.py']=sha(worker);overlays.append('scripts/native_road_learning.py')
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+7200,parent_batch=parent.name,
        resource_limits={**old['resource_limits'],'wall_seconds':7200,'automatic_retries':0},
        scope='Canonical22earlier actual native policy on16base-road TRAIN48;10240newrealPPOcontrols,no newlabels. Current physics/calibration,reward and semantics retained;exact observation-only speedups disclosed.',
        direct_PPO_course=dict(version='canonical22_native_parent_TRAIN48_v1',new_PPO_controls=10240,new_auxiliary_steps=0),
        strong_native_parent_PPO={**registration,'registration_sha256':sha(execution/'registration.json'),'observation_only_overlays':overlays})
    recent_cost=read(rejected/'B1/summary.json')['charged_control_steps_lower_bound']+read(rejected/'B1/endpoint_raw_audit.json')['evaluation_control_frames_verified']
    p['recovery']['B1'].update(checkpoint='resume/B1',parent_batch=parent.name,parent_checkpoint='B1/'+origin.name,
        lineage_controls=40960,consumed_controls_lower_bound=max(recent_cost,meta['charged_control_steps_lower_bound']),
        retained_optimizer_steps=old['recovery']['B1'].get('retained_optimizer_steps',0)+meta['optimizer_steps'],
        model_sha256=meta['model_sha256'],normalization_sha256=meta['normalization_sha256'])
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
    atomic_json(execution/'batch.json',dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json')))
    atomic_json(execution/'status.json',dict(phase='ACTUAL_TRAIN_ONLY_PPO_RUNNING',batch=batch.name,pid=os.getpid()))
    try:
        with (execution/'native.log').open('xb') as stream:
            subprocess.run([shutil.which('pwsh'),'-NoProfile','-File',str(ROOT/'scripts/run_registered_native.ps1'),
                '-Batch',str(batch),'-UseCurrentGuard'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=7800,check=True)
        require_finished_parent(batch);summary=read(batch/'B1/summary.json')
        if summary['physics_steps']!=10240:raise ValueError('Actual native budget incomplete')
        atomic_json(execution/'complete.json',dict(batch=batch.name,actual_new_PPO_controls=10240,
            actual_Adam_steps=summary['optimizer_steps'],research_acceptance='UNMET',endpoint_validation='PENDING'))
        atomic_json(execution/'status.json',dict(phase='PPO_COURSE_COMPLETE_REQUIRES_FROZEN_ENDPOINT_AUDIT',batch=batch.name))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rehearsal',action='store_true')
    run(parser.parse_args().rehearsal)

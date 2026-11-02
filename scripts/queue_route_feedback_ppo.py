"""Queue one actual PPO course behind the exact live frozen-policy probe."""
from copy import deepcopy
from datetime import datetime,timezone
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.owned_process import OwnedProcess
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def training_inventory(reference,inventory):
    from navstack.native_cases import registered_native_cases
    routes=deepcopy(reference['routes'][:4])+deepcopy(inventory['routes'])
    # Reuse the actual collected TRAIN hazard templates on every base road.
    hazards={}
    for case in inventory['native_case_manifest']['cases']:
        hazard=case['hazard'];hazards.setdefault('curve' if hazard is None else hazard['kind'],deepcopy(hazard))
    manifest=dict(version='native_development_cases_v1',split='development',cases=[
        dict(route_index=i,hazard=deepcopy(hazards[family]))
        for i in range(16) for family in ('curve','static_vehicle','crossing_walker')])
    if len(registered_native_cases(dict(routes=routes,native_case_manifest=manifest)))!=48:
        raise ValueError('Require the fixed TRAIN48 inventory')
    return routes,manifest


def run():
    execution=ROOT/'runs/B1_route_feedback_PPO5120_execution_20261102'
    if execution.exists():raise FileExistsError('Never repeat a registered native course')
    dependency=ROOT/'runs/B1_route_feedback_actor_execution_20261102'
    inventory_file=ROOT/'experiments/native_diverse_TRAIN36_geometry_v2_20261102.json'
    reference_file=ROOT/'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO/protocol.json'
    routes,manifest=training_inventory(read(reference_file),read(inventory_file))
    pid=int(read(dependency/'probe_started.json')['pid'])
    command=(f"$taskProcess=Get-Process -Id {pid} -ErrorAction Stop; "
        "@{pid=$taskProcess.Id;creation_epoch=([DateTimeOffset]$taskProcess.StartTime).ToUnixTimeMilliseconds()/1000;"
        "executable=$taskProcess.Path} | ConvertTo-Json -Compress")
    identity=json.loads(subprocess.check_output([shutil.which('pwsh'),'-NoProfile','-Command',command],encoding='utf-8-sig'))
    owner=OwnedProcess(**identity,allow_terminate=False)
    execution.mkdir();deadline=time.monotonic()+10800
    atomic_json(execution/'TRAIN_inventory.json',dict(routes=routes,native_case_manifest=manifest))
    atomic_json(execution/'registration.json',dict(version='route_feedback_actual_PPO5120_v1',
        dependency=identity,new_actual_PPO_controls=5120,new_auxiliary_steps=0,seed=22,
        script_sha256=sha(Path(__file__)),TRAIN_inventory_sha256=sha(execution/'TRAIN_inventory.json'),
        source_inventory_sha256=sha(inventory_file),reference_TRAIN_protocol_sha256=sha(reference_file),
        final_conditions=0,automatic_retries=0,selection='One fixed final endpoint;registered while preceding known-VAL probe is still running. No seed or checkpoint selection.',
        motivation='Offline TRAIN regression is insufficient evidence of closed-loop reinforcement learning;update the versioned direct policy from actual CARLA feedback.',
        training='48 TRAIN conditions:4 original base roads plus12 diverse base roads,each curve/static/walker,interleaved by road. No VAL labels or final conditions.',
        learning='Retain PPO actor/critic/Adam/RNG and current Gaussian exploration;5120=4x1280 rollouts,10epochs,batch320. Original clipping/reward/physics and frozen RMS.'))
    try:
        while owner.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Finite dependency wait exhausted')
            atomic_json(execution/'status.json',dict(phase='WAITING_FOR_EXACT_FROZEN_POLICY_PROBE',pid=os.getpid(),epoch=time.time()))
            time.sleep(10)
        if owner.poll()!=0 or not (dependency/'complete.json').exists():raise RuntimeError('Dependency failed;no training or retry')
        parent=ROOT/'runs'/read(dependency/'basic_registration.json')['batch'];require_finished_parent(parent)
        old=read(parent/'protocol.json');basic=read(parent/'B1/evaluation.json')
        if len(basic['episodes'])!=4 or not all(e['success'] for e in basic['episodes']):raise ValueError('Require the actual basic-motion gate')
        sequence=read(parent/'B1/whole_sequence_inference.json')
        if sequence['total_bit_exact_actions']!=sum(e['physics_steps'] for e in basic['episodes']):raise ValueError('Require exact original actions')
        for name,digest in old['source_hashes'].items():
            if sha(parent/'source'/name)!=digest:raise ValueError('Frozen implementation changed:'+name)
        source=parent/'B1'/read(parent/'B1/latest_checkpoint.json')['folder'];metadata=verified_checkpoint_metadata(source)
        if metadata['model_sha256']!='558f162fa342302c6b45b286e0fbd55efd2009b7bea0f8ccbbf9419065445c30':raise ValueError('Fixed initializer changed')
        p=deepcopy(old);p.update(routes=routes,native_case_manifest=manifest)
        okay,reason=resources_ok(ROOT)
        if not okay or shutil.disk_usage(ROOT).free<18*1024**3:raise RuntimeError(str(reason))
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_route_feedback_PPO5120')
        batch.mkdir();(batch/'infrastructure').mkdir()
        for name in ('source','calibration'):shutil.copytree(parent/name,batch/name,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(source,batch/'resume/B1')
        shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
        p.pop('measured_actor_basic_probe');now=time.time()
        p.update(evaluation_only=False,deterministic_training_route_evaluation=False,
            controlled_budget_per_method=5120,method_budgets={'B1':5120},checkpoint_thresholds=[1280,2560,3840,5120],
            freeze_observation_statistics=True,created_epoch=now,deadline_epoch=now+7200,
            resource_limits={**old['resource_limits'],'wall_seconds':7200,'automatic_retries':0},
            scope='One short actual CARLA PPO adaptation of the TRAIN-supervised known-route direct NN. Fixed low level/reward/RMS;48 TRAIN conditions only;not five-seed or budget-matched final evidence.',
            route_feedback_PPO=dict(version='route_feedback_actual_PPO5120_v1',registration_sha256=sha(execution/'registration.json'),new_PPO_controls=5120,new_auxiliary_steps=0))
        # Charge the completed frozen validation even though its states/labels are never fitted.
        val=ROOT/'runs'/read(dependency/'complete.json')['validation_batch']
        total=read(val/'B1/summary.json')['charged_control_steps_lower_bound']+read(val/'B1/endpoint_raw_audit.json')['evaluation_control_frames_verified']
        p['recovery']['B1'].update(checkpoint='resume/B1',parent_batch=parent.name,parent_checkpoint='B1/'+source.name,
            consumed_controls_lower_bound=total,model_sha256=metadata['model_sha256'],normalization_sha256=metadata['normalization_sha256'])
        atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
        atomic_json(execution/'batch.json',dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json')))
        atomic_json(execution/'status.json',dict(phase='ACTUAL_TRAIN_ONLY_PPO_RUNNING',batch=batch.name,pid=os.getpid()))
        with (execution/'native.log').open('xb') as stream:
            subprocess.run([shutil.which('pwsh'),'-NoProfile','-File',str(ROOT/'scripts/run_registered_native.ps1'),
                '-Batch',str(batch),'-UseCurrentGuard'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=7800,check=True)
        require_finished_parent(batch);summary=read(batch/'B1/summary.json')
        if summary['physics_steps']!=5120 or summary['optimizer_steps']<=0:raise ValueError('Actual control/update budget incomplete')
        atomic_json(execution/'complete.json',dict(batch=batch.name,actual_new_PPO_controls=summary['physics_steps'],
            actual_Adam_steps=summary['optimizer_steps'],research_acceptance='UNMET',endpoint_validation='PENDING'))
        atomic_json(execution/'status.json',dict(phase='PPO_COURSE_COMPLETE_REQUIRES_FROZEN_ENDPOINT_AUDIT',batch=batch.name))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise
    finally:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':run()

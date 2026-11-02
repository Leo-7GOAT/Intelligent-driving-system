"""Audit the exact live PPO course, then measure its fixed endpoint without retraining."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.owned_process import OwnedProcess
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.run_broader_native_recovery import basic_gate


def prepare(parent,execution,phase,training_audit):
    require_finished_parent(parent);old=read(parent/'protocol.json')
    origin=parent/'B1'/read(parent/'B1/latest_checkpoint.json')['folder'];meta=verified_checkpoint_metadata(origin)
    proof=read(training_audit)
    if (meta['model_sha256']!=proof['model_sha256'] or proof['new_actual_PPO_controls']!=5120
            or not proof['observation_RMS_exact'] or not proof['all_actual_frames_verified']):
        raise ValueError('Require the fixed actually-trained and audited endpoint')
    if phase=='known_validation' and not basic_gate(read(parent/'B1/evaluation.json')):
        raise ValueError('Require the actual post-PPO basic gate')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Frozen source changed:'+name)
    reference=read(ROOT/'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO/protocol.json')
    p=deepcopy(old)
    if phase=='basic':
        p.update(routes=deepcopy(reference['routes'][:4]),native_case_manifest=dict(
            version='native_development_cases_v1',split='development',cases=[dict(route_index=i,hazard=None) for i in range(4)]))
        p.pop('native_validation',None)
    else:
        for name in ('routes','native_case_manifest','native_validation'):p[name]=deepcopy(reference[name])
    p.pop('freeze_observation_statistics',None)
    now=time.time();wall=1800 if phase=='basic' else 3600
    p.update(evaluation_only=True,deterministic_training_route_evaluation=True,
        controlled_budget_per_method=0,method_budgets={'B1':0},checkpoint_thresholds=[],
        parent_batch=parent.name,created_epoch=now,deadline_epoch=now+wall,
        resource_limits={**old['resource_limits'],'wall_seconds':wall,'automatic_retries':0},
        scope='Frozen canonical22 endpoint after exactly5120newCARLAPPOcontrols;'+phase+' development only. No new training,final conditions or model selection.',
        route_feedback_post_PPO_probe=dict(version='route_feedback_actual_PPO5120_v1',phase=phase,
            training_audit_sha256=sha(training_audit),model_sha256=meta['model_sha256'],new_PPO_updates=0,final_conditions=0))
    summary=read(parent/'B1/summary.json');extra=0
    if (parent/'B1/endpoint_raw_audit.json').exists():extra=read(parent/'B1/endpoint_raw_audit.json')['evaluation_control_frames_verified']
    p['recovery']['B1'].update(checkpoint='resume/B1',parent_batch=parent.name,parent_checkpoint='B1/'+origin.name,
        lineage_controls=summary['lineage_control_steps'],
        consumed_controls_lower_bound=summary['charged_control_steps_lower_bound']+extra,
        retained_optimizer_steps=old['recovery']['B1'].get('retained_optimizer_steps',0)+summary['optimizer_steps'],
        model_sha256=meta['model_sha256'],normalization_sha256=meta['normalization_sha256'])
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<18*1024**3:raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_route_feedback_PPO_'+phase)
    batch.mkdir();(batch/'infrastructure').mkdir()
    for name in ('source','calibration'):shutil.copytree(parent/name,batch/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(origin,batch/'resume/B1')
    shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    evidence=batch/'measured_actor_evidence';evidence.mkdir();shutil.copy2(training_audit,evidence/'training_raw_learning_audit.json')
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
    atomic_json(execution/(phase+'_registration.json'),dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json')))
    return batch


def run():
    execution=ROOT/'runs/B1_route_feedback_PPO5120_postprobe_v2_20261102'
    if execution.exists():raise FileExistsError('Never repeat the fixed post-training probe')
    dependency=ROOT/'runs/B1_route_feedback_PPO5120_execution_20261102'
    pid=int(read(dependency/'status.json')['pid'])
    command=(f"$taskProcess=Get-Process -Id {pid} -ErrorAction Stop; "
        "@{pid=$taskProcess.Id;creation_epoch=([DateTimeOffset]$taskProcess.StartTime).ToUnixTimeMilliseconds()/1000;executable=$taskProcess.Path} | ConvertTo-Json -Compress")
    identity=json.loads(subprocess.check_output([shutil.which('pwsh'),'-NoProfile','-Command',command],encoding='utf-8-sig'))
    owner=OwnedProcess(**identity,allow_terminate=False);execution.mkdir();deadline=time.monotonic()+14400
    hashes={n:sha(ROOT/n) for n in ('scripts/complete_route_feedback_ppo.py','scripts/audit_route_feedback_ppo.py')}
    for name in hashes:shutil.copy2(ROOT/name,execution/Path(name).name)
    atomic_json(execution/'registration.json',dict(dependency=identity,source_hashes=hashes,automatic_retries=0,
        selection='Fixed final5120-control endpoint only;actual raw/Adam/RMS audit,then original4roads,conditionalknown24VAL,no automatic additional training.'))
    def status(phase,**extra):atomic_json(execution/'status.json',dict(phase=phase,pid=os.getpid(),epoch=time.time(),**extra))
    def call(args,label,timeout=900,env=None):
        for name,digest in hashes.items():
            if sha(ROOT/name)!=digest:raise ValueError('Registered audit source changed:'+name)
        status(label)
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,args)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                timeout=min(timeout,max(1,deadline-time.monotonic())),check=True)
    def measure(batch,label):
        call([shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1','-Batch',batch,'-UseCurrentGuard'],label+'_native',4200)
        require_finished_parent(batch);env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
        call([sys.executable,ROOT/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method','B1'],label+'_raw',env=env)
        call([sys.executable,ROOT/'scripts/audit_native_policy_inference.py','--batch',batch,'--method','B1',
            '--output',batch/'B1/measured_start_inference.json'],label+'_starts',env=env)
        call([sys.executable,ROOT/'scripts/audit_native_control_delivery.py','--batch',batch,'--output',batch/'B1/strict_delivery.json'],label+'_delivery')
        if not read(batch/'B1/strict_delivery.json')['strict_delivery_gate_passed']:raise ValueError('Engine delivery failed')
    try:
        while owner.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Finite training dependency wait exhausted')
            status('WAITING_FOR_EXACT_REAL_PPO_COURSE');time.sleep(10)
        if owner.poll()!=0:raise RuntimeError('Training failed;no retry')
        training=ROOT/'runs'/read(dependency/'complete.json')['batch'];proof=training/'B1/training_raw_learning_audit.json'
        call([sys.executable,ROOT/'scripts/audit_route_feedback_ppo.py','--batch',training,'--output',proof],'training_raw_learning_audit')
        basic=prepare(training,execution,'basic',proof);measure(basic,'basic')
        call([sys.executable,ROOT/'scripts/audit_native_flat_sequence.py','--batch',basic,'--output',basic/'B1/whole_sequence_inference.json'],'basic_sequence')
        passed=basic_gate(read(basic/'B1/evaluation.json'));result=dict(training_batch=training.name,basic_batch=basic.name,basic_gate_passed=passed,research_acceptance='UNMET')
        if passed:
            validation=prepare(basic,execution,'known_validation',proof);measure(validation,'known_validation')
            episodes=read(validation/'B1/evaluation.json')['episodes']
            result.update(validation_batch=validation.name,successes=sum(e['success'] for e in episodes),cases=len(episodes))
        atomic_json(execution/'complete.json',result);status('POST_PPO_DEVELOPMENT_PROBES_COMPLETE',**result)
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise
    finally:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':run()

"""Run a registered fixed-budget adaptation and its same-condition frozen endpoint."""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from navstack.independent_clip_ppo import load_ppo
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.collect_native_recovery import tensor_digest


def endpoint_task_profile(protocol):
    """An explicit new pilot may evaluate its locked separated validation set."""
    pilot=protocol.get('contextual_feedback_pilot')
    if pilot is None:return None
    from scripts.prepare_contextual_outcome_pilot import canonical_digest
    profile=pilot['validation_tasks']
    digest=canonical_digest(profile)
    if (set(profile)!={'routes','native_case_manifest','native_validation'}
            or digest!=pilot['validation_tasks_sha256']
            or pilot['validation_case_ids']!=list(range(24))
            or len(profile['native_validation']['manifest']['cases'])!=24):
        raise ValueError('Conditional pilot validation registration changed')
    return deepcopy(profile)


def endpoint(batch,target):
    """Clone the fixed course source and task list;change only learning/eval bookkeeping."""
    require_finished_parent(batch);p=deepcopy(read(batch/'protocol.json'));s=read(batch/'H/summary.json')
    ids=p.get('fixed_lowlevel_adaptation_registration',{}).get('original_known_development_case_ids',[0,1,4])
    source=batch/'H'/read(batch/'H/latest_checkpoint.json')['folder'];m=verified_checkpoint_metadata(source)
    if target.exists():raise FileExistsError(target)
    target.mkdir();(target/'infrastructure').mkdir()
    for name in ['source','calibration']:shutil.copytree(batch/name,target/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(source,target/'resume/H')
    p.pop('lineage_targets',None);p.pop('fixed_lowlevel_training',None)
    p.pop('freeze_observation_statistics',None)
    p.update(evaluation_only=True,controlled_budget_per_method=0,method_budgets={'H':0},checkpoint_thresholds=[],
        deterministic_training_route_evaluation=True,scope=f'Same{len(ids)}known development conditions after one fixed-lowlevel native adaptation;zero new updates')
    profile=endpoint_task_profile(p)
    if profile is not None:
        p.update(profile);p.pop('training_case_schedule',None)
        p['scope']='Prespecified geometry-separated known24 development validation after conditional-feedback pilot;zero updates,not final confirmation'
    prior=p['recovery']['H']
    p['recovery']={'H':dict(checkpoint='resume/H',parent_batch=batch.name,parent_checkpoint='H/'+source.name,
        lineage_controls=m['lineage_control_steps'],consumed_controls_lower_bound=m['charged_control_steps_lower_bound'],
        retained_optimizer_steps=prior['retained_optimizer_steps']+s['optimizer_steps'],unknown_unflushed_parent_tail=False,
        model_sha256=m['model_sha256'],normalization_sha256=m['normalization_sha256'])}
    p['fixed_lowlevel_endpoint']=dict(parent=batch.name,parent_protocol_sha256=sha(batch/'protocol.json'),
        same_all3_conditions=ids==[0,1,4],same_all_conditions=profile is None,original_case_ids=ids,
        same_frozen_source=True,new_PPO_updates=0,new_final_conditions=0)
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+p['resource_limits']['wall_seconds'])
    atomic_json(target/'protocol.json',p);atomic_json(target/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(target/'protocol.json')))


def run(batch,execution):
    import numpy as np
    batch,execution=Path(batch).resolve(),Path(execution).resolve();execution.mkdir()
    p=read(batch/'protocol.json')
    budget=p['controlled_budget_per_method'];parameter_count=p.get('contextual_feedback_registration',
        p.get('outcome_context_registration',{})).get('parameter_states',32)
    ids=p.get('fixed_lowlevel_adaptation_registration',{}).get('original_known_development_case_ids',[0,1,4])
    atomic_json(execution/'registration.json',dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json'),
        controls=budget,endpoint_case_ids=ids,new_final_conditions=0,automatic_retries=0))
    def call(args,label,timeout=1800,env=None):
        atomic_json(execution/'status.json',dict(phase=label,pid=os.getpid(),epoch=time.time()))
        with (execution/(label+'.log')).open('xb') as log:
            subprocess.run(list(map(str,args)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout,env=env)
    def native(target,label):
        arguments=[shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1','-Batch',target,'-UseCurrentGuard']
        if p.get('contextual_feedback_pilot'):
            from scripts.run_independent_outcome_cohort import resource_admission
            atomic_json(execution/(label+'_resource_admission.json'),resource_admission(target.parent))
            arguments+=['-Python',sys.executable,'-CarlaPython',p['contextual_feedback_pilot']['carla_python'],'-ActivateQueued']
        call(arguments,
            label,read(target/'protocol.json')['resource_limits']['wall_seconds']+600)
        require_finished_parent(target)
    def engine(target,label):
        call([sys.executable,ROOT/'scripts/audit_native_control_delivery.py','--batch',target,'--output',target/'strict_delivery.json'],label)
        if not read(target/'strict_delivery.json')['strict_delivery_gate_passed']:raise ValueError('Engine delivery failed')
    try:
        native(batch,'training_native');engine(batch,'training_engine')
        env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
        call([sys.executable,ROOT/'scripts/audit_native_training_only.py','--batch',batch,'--output',batch/'training_only_raw_audit.json'],'training_raw',env=env)
        folder=batch/'H';s=read(folder/'summary.json');before=load_ppo(batch/'resume/H/model.zip',device='cpu');after=load_ppo(folder/'model.zip',device='cpu')
        if not budget<=s['physics_steps']<budget+320 or s['stopped_reason'] is not None or after.num_timesteps-before.num_timesteps!=s['actions']:
            raise ValueError('Fixed native budget/decision counters incomplete')
        fp=read(folder/'initial_parameter_fingerprints.json')
        expected={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in before.policy.state_dict().items()}
        if fp['parameters']!=expected or fp['optimizer_sha256']!=tensor_digest(before.policy.optimizer.state_dict()):
            raise ValueError('Actual initialized weights/Adam differ from registered checkpoint')
        with (batch/'resume/H/resume_state.pkl').open('rb') as f:rng=pickle.load(f)
        if fp['torch_rng_sha256']!=hashlib.sha256(rng['torch'].numpy().tobytes()).hexdigest():raise ValueError('Initial Torch RNG differs')
        old,new=dict(before.policy.named_parameters()),dict(after.policy.named_parameters())
        if set(old)!=set(new) or len(new)!=parameter_count:raise ValueError('Parameter inventory changed')
        for name,value in new.items():
            initial=before.policy.optimizer.state[old[name]];final=after.policy.optimizer.state[value]
            if float(final['step'])-float(initial['step'])!=s['optimizer_steps']:raise ValueError('Adam advancement differs:'+name)
        with (batch/'resume/H/normalization.pkl').open('rb') as f:a=pickle.load(f).obs_rms
        with (folder/'normalization.pkl').open('rb') as f:z=pickle.load(f).obs_rms
        np.testing.assert_array_equal(a.mean,z.mean);np.testing.assert_array_equal(a.var,z.var)
        if a.count!=z.count or sha(folder/'model.zip')==sha(batch/'resume/H/model.zip'):raise ValueError('Frozen RMS or actual learning check failed')
        proof=dict(summary=s,all_parameter_Adam_states_advance=s['optimizer_steps'],parameter_count=parameter_count,actual_initialized_parameters_and_Adam_exact=True,
            initial_torch_RNG_exact=True,RMS_exact=True,new_supervision=0,model_sha256=sha(folder/'model.zip'),
            learned_reference_gain=float(after.policy.action_net.reference_bias.gain.detach()))
        if parameter_count==32:proof['all32Adamstates_advance']=s['optimizer_steps']
        else:proof['learned_failure_gain']=float(after.policy.action_net.reference_bias.failure_gain.detach())
        atomic_json(folder/'saved_learning_proof.json',proof);atomic_json(execution/'training_complete.json',proof)
        target=batch.parent/(batch.name+f'_same{len(ids)}_endpoint');endpoint(batch,target)
        atomic_json(execution/'endpoint_prepared.json',dict(batch=str(target),protocol_sha256=sha(target/'protocol.json')))
        native(target,'endpoint_native');engine(target,'endpoint_engine')
        env=dict(env,NAVSTACK_AUDIT_SOURCE_ROOT=str(target/'source'))
        call([sys.executable,ROOT/'scripts/audit_native_method_endpoint.py','--batch',target,'--method','H'],'endpoint_raw',env=env)
        inference_jobs=([(['--all-actions'],'all_recorded_action_inference.json')] if p.get('contextual_feedback_pilot')
            else [([], 'measured_start_inference.json'),(['--all-static-actions'],'static_action_inference.json')])
        for args,name in inference_jobs:
            call([sys.executable,ROOT/'scripts/audit_native_reference_inference.py','--batch',target,'--method','H',*args,
                '--output',target/'H'/name],'endpoint_'+name,env=env)
        e=read(target/'H/evaluation.json')['episodes']
        if len(e)!=len(ids):raise ValueError('Incomplete registered endpoint inventory')
        atomic_json(execution/'complete.json',dict(training=batch.name,endpoint=target.name,
            results=[dict(original_case_id=i,success=x['success'],termination_reason=x['termination_reason']) for i,x in zip(ids,e)],
            research_acceptance='UNMET',limits=('One selected exploratory warm-started conditional-feedback arm;known separated24validation,not final confirmation.'
                if p.get('contextual_feedback_pilot') else 'Same prespecified known training/development conditions,one continued seed,not independent or a formal contribution claim.')))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--execution',required=True)
    a=p.parse_args();run(a.batch,a.execution)

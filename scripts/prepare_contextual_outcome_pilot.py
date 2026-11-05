"""Prepare a paired, zero-effect conditional-feedback pilot; no native launch."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata

ANCHOR='                configure_continuation_normalization(vec,protocol,recovery)'
VALIDATION_BLOCK='''                if (protocol.get('contextual_feedback_registration') is not None
                        or getattr(model.policy,'contextual_feedback_version',None) is not None):
                    from navstack.contextual_outcome_policy import validate_native_registration
                    validate_native_registration(protocol,model)
'''


def worker_overlay(text):
    if text.count(ANCHOR)!=1 or 'contextual_feedback_registration' in text:
        raise ValueError('Require one unchanged frozen worker admission anchor')
    return text.replace(ANCHOR,VALIDATION_BLOCK+ANCHOR)


def canonical_digest(value):
    def numeric_semantics(item):
        if isinstance(item,dict):return {k:numeric_semantics(v) for k,v in item.items()}
        if isinstance(item,list):return [numeric_semantics(v) for v in item]
        return int(item) if isinstance(item,float) and item.is_integer() else item
    return hashlib.sha256(json.dumps(numeric_semantics(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def identifies_dependency(command,flat):
    try:
        return (command.count('--output')==1
            and Path(command[command.index('--output')+1]).resolve()==Path(flat).resolve()
            and any(Path(arg).name=='run_matched_flat_cohort.py' for arg in command))
    except (ValueError,IndexError,OSError):return False


def prepare(output,cohort,flat,owner_pid):
    import numpy as np
    import psutil
    import torch
    from navstack.config import EnvConfig,VehicleConfig
    from navstack.contextual_outcome_policy import migrate_learner
    from navstack.contextual_feedback_registration import VERSION,TRAINING_REGISTRATION
    from navstack.fixed_lowlevel_training import registered
    from navstack.independent_clip_ppo import load_ppo
    from navstack.owned_process import OwnedProcess
    from scripts.collect_native_recovery import tensor_digest
    from scripts.fixed_shared_native_rule import without_functions
    from scripts.audit_native_policy_sequence import observation_env
    from scripts.native_road_learning import training_config
    output,cohort,flat=map(lambda p:Path(p).resolve(),(output,cohort,flat))
    if output.exists():raise FileExistsError('Preserve all pilot preparations')
    if any((output.parent/(output.name+'_'+arm)).exists() for arm in ('Hc','Hc0')):
        raise FileExistsError('Preserve all pilot arm preparations')
    if output.drive.lower()!='d:' or shutil.disk_usage(output.parent).free<24*1024**3:
        raise ValueError('Use D archive with24GiB free before preparation')
    oldplan=read(cohort/'registration.json');flatplan=read(flat/'registration.json')
    if Path(flatplan['h_cohort']).resolve()!=cohort or flatplan['h_registration_sha256']!=sha(cohort/'registration.json'):
        raise ValueError('Matched queue does not reference the immutable H cohort')
    process=psutil.Process(owner_pid)
    if not identifies_dependency(process.cmdline(),flat):
        raise ValueError('Dependency process does not identify the matched queue')
    identity=dict(pid=owner_pid,creation_epoch=process.create_time(),executable=process.exe())
    if '--parallel-port' not in process.cmdline() or '2120' not in process.cmdline():
        raise ValueError('Require the explicitly isolated2120matched queue')
    owner=OwnedProcess(**identity,allow_terminate=False)
    if owner.poll() is not None:raise ValueError('Matched queue coordinator is not live')
    archive=output.parent
    parent=archive/'20261104_independent_outcome_seed22_H0_TRAIN48_PPO30720'
    validation=archive/'20261104_independent_outcome_seed22_H0_known24'
    require_finished_parent(parent);require_finished_parent(validation)
    template=read(parent/'protocol.json');vp=read(validation/'protocol.json')
    for relative,digest in oldplan['source_hashes'].items():
        if sha(cohort/'source'/relative)!=digest:raise ValueError('Frozen cohort source changed:'+relative)
    gate='navstack/fixed_lowlevel_training.py'
    if without_functions(ROOT/gate,{'registered','outcome_registered'})!=without_functions(cohort/'source'/gate,{'registered','outcome_registered'}):
        raise ValueError('Admission overlay changes another definition')
    for name in ('outcome_context_policy.py','reference_context_policy.py','reference_context.py','road_candidate_policy.py','candidate_filter.py'):
        if (ROOT/'navstack'/name).read_text()!=(cohort/'source/navstack'/name).read_text():
            raise ValueError('Canonical policy family changed:'+name)
    origin=cohort/'initializers/seed22_H';meta=verified_checkpoint_metadata(origin)
    if meta['physics_steps'] or meta['lineage_control_steps'] or meta['optimizer_steps']:
        raise ValueError('Pilot must start from the canonical zero-PPO actor')
    torch.set_num_threads(1)
    old=load_ppo(origin/'model.zip',device='cpu')
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    with (origin/'resume_state.pkl').open('rb') as stream:rng=pickle.load(stream)
    norm.training=False
    cfg=EnvConfig.from_dict(read(origin/'config.json')['env'])
    samples=[]
    for path in sorted((archive/'20261104_outcome_frozen_H_placements30/H').glob('scene_*.json')):
        evidence=read(path);env=observation_env(cfg,evidence['scene'],evidence['initial'])
        try:samples.append(env._observation())
        finally:env.close()
    if len(samples)!=30:raise ValueError('All30saved static measured bases required')
    bases=np.c_[np.stack(samples),np.zeros((30,31),dtype=np.float32)]
    observations=[]
    for value in (0.,1.,-1.):
        variant=bases.copy();variant[:,-29]=value
        observations.append(norm.normalize_obs(variant))
    observations=torch.as_tensor(np.concatenate(observations))
    output.mkdir();(output/'initializers').mkdir()
    shutil.copytree(cohort/'source/navstack',output/'source/navstack',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'scripts',output/'source/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('contextual_outcome_policy.py','contextual_feedback_registration.py','fixed_lowlevel_training.py'):
        shutil.copy2(ROOT/'navstack'/name,output/'source/navstack'/name)
    frozen_worker=(cohort/'source/scripts/native_road_learning.py').read_text()
    (output/'source/scripts/native_road_learning.py').write_text(worker_overlay(frozen_worker),encoding='utf-8')
    shutil.copytree(cohort/'calibration',output/'calibration')
    source_hashes={q.relative_to(output/'source').as_posix():sha(q) for q in (output/'source').rglob('*') if q.is_file()}
    tasks={k:deepcopy(vp[k]) for k in ('routes','native_case_manifest','native_validation')}
    arms=[]
    for name,ablated in (('Hc',False),('Hc0',True)):
        torch.set_rng_state(rng['torch'])
        before_rng=torch.get_rng_state().clone();model=migrate_learner(old,ablated=ablated)
        if not torch.equal(before_rng,torch.get_rng_state()):raise ValueError('Migration advanced Torch RNG')
        with torch.no_grad():
            if (not torch.equal(old.policy.get_distribution(observations).distribution.probs,model.policy.get_distribution(observations).distribution.probs)
                    or not torch.equal(old.policy.predict_values(observations),model.policy.predict_values(observations))):
                raise ValueError('Zero-effect functional migration differs')
        target=output/'initializers'/name;shutil.copytree(origin,target);model.save(target/'model.zip')
        metadata={**meta,'model_sha256':sha(target/'model.zip'),'checkpoint_schema_intervention':True,
            'conditional_feedback_version':VERSION,'parameter_states':42,'new_zero_Adam_states':8}
        atomic_json(target/'metadata.json',metadata);verified_checkpoint_metadata(target)
        restored=load_ppo(target/'model.zip',device='cpu')
        if tensor_digest(restored.policy.state_dict())!=tensor_digest(model.policy.state_dict()):raise ValueError('Saved policy roundtrip differs')
        batch=archive/(output.name+'_'+name);batch.mkdir();(batch/'infrastructure').mkdir()
        for folder in ('source','calibration'):shutil.copytree(output/folder,batch/folder)
        shutil.copytree(target,batch/'resume/H')
        p=deepcopy(template)
        for key in ('independent_outcome_cohort','fixed_cohort_recovery','lineage_targets','native_validation'):
            p.pop(key,None)
        p.update(seed=22,methods=['H'],evaluation_only=False,controlled_budget_per_method=10240,
            method_budgets={'H':10240},fixed_lowlevel_training=deepcopy(TRAINING_REGISTRATION),
            checkpoint_thresholds=list(range(320,10241,320)),source_hashes=source_hashes,
            carla_rpc_port=2120,carla_rpc_port_span=[2120,2121,2122],
            outcome_context_registration={**p['outcome_context_registration'],'failure_feedback_ablated':False},
            contextual_feedback_registration=dict(version=VERSION,hidden=32,actor_scale=100.,parameter_states=42,
                new_parameter_states=8,new_supervision=0,ablated=ablated),
            fixed_lowlevel_adaptation_registration=dict(original_known_development_case_ids=list(range(24))),
            contextual_feedback_pilot=dict(arm=name,selected_exploratory_seed=22,new_supervision=0,new_final_conditions=0,
                validation_tasks=tasks,validation_tasks_sha256=canonical_digest(tasks),validation_case_ids=list(range(24)),
                validation_parent=validation.name,validation_parent_protocol_sha256=sha(validation/'protocol.json'),
                carla_python=oldplan['runtime_carla_python']),
            recovery={'H':dict(checkpoint='resume/H',parent_batch=cohort.name,parent_checkpoint='initializers/seed22_H',
                lineage_controls=0,consumed_controls_lower_bound=meta['charged_control_steps_lower_bound'],
                retained_optimizer_steps=0,unknown_unflushed_parent_tail=False,
                model_sha256=metadata['model_sha256'],normalization_sha256=metadata['normalization_sha256'])},
            scope='Paired selected-seed conditional-feedback pilot from canonical zero-PPO warm actor;TRAIN48curriculum,known24validation,not final confirmation')
        now=time.time();p.update(created_epoch=now,deadline_epoch=now+14400,
            resource_limits={**p['resource_limits'],'wall_seconds':14400,'automatic_retries':0})
        vehicle=VehicleConfig(**read(batch/'calibration/calibration.json')['vehicle'])
        if training_config(p,'H',vehicle).to_dict()!=cfg.to_dict() or not registered(p):raise ValueError('Pilot physical config/admission changed')
        atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',dict(status='QUEUED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
        arms.append(dict(arm=name,batch=batch.name,queued_protocol_sha256=sha(batch/'protocol.json'),
            initializer_model_sha256=metadata['model_sha256'],weights_digest=tensor_digest(model.policy.state_dict()),
            optimizer_digest=tensor_digest(model.policy.optimizer.state_dict()),normalization_sha256=sha(target/'normalization.pkl'),
            resume_state_sha256=sha(target/'resume_state.pkl')))
    for key in ('weights_digest','optimizer_digest','normalization_sha256','resume_state_sha256'):
        if arms[0][key]!=arms[1][key]:raise ValueError('Paired initializers differ:'+key)
    plan=dict(version=VERSION,source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=source_hashes,dependency_cohort=str(flat),dependency_registration_sha256=sha(flat/'registration.json'),
        dependency_identity=identity,dependency_execution='execution_parallel_20261105',maximum_wait_seconds=86400,
        preparation_resources=dict(free_RAM_GiB=psutil.virtual_memory().available/1024**3,
            free_disk_GiB=shutil.disk_usage(output.parent).free/1024**3,new_native_jobs=0),
        arms=arms,controls_per_arm=10240,max_overshoot=319,
        maximum_native_jobs=4,maximum_simultaneous_servers=2,new_supervised_updates=0,new_final_conditions=0,
        measured_base_starts=30,synthetic_feedback_variants=3,exact_functional_inputs=90,
        all_old_tensors_Adam_retained=True,initial_Torch_RNG_exact=True,new_Adam_states_zero=8,
        physical_navstack_unchanged_except_admission=True,worker_overlay='Only explicit conditional saved-policy admission block',
        scope='One selected exploratory seed22pair,not independent five-seed evidence;all outcomes retained,no final test',
        budgets='At most2*(10240+319)training controls and2*24*900evaluation controls;4h/job,0automatic retries;wait<=24h',
        inherited_controls_count_once=meta['charged_control_steps_lower_bound'])
    atomic_json(output/'registration.json',plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('output','cohort','flat'):parser.add_argument('--'+arg,required=True)
    parser.add_argument('--owner-pid',type=int,required=True)
    args=parser.parse_args();result=prepare(args.output,args.cohort,args.flat,args.owner_pid)
    print(json.dumps(dict(status='PREPARED_NOT_STARTED',arms=[r['batch'] for r in result['arms']])))

"""Prepare five uniform flat PPO learners and run them after the active H cohort."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import os
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
from scripts.run_independent_outcome_cohort import resource_admission

SEEDS=(11,22,33,44,55)
BUDGET=30720
VERSION='matched_flat_TRAIN48_uniform_recipe_v1'


def standardize(model_path,norm):
    """Change only the declared zero-PPO recipe and Gaussian exploration."""
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from navstack.fixed_reward_units import initialize_fixed_reward_units,FixedRewardUnitsNormalize
    from scripts.prepare_native_supervised_ppo import fresh_rollout,fresh_learning_rate,fresh_target_kl,initial_exploration
    before=load_ppo(model_path,device='cpu')
    if before.num_timesteps or before._n_updates or before.policy.optimizer.state:
        raise ValueError('Require an independently fitted zero-PPO actor')
    original={n:v.clone() for n,v in before.policy.state_dict().items()}
    model,rollout=(before,None) if before.n_steps==1280 else fresh_rollout(model_path,1280)
    rate=fresh_learning_rate(model,3e-6);kl=fresh_target_kl(model,.01)
    exploration=initial_exploration(model,-4.)
    if model.batch_size!=320 or model.n_epochs!=10:
        raise ValueError('Use the existing flat PPO minibatch/epoch recipe')
    for name,value in model.policy.state_dict().items():
        if name!='log_std' and not torch.equal(value,original[name]):
            raise ValueError('Recipe migration changed actor/critic weights')
    if type(norm).__name__=='RoadCandidateVecNormalize':
        initialize_fixed_reward_units(norm,.05)
    if type(norm) is not FixedRewardUnitsNormalize or norm.fixed_reward_scale!=.05 or norm.norm_reward:
        raise ValueError('Require the disclosed fixed1/20learner reward units')
    norm.training=False
    return model,norm,dict(rollout=rollout,learning_rate=rate,target_kl=kl,exploration=exploration,
        physical_reward_unchanged=True,PPO_reward_scale=.05,all_weights_exact_except_log_std=True)


def prepare(output,h_cohort):
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.experiments import actor_vector,signature
    from navstack.independent_clip_ppo import load_ppo
    from scripts.audit_native_ppo_configuration import profile
    output,h_cohort=Path(output).resolve(),Path(h_cohort).resolve()
    hp=read(h_cohort/'registration.json')
    if output.exists() or output.drive.lower()!='d:' or shutil.disk_usage(output.parent).free<32*1024**3:
        raise ValueError('Preserve existing preparations;use D archive with32GiB headroom')
    if hp['seeds']!=list(SEEDS) or hp['requested_training_controls_per_arm']!=BUDGET:
        raise ValueError('Bind the existing complete five-seed H/H0 design')
    parents=read(ROOT/'protocols/native_B1_all_five_actor_maturation_20261031.json')['parents']
    if tuple(r['seed'] for r in parents)!=SEEDS:
        raise ValueError('Retain all five independently fitted B1actors')
    identities=read(h_cohort.parent/'independent_outcome_cohort_process_identities.json')
    pid=read(h_cohort/'execution/status.json')['pid']
    identity=next(r for r in identities if r['ProcessId']==pid)
    dependency=dict(pid=pid,creation_epoch=datetime.fromisoformat(identity['CreationDate']).timestamp(),
        executable=identity['ExecutablePath'])
    output.mkdir();torch.set_num_threads(1)
    shutil.copytree(h_cohort/'calibration',output/'calibration')
    for folder in ('navstack','scripts'):
        shutil.copytree(ROOT/folder,output/'source'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={p.relative_to(output/'source').as_posix():sha(p) for p in (output/'source').rglob('*') if p.is_file()}
    actors=set();dataset_hashes=set();results=[];reference_profile=None
    for row in parents:
        parent=ROOT/'runs'/row['batch'];origin=parent/'resume/B1'
        metadata=verified_checkpoint_metadata(origin);p=read(parent/'protocol.json')
        if metadata['physics_steps'] or metadata['lineage_control_steps'] or metadata['optimizer_steps']:
            raise ValueError('No selected prior PPO learner may replace a pristine seed')
        data=parent/'measured_demonstrations.npz';expected=p['extra_supervision']['dataset_sha256']
        if sha(data)!=expected:raise ValueError('Inherited TRAIN supervision changed')
        dataset_hashes.add(expected)
        with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
        moments=deepcopy((norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count))
        model,norm,changes=standardize(origin/'model.zip',norm)
        cfg=EnvConfig.from_dict({**read(origin/'config.json')['env'],'goal_parking_assist':True})
        allowed={'method','flat_hold','endpoint_candidate_filter'}
        differences={k for k in cfg.to_dict().keys()|hp['config'].keys() if cfg.to_dict().get(k)!=hp['config'].get(k)}
        if differences-allowed:raise ValueError('Shared physical/reward/task config differs:'+str(differences-allowed))
        if p['extra_supervision']['teacher_controls']!=4329:raise ValueError('Teacher interaction budget differs')
        actor_sha=hashlib.sha256(actor_vector(model).detach().cpu().numpy().tobytes()).hexdigest()
        if actor_sha in actors:raise ValueError('A flat actor was duplicated across seeds')
        actors.add(actor_sha)
        destination=output/'initializers'/f'seed{row["seed"]}';destination.mkdir(parents=True)
        model.save(destination/'model.zip');norm.save(destination/'normalization.pkl')
        shutil.copy2(origin/'resume_state.pkl',destination/'resume_state.pkl')
        with (origin/'resume_state.pkl').open('rb') as f:state=pickle.load(f)
        if state['next_case_index']!=0:raise ValueError('Require the pristine TRAIN cursor zero')
        atomic_json(destination/'config.json',dict(read(origin/'config.json'),env=cfg.to_dict(),signature=signature(cfg),
            explicit_current_wrapper_transition='Current shared parking/calibration fixed throughout new native PPO'))
        meta=dict(metadata,model_sha256=sha(destination/'model.zip'),normalization_sha256=sha(destination/'normalization.pkl'),
            resume_state_sha256=sha(destination/'resume_state.pkl'))
        atomic_json(destination/'metadata.json',meta);verified_checkpoint_metadata(destination)
        actual=profile(destination)
        if reference_profile is None:reference_profile=actual
        if actual!=reference_profile:raise ValueError('Prepared saved PPOrecipes differ across seeds')
        reloaded=load_ppo(destination/'model.zip',device='cpu')
        with np.load(data) as z:sample=z['observations'][::max(1,len(z['observations'])//64)].copy()
        with (origin/'normalization.pkl').open('rb') as f:oldnorm=pickle.load(f)
        np.testing.assert_array_equal(oldnorm.normalize_obs(sample),norm.normalize_obs(sample))
        before=load_ppo(origin/'model.zip',device='cpu')
        a=torch.as_tensor(oldnorm.normalize_obs(sample));b=torch.as_tensor(norm.normalize_obs(sample))
        with torch.no_grad():
            if not torch.equal(before.policy.predict_values(a),reloaded.policy.predict_values(b)):
                raise ValueError('Recipe migration changed native TRAIN value predictions')
            np.testing.assert_array_equal(before.predict(a.numpy(),deterministic=True)[0],
                reloaded.predict(b.numpy(),deterministic=True)[0])
        for a,b in zip(moments,(norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count)):
            np.testing.assert_array_equal(a,b)
        results.append(dict(seed=row['seed'],origin=str(parent.relative_to(ROOT)),initializer=destination.relative_to(output).as_posix(),
            independent_actor_sha256=actor_sha,model_sha256=meta['model_sha256'],config=cfg.to_dict(),
            inherited_controls_lower_bound=metadata['charged_control_steps_lower_bound'],teacher_controls=4329,
            inherited_supervised_updates=p['extra_supervision']['supervised_optimizer_steps'],dataset_sha256=expected,
            preparation_changes=changes,RMS_and_saved_RNG_retained=True,
            native_TRAIN_sample_predictions_exact=len(sample),source_protocol=p))
    if len(dataset_hashes)!=1:raise ValueError('All five seeds must share the same inherited TRAIN data')
    plan=dict(version=VERSION,seeds=list(SEEDS),source_hashes=hashes,initializers=results,
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        h_cohort=str(h_cohort),h_registration_sha256=sha(h_cohort/'registration.json'),dependency_identity=dependency,
        runtime_python=sys.executable,runtime_carla_python=hp['runtime_carla_python'],
        calibration_sha256=sha(output/'calibration/calibration.json'),recipe=reference_profile,
        training_routes=hp['training_routes'],training_manifest=hp['training_manifest'],training_schedule=hp['training_schedule'],
        requested_new_PPO_controls=BUDGET,maximum_training_controls=5*(BUDGET+1279),
        maximum_native_jobs=20,maximum_evaluation_controls=5*(4+4+24)*900,
        maximum_wait_seconds=hp['maximum_total_wall_seconds'],maximum_execution_wall_seconds=20*15000,
        automatic_retries=0,new_supervised_updates=0,new_final_conditions=0,
        limits='Matched new30720PPOwindow and common4329teacher physical controls;historical development costs differ and remain charged.B1uses fixed.05learner reward units,3e-6rate,20Hz direct control,1280-action rollout;Huses its registered2Hz recipe.No claim of equal end-to-end development cost or healthy baseline until actual post-training basic probes.')
    atomic_json(output/'registration.json',plan)


def dependency_complete(cohort):
    cohort=Path(cohort);done=cohort/'execution/complete.json'
    if (cohort/'execution/failure.json').exists():raise RuntimeError('H cohort failed;no B1native launch')
    if not done.exists():return False
    result=read(done)
    if result['status']!='ALL_REGISTERED_JOBS_CLOSED' or result['seeds']!=list(SEEDS):
        raise ValueError('Incomplete H cohort cannot release native dependency')
    for seed in SEEDS:
        for suffix in ('baseline','H_training','H_endpoint','H0_training','H0_endpoint'):
            item=read(cohort/'execution'/f'seed{seed}_{suffix}_complete.json')
            batch=cohort.parent/item['batch'];require_finished_parent(batch)
            if sha(batch/'protocol.json')!=item['protocol_sha256']:
                raise ValueError('Completed dependency protocol changed')
    return True


def protocol_for(output,row,phase,checkpoint):
    from navstack.config import VehicleConfig
    from scripts.native_road_learning import training_config
    from navstack.experiments import signature
    if phase not in ('zero_PPO_basic','TRAIN48_PPO30720','basic','known24'):
        raise ValueError('Unknown registered flat cohort phase')
    output=Path(output);plan=read(output/'registration.json')
    training=phase=='TRAIN48_PPO30720'
    p=deepcopy(row['source_protocol'])
    for key in ('jobs','native_validation','own_success_replay','native_recovery_collection','lineage_targets',
                'gradient_clipping_experiment','rollout_length_experiment','flat_distribution_experiment','actor_head_experiment',
                'supervised_holdout_intervention','supervised_representation_gates','supervised_ppo_intervention','approach_initialization'):
        p.pop(key,None)
    p.update(methods=['B1'],seed=row['seed'],goal_parking_assist=True,evaluation_only=not training,
        deterministic_training_route_evaluation=not training,record_initial_parameters=True,
        rollout_actions=1280,ppo_batch_size=320,ppo_epochs=10,learning_rate=3e-6,
        controlled_budget_per_method=BUDGET if training else 0,method_budgets={'B1':BUDGET if training else 0},
        source_hashes=plan['source_hashes'],calibration_sha256=plan['calibration_sha256'],
        carla_rpc_port=2110,carla_rpc_port_span=[2110,2111,2112],allow_disjoint_carla=True,
        PPO_reward_units=dict(version='fixed_physical_reward_units_v1',scale=.05),
        scope=plan['limits'],checkpoint_thresholds=list(range(1280,BUDGET+1,1280)) if training else [])
    if training:
        p.update(routes=plan['training_routes'],native_case_manifest=plan['training_manifest'],
            training_case_schedule=plan['training_schedule'],freeze_observation_statistics=True)
    elif phase=='known24':
        known=read(Path(plan['h_cohort']).parent/'20261104_outcome_frozen_H_known24/protocol.json')
        p.update(routes=plan['training_routes'],native_case_manifest=plan['training_manifest'],native_validation=known['native_validation'])
    else:
        p.update(routes=plan['training_routes'][:4],native_case_manifest=dict(version=plan['training_manifest']['version'],
            split='development',cases=[dict(route_index=i,hazard=None) for i in range(4)]))
    if not training:
        p.pop('training_case_schedule',None);p.pop('freeze_observation_statistics',None)
    cfg=training_config(p,'B1',VehicleConfig(**read(output/'calibration/calibration.json')['vehicle']))
    if cfg.to_dict()!=row['config'] or signature(cfg)!=read(Path(checkpoint)/'config.json')['signature']:
        raise ValueError('Flat worker configuration differs from initialized physical semantics')
    m=verified_checkpoint_metadata(checkpoint)
    p['recovery']={'B1':dict(checkpoint='resume/B1',parent_batch=output.name,lineage_controls=m['lineage_control_steps'],
        consumed_controls_lower_bound=m['charged_control_steps_lower_bound'],retained_optimizer_steps=m['optimizer_steps'],
        unknown_unflushed_parent_tail=False,model_sha256=m['model_sha256'],normalization_sha256=m['normalization_sha256'])}
    return p


def stage(output,row,phase,checkpoint):
    output=Path(output);plan=read(output/'registration.json')
    batch=output.parent/f'20261104_matched_flat_seed{row["seed"]}_{phase}'
    if batch.exists():raise FileExistsError('Preserve every flat native attempt')
    p=protocol_for(output,row,phase,checkpoint)
    p['resource_admission']=resource_admission(output.parent)
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+14400,
        resource_limits={**p['resource_limits'],'wall_seconds':14400,'automatic_retries':0},
        matched_flat_cohort=dict(version=VERSION,registration_sha256=sha(output/'registration.json'),phase=phase,
            seed=row['seed'],new_final_conditions=0,new_supervised_updates=0))
    batch.mkdir();(batch/'infrastructure').mkdir()
    for name in ('source','calibration'):shutil.copytree(output/name,batch/name)
    shutil.copytree(checkpoint,batch/'resume/B1')
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
    return batch


def run(output,resume_dependency=None):
    import numpy as np
    import torch
    from navstack.owned_process import OwnedProcess
    from navstack.independent_clip_ppo import load_ppo
    from scripts.collect_native_recovery import tensor_digest
    from scripts.run_native_basic_motion_ppo_cohort import basic_qualification
    output=Path(output).resolve();plan=read(output/'registration.json')
    execution=output/('execution_resume_20261105' if resume_dependency else 'execution');execution.mkdir()
    if resume_dependency:
        if any((output.parent/f'20261104_matched_flat_seed{seed}_{phase}').exists()
               for seed in SEEDS for phase in ('zero_PPO_basic','TRAIN48_PPO30720','basic','known24')):
            raise ValueError('This observer recovery requires zero previous native B1 jobs')
        plan['dependency_identity']=read(Path(resume_dependency))
        atomic_json(execution/'dependency_recovery.json',dict(identity=plan['dependency_identity'],
            original_registration_sha256=sha(output/'registration.json'),physical_or_policy_changes=0))
    h=Path(plan['h_cohort']);owner=None;torch.set_num_threads(1)
    def status(phase,**values):atomic_json(execution/'status.json',dict(phase=phase,pid=os.getpid(),epoch=time.time(),**values))
    def call(command,label,timeout=1800,env=None):
        status(label)
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,command)),check=True,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    try:
        if sha(h/'registration.json')!=plan['h_registration_sha256']:raise ValueError('H dependency registration changed')
        until=time.time()+plan['maximum_wait_seconds']
        if not dependency_complete(h):
            d=plan['dependency_identity'];owner=OwnedProcess(d['pid'],d['creation_epoch'],d['executable'],allow_terminate=False)
            while owner.poll() is None:
                if (h/'execution/failure.json').exists() or time.time()>until:
                    raise RuntimeError('Dependency failed or bounded wait expired;no native launch')
                status('WAITING_FOR_OWNED_H_COHORT',dependency_pid=d['pid'],new_native_jobs=0)
                time.sleep(30)
            owner.kernel.CloseHandle(owner.handle);owner=None
        if not dependency_complete(h):raise ValueError('H driver ended without complete evidence;no native launch')
        started=time.time();results=[]
        for row in plan['initializers']:
            checkpoint=output/row['initializer'];training_folder=None
            for phase in ('zero_PPO_basic','TRAIN48_PPO30720','basic','known24'):
                if time.time()-started>plan['maximum_execution_wall_seconds']-15000:
                    raise TimeoutError('Cannot admit another full bounded native job')
                for name,digest in plan['source_hashes'].items():
                    if sha(output/'source'/name)!=digest:raise ValueError('Frozen flat source changed')
                batch=stage(output,row,phase,checkpoint);label=f'seed{row["seed"]}_{phase}'
                call([shutil.which('pwsh'),'-NoProfile','-File',output/'source/scripts/run_registered_native.ps1',
                    '-Batch',batch,'-Python',plan['runtime_python'],'-CarlaPython',plan['runtime_carla_python']],label+'_native',15000)
                require_finished_parent(batch)
                env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
                call([sys.executable,output/'source/scripts/audit_native_control_delivery.py','--batch',batch,'--output',batch/'strict_delivery.json'],label+'_engine',env=env)
                engine=read(batch/'strict_delivery.json')
                if not engine['strict_delivery_gate_passed']:raise ValueError('Flat engine delivery failed')
                if phase=='TRAIN48_PPO30720':
                    call([sys.executable,output/'source/scripts/audit_native_training_only.py','--batch',batch,'--output',batch/'training_only_raw_audit.json'],label+'_raw',env=env)
                    call([sys.executable,output/'source/scripts/audit_native_training_schedule.py','--batch',batch,'--output',batch/'training_schedule_audit.json'],label+'_schedule',env=env)
                    folder=batch/'B1';s=read(folder/'summary.json');fp=read(folder/'initial_parameter_fingerprints.json')
                    a=load_ppo(batch/'resume/B1/model.zip',device='cpu');b=load_ppo(folder/'model.zip',device='cpu')
                    if (not BUDGET<=s['physics_steps']<BUDGET+1280 or s['stopped_reason'] is not None
                            or b.num_timesteps-a.num_timesteps!=s['actions'] or fp['optimizer_sha256']!=tensor_digest(a.policy.optimizer.state_dict())
                            or fp['parameters']!={n:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for n,v in a.policy.state_dict().items()}
                            or tensor_digest(a.policy.state_dict())==tensor_digest(b.policy.state_dict())):raise ValueError('Flat learning/initialization proof failed')
                    with (batch/'resume/B1/resume_state.pkl').open('rb') as f:initial_rng=pickle.load(f)
                    if fp['torch_rng_sha256']!=hashlib.sha256(initial_rng['torch'].numpy().tobytes()).hexdigest():
                        raise ValueError('Actual initial flat RNG differs')
                    for param in b.policy.parameters():
                        if float(b.policy.optimizer.state[param]['step'])!=s['optimizer_steps']:raise ValueError('Saved flat Adam step differs')
                    with (batch/'resume/B1/normalization.pkl').open('rb') as f:x=pickle.load(f).obs_rms
                    with (folder/'normalization.pkl').open('rb') as f:y=pickle.load(f).obs_rms
                    for x1,y1 in ((x.mean,y.mean),(x.var,y.var),(x.count,y.count)):np.testing.assert_array_equal(x1,y1)
                    atomic_json(folder/'saved_learning_proof.json',dict(actual_initial_weights_Adam_exact=True,RMS_exact=True,all_Adam_steps=s['optimizer_steps'],new_supervision=0,summary=s))
                    training_folder=folder;checkpoint=folder/read(folder/'latest_checkpoint.json')['folder'];controls=s['physics_steps']
                else:
                    call([sys.executable,output/'source/scripts/audit_native_method_endpoint.py','--batch',batch,'--method','B1'],label+'_raw',env=env)
                    call([sys.executable,output/'source/scripts/audit_native_policy_inference.py','--batch',batch,'--method','B1','--output',batch/'B1/measured_start_inference.json'],label+'_starts',env=env)
                    episodes=read(batch/'B1/evaluation.json')['episodes'];controls=sum(e['physics_steps'] for e in episodes)
                    if phase=='basic':atomic_json(execution/f'seed{row["seed"]}_basic_health.json',dict(passed=basic_qualification(episodes),episodes=episodes))
                    if phase=='zero_PPO_basic':checkpoint=output/row['initializer']
                if engine['counts']['controls']!=controls:raise ValueError('Flat frame accounting differs')
                results.append(dict(seed=row['seed'],phase=phase,batch=batch.name,actual_controls=controls,protocol_sha256=sha(batch/'protocol.json')))
                atomic_json(execution/(label+'_complete.json'),results[-1]);atomic_json(execution/'completed_jobs.json',results)
        atomic_json(execution/'complete.json',dict(results=results,seeds=list(SEEDS),research_acceptance='UNMET'))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise
    finally:
        if owner is not None:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','run'))
    p.add_argument('--output',required=True);p.add_argument('--h-cohort');p.add_argument('--resume-dependency');args=p.parse_args()
    if args.phase=='prepare':prepare(args.output,args.h_cohort)
    else:run(args.output,args.resume_dependency)

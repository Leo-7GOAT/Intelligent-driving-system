"""Matched actor-only replay versus broader measured TRAIN views;zero PPO."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,pickle,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navstack.corrective_imitation import fit_corrective_actor
from navstack.experiments import signature
from navstack.independent_clip_ppo import load_ppo
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_approach_pair import verify_initializer_runtime,verified_reward_units
from scripts.prepare_native_supervised_ppo import copy_walker_spawn_harness
from scripts.native_partition_initialization import install_inventory


def evaluation_input_protocol(protocol):
    current=deepcopy(protocol)
    # Evaluation freezes statistics in the evaluation loop. This separate flag
    # is deliberately restricted by the runtime to actual learning continuations.
    for name in ('jobs','own_success_replay','freeze_observation_statistics'):
        current.pop(name,None)
    return current


def prepare(parent,views,inventory,steps=512,resume_pair=None,resume_replay=None):
    parent,views,inventory=map(lambda p:Path(p).resolve(),(parent,views,inventory))
    if type(steps) is not int or not 64<=steps<=2000:raise ValueError('Bounded actor fit required')
    if (resume_pair is None)!=(resume_replay is None):raise ValueError('Both interrupted preparation paths are required')
    require_finished_parent(parent);protocol=read(parent/'protocol.json');manifest=read(views/'manifest.json')
    if protocol['methods']!=['B1'] or manifest['new_controls'] or manifest['PPO_updates'] or manifest['supervised_updates']:
        raise ValueError('Require complete B1donor and read-only TRAIN views')
    for name,key in [('measured_training_views.npz','dataset_sha256'),('executed_view_builder.py','script_sha256'),('executed_tick_verifier.py','tick_verifier_sha256')]:
        if sha(views/name)!=manifest[key]:raise ValueError('Measured view provenance differs:'+name)
    if manifest['training_conditions']!=36 or manifest['teacher_training_controls']<=0 or manifest['stride']!=5:
        raise ValueError('Require the complete declared36condition TRAIN reconstruction')
    teacher=ROOT/'runs'/manifest['batch']
    if sha(teacher/'protocol.json')!=manifest['protocol_sha256'] or sha(teacher/'H/summary.json')!=manifest['teacher_summary_sha256']:
        raise ValueError('Training teacher source binding differs')
    for name,digest in manifest['trace_hashes'].items():
        if sha(teacher/'H'/name)!=digest:raise ValueError('Teacher TRAIN trace changed')
    for name,digest in manifest['bridge_hashes'].items():
        if sha(teacher/'infrastructure'/name)!=digest:raise ValueError('Teacher bridge stream changed')
    for name,digest in protocol['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('B1source changed')
    compatibility=verify_initializer_runtime(parent/'source',ROOT)
    base_path=parent/'measured_demonstrations.npz'
    if sha(base_path)!=protocol['extra_supervision']['dataset_sha256']:raise ValueError('Base training dataset changed')
    with np.load(base_path,allow_pickle=False) as d:
        mask=d['train_mask'];base={k:d[k][mask].copy() for k in ('observations','actions','returns','groups')}
        weights=d['weights'][mask].copy() if 'weights' in d else np.ones(mask.sum())
    with np.load(views/'measured_training_views.npz',allow_pickle=False) as d:
        extra={k:d[k].copy() for k in ('observations','actions','groups','conditions')}
    if len(extra['actions'])!=manifest['sampled_rows'] or set(extra['conditions'])!=set(range(36)):
        raise ValueError('Incomplete measured TRAIN condition inventory')
    origin=parent/'resume/B1';initial=verified_checkpoint_metadata(origin)
    if initial['physics_steps'] or initial['optimizer_steps']:raise ValueError('Require zero-PPOorigin')
    configuration=read(origin/'config.json')
    # Target views use exactly the same resolved B1physical and observation interface.
    reference=ROOT/'runs/20261031T073909784878Z_native_supervised_ppo/B1/config.json'
    if manifest['reference_config_sha256']!=sha(reference) or configuration['env']!=read(reference)['env']:
        raise ValueError('View target configuration differs from initializer')
    old_cost=initial['charged_control_steps_lower_bound']
    # New teacher TRAIN controls are a distinct raw source;all other teacher ancestry
    # and selection costs are reported separately rather than pretending exact dedup.
    new_teacher_cost=manifest['teacher_training_controls']
    with ProcessLease(parent/'measured_training_view_pair_prepare.lock'):
        registration=parent/'measured_training_view_pair_registration.json'
        if registration.exists():raise FileExistsError('Preserve previous registered actor fits')
        recipe={'scope':__doc__,'parent':parent.name,'views_manifest_sha256':sha(views/'manifest.json'),'base_dataset_sha256':sha(base_path),
            'base_rows':len(base['actions']),'extra_rows':len(extra['actions']),'steps_each':steps,'learning_rate':1e-4,
            'prior_input_controls_lower_bound':old_cost,'additional_distinct_teacher_TRAIN_controls':new_teacher_cost,
            'teacher_ancestor_and_selection_costs_not_fully_deduplicated':True,'teacher_charged_controls_lower_bound':manifest['teacher_charged_controls_lower_bound'],
            'gate':'Both arms on the same24geometry-separated DEVELOPMENT validation conditions;not final confirmation',
            'training_labels':'All fixed-stride HTRAIN controls,including failed and unfinished training episodes;no validation labels',
            'source_compatibility':compatibility,'PPO_updates':0,'new_controls':0}
        if resume_pair is None:
            pair=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_training_view_pair');pair.mkdir()
            (pair/'registration.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
            shutil.copy2(Path(__file__),pair/'executed_preparation.py')
        else:
            pair=Path(resume_pair).resolve();resume_replay=Path(resume_replay).resolve()
            if not pair.is_relative_to(ROOT/'runs') or not resume_replay.is_relative_to(ROOT/'runs'):
                raise ValueError('Preparation recovery must stay inside this workspace run archive')
            if read(pair/'registration.json')!=recipe or (pair/'complete.json').exists() or (resume_replay/'protocol.json').exists():
                raise ValueError('Only the exact unfinished metadata preparation may be recovered')
            heartbeat=read(pair/'fit_heartbeat.json')
            if heartbeat['arm']!='replay' or heartbeat['additional_supervised_steps']!=steps:
                raise ValueError('No evidence of the completed saved replay fit')
            with (pair/'metadata_failure_recovery.json').open('x') as stream:
                json.dump({'scope':'Saved replay actor fit completed before missing optional checkpoint metadata field caused KeyError;finish metadata without repeating fit',
                    'saved_model_sha256':sha(resume_replay/'resume/B1/model.zip'),'original_preparation_sha256':sha(pair/'executed_preparation.py'),
                    'last_fit_heartbeat':heartbeat,'new_controls':0,'PPO_updates':0},stream,indent=2)
            shutil.copy2(Path(__file__),pair/'executed_preparation_resume.py')
        deadline=time.monotonic()+600;result=[];torch.set_num_threads(1)
        for arm in ('replay','measured36'):
            recovered=resume_replay is not None and arm=='replay'
            if recovered:
                batch=resume_replay
                if any((batch/name).exists() for name in ('started.json','activation.json','B1','launcher_execution.json')):
                    raise ValueError('Never recover preparation over an executed native lifecycle')
                for name,digest in protocol['source_hashes'].items():
                    if sha(batch/'source'/name)!=digest:raise ValueError('Incomplete preparation source changed')
                model=load_ppo(batch/'resume/B1/model.zip',device='cpu');before=load_ppo(origin/'model.zip',device='cpu')
                if model.num_timesteps or model._n_updates or model.policy.optimizer.state:raise ValueError('Saved actor contains PPO state')
                for name,value in before.policy.named_parameters():
                    if 'value_net' in name and not torch.equal(value,dict(model.policy.named_parameters())[name]):
                        raise ValueError('Saved actor fit changed the critic')
            else:
                batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_'+arm+'_training_view_bc');batch.mkdir();(batch/'infrastructure').mkdir()
                shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'));shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(origin,batch/'resume/B1')
                model=load_ppo(origin/'model.zip',device='cpu')
            with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
            verified_reward_units(norm,protocol.get('PPO_reward_units'))
            observations=base['observations'];actions=base['actions'];returns=base['returns'];groups=base['groups'];fit_weights=weights
            if arm=='measured36':
                observations=np.concatenate((observations,extra['observations']));actions=np.concatenate((actions,extra['actions']))
                returns=np.r_[returns,np.zeros(len(extra['actions']),dtype=np.float32)];groups=np.r_[groups,extra['groups']+1048576]
                fit_weights=np.r_[fit_weights,np.ones(len(extra['actions']))]
            def progress(row):
                healthy,reason=resources_ok(ROOT)
                if not healthy or time.monotonic()>deadline:raise RuntimeError('Fit resource/wall guard:'+str(reason))
                (pair/'fit_heartbeat.json').write_text(json.dumps({'arm':arm,'epoch':time.time(),**row}),encoding='utf-8')
            if recovered:
                fit={'additional_supervised_steps':steps,'last_observed_fit_row':heartbeat,'critic_exact':True,
                    'normalization_statistics_exact':True,'PPO_updates':0,'completed_fit_reused_without_retraining':True,
                    'scope':'Recovered saved actor plus last heartbeat;intermediate fit history was not retained. No repeated fitting or new driving.'}
            else:
                fit=fit_corrective_actor(model,norm,observations,actions,fit_weights,steps=steps,learning_rate=1e-4,seed=protocol['seed'],callback=progress)
            destination=batch/'resume/B1'
            if not recovered:model.save(destination/'model.zip')
            if any(sha(destination/name)!=sha(origin/name) for name in ('normalization.pkl','resume_state.pkl')):raise ValueError('Actor fit changed normalizer or RNGbytes')
            charged=old_cost+(new_teacher_cost if arm=='measured36' else 0)
            metadata={**initial,'model_sha256':sha(destination/'model.zip'),'charged_control_steps_lower_bound':charged,
                'supervised_optimizer_steps':protocol['extra_supervision']['supervised_optimizer_steps']+steps,'additional_actor_fit_steps':steps}
            (destination/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');verified_checkpoint_metadata(destination)
            np.savez_compressed(batch/'measured_demonstrations.npz',observations=observations,actions=actions,returns=returns,groups=groups,
                train_mask=np.ones(len(actions),dtype=bool),weights=fit_weights)
            current=evaluation_input_protocol(protocol)
            install_inventory(current,batch/'source',inventory)
            change=copy_walker_spawn_harness(parent/'source/scripts/carla_bridge37.py',ROOT/'scripts/carla_bridge37.py',batch/'source/scripts/carla_bridge37.py')
            now=time.time();current.update(methods=['B1'],evaluation_only=True,controlled_budget_per_method=0,method_budgets={'B1':0},
                rollout_actions=model.n_steps,ppo_epochs=model.n_epochs,ppo_batch_size=model.batch_size,checkpoint_thresholds=[],
                created_epoch=now,deadline_epoch=now+1800,parent_batch=parent.name,scope='Matched actor-only TRAIN-view coverage test;zero PPO,all24development validation cases,not final confirmation',
                resource_limits={**protocol['resource_limits'],'wall_seconds':1800},walker_spawn_initialization=change,
                measured_training_view_initialization={**recipe,'arm':arm,'fit':fit},
                recovery={'B1':{'checkpoint':'resume/B1','parent_batch':parent.name,'parent_checkpoint':'resume/B1','lineage_controls':0,
                    'consumed_controls_lower_bound':charged,'retained_optimizer_steps':0,'unknown_unflushed_parent_tail':False}})
            current['extra_supervision'].update(dataset_sha256=sha(batch/'measured_demonstrations.npz'),supervised_optimizer_steps=metadata['supervised_optimizer_steps'],
                teacher_controls=protocol['extra_supervision']['teacher_controls']+(new_teacher_cost if arm=='measured36' else 0),
                scope='Original TRAIN rows plus optional measured HTRAIN views;all extra physical prior costs disclosed;returns placeholders never critic targets')
            (batch/'measured_training_view_input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            shutil.copy2(Path(__file__),batch/'source/scripts'/Path(__file__).name);shutil.copy2(ROOT/'navstack/corrective_imitation.py',batch/'source/navstack/corrective_imitation.py')
            current['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            (batch/'protocol.json').write_text(json.dumps(current,indent=2),encoding='utf-8');(batch/'prepared.json').write_text(json.dumps({'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')},indent=2),encoding='utf-8')
            result.append({'arm':arm,'batch':batch.name,'fit':fit,'protocol_sha256':sha(batch/'protocol.json'),'model_sha256':sha(destination/'model.zip')})
            (pair/'completed_arms.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'arm':arm,'batch':batch.name,'PPO_updates':0}),flush=True)
        (registration).write_text(json.dumps({'pair':pair.name,'arms':result},indent=2),encoding='utf-8')
        (pair/'complete.json').write_text(json.dumps({'arms':result,'new_controls':0,'PPO_updates':0},indent=2),encoding='utf-8');return pair


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('parent','views','inventory'):p.add_argument('--'+name,required=True)
    p.add_argument('--steps',type=int,default=512);p.add_argument('--resume-pair');p.add_argument('--resume-replay')
    a=p.parse_args();prepare(a.parent,a.views,a.inventory,a.steps,a.resume_pair,a.resume_replay)

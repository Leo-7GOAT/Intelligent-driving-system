"""Versioned all-five native PPO learning course;qualification is checked AFTER learning."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navstack.research import atomic_json,resources_ok
from navstack.fixed_reward_units import initialize_fixed_reward_units
from scripts.audit_native_method_endpoint import read,sha
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_supervised_ppo import fresh_rollout,fresh_learning_rate,fresh_target_kl,initial_exploration


def course_protocol(original,parent_name,controls=10240):
    if (original['methods']!=['B1'] or type(controls) is not int or controls!=10240
            or original.get('native_validation') or original.get('own_success_replay') or original.get('jobs')):
        raise ValueError('Require the canonical direct initializer and fixed basic learning budget')
    protocol=deepcopy(original)
    cases=protocol['native_case_manifest']['cases']
    basic=[c for c in cases if c['hazard'] is None]
    if (len(cases)!=12 or len(basic)!=4 or sorted(c['route_index'] for c in basic)!=list(range(4))):
        raise ValueError('Require every original unobstructed road,without selecting successful cases')
    for name in ('initialization_gate_profile','supervised_representation_gates','supervised_ppo_intervention',
                 'actor_maturation','actor_maturation_preparation'):
        protocol.pop(name,None)
    protocol['native_case_manifest']['cases']=basic
    protocol.update(evaluation_only=False,controlled_budget_per_method=controls,method_budgets={'B1':controls},
        rollout_actions=1280,ppo_batch_size=320,ppo_epochs=10,learning_rate=3e-5,
        freeze_observation_statistics=True,checkpoint_thresholds=list(range(1280,10241,1280)),
        PPO_reward_units={'version':'fixed_physical_reward_units_v1','scale':.05},
        basic_motion_learning={'version':'all_five_basic_PPO10240_lr3e5_v1','parent':parent_name,
            'original_full_case_manifest':deepcopy(original['native_case_manifest']),
            'gate_timing':'AFTER actual PPO learning;this course is not a qualified/full-task baseline',
            'same_all_five_recipe':True,'additional_supervised_updates':0,
            'expected_controls_each':10240,'target_basic_success_fraction':.75,'maximum_basic_unsafe_episodes':0,
            'limits':'New DEVELOPMENT curriculum version after failed supervised-entry gates. Earlier gates are not waived or rewritten. Course interactions and all earlier failed fits/evaluations are charged separately. Any later full-task training needs a new continuation;no H/B1equal-budget or final-performance claim from this course.'},
        scope='Real PPO basic-motion learning on all four known unobstructed roads;all five seeds retained;not a full-task baseline or final confirmation')
    protocol['resource_limits']['wall_seconds']=7200
    protocol['resource_limits']['automatic_retries']=0
    return protocol


def prepare(plan,output):
    plan,output=Path(plan),Path(output);spec=read(plan)
    if [r['seed'] for r in spec['parents']]!=[11,22,33,44,55]:raise ValueError('All five canonical seeds required')
    if output.exists():raise FileExistsError('Preserve prepared course')
    output.mkdir(parents=True);torch.set_num_threads(1);rows=[];datasets=set()
    shutil.copy2(plan,output/'registered_initializers.json')
    try:
        for row in spec['parents']:
            healthy,reason=resources_ok(ROOT)
            if not healthy:raise RuntimeError(str(reason))
            parent=ROOT/'runs'/row['batch'];old=read(parent/'protocol.json');origin=parent/'resume/B1'
            metadata=verified_checkpoint_metadata(origin)
            if old['seed']!=row['seed'] or metadata['physics_steps'] or metadata['optimizer_steps']:
                raise ValueError('Require the original pristine zero-PPO initializer of each seed')
            for name,digest in old['source_hashes'].items():
                if sha(parent/'source'/name)!=digest:raise ValueError('Frozen initializer source changed')
            dataset=parent/'measured_demonstrations.npz'
            if sha(dataset)!=old['extra_supervision']['dataset_sha256']:raise ValueError('Teacher data changed')
            datasets.add(sha(dataset))
            if len(datasets)!=1:raise ValueError('All seeds must share exactly the same TRAIN data')
            batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_B1_basic_PPO_course')
            batch.mkdir();(batch/'infrastructure').mkdir()
            shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(parent/'calibration',batch/'calibration');shutil.copy2(dataset,batch/dataset.name)
            destination=batch/'resume/B1';shutil.copytree(origin,destination)
            from navstack.independent_clip_ppo import load_ppo
            model=load_ppo(origin/'model.zip',device='cpu')
            if model.num_timesteps or model._n_updates or model.policy.optimizer.state:raise ValueError('Initializer has PPOhistory')
            weights={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
            rollout=None
            if model.n_steps!=1280:model,rollout=fresh_rollout(origin/'model.zip',1280)
            if model.batch_size!=320:raise ValueError('Unexpected original PPOminibatch size')
            learning=fresh_learning_rate(model,3e-5);kl=fresh_target_kl(model,.01);exploration=initial_exploration(model,-4.)
            if any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in weights.items() if k!='log_std'):
                raise ValueError('Preparing a curriculum changed actor/critic weights')
            model.save(destination/'model.zip')
            with (origin/'normalization.pkl').open('rb') as f:norm=pickle.load(f)
            moments=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
            reward_change=None
            if type(norm).__name__=='RoadCandidateVecNormalize':reward_change=initialize_fixed_reward_units(norm,.05)
            if type(norm).__name__!='FixedRewardUnitsNormalize' or norm.__dict__.get('fixed_reward_scale')!=.05:
                raise ValueError('Unexpected reward transform')
            norm.training=False;norm.norm_reward=False
            with (destination/'normalization.pkl').open('wb') as f:pickle.dump(norm,f)
            assert np.array_equal(moments[0],norm.obs_rms.mean) and np.array_equal(moments[1],norm.obs_rms.var) and moments[2]==norm.obs_rms.count
            if sha(destination/'resume_state.pkl')!=sha(origin/'resume_state.pkl'):raise ValueError('Seeded PPO RNG changed')
            atomic_json(destination/'metadata.json',{**metadata,'model_sha256':sha(destination/'model.zip'),
                'normalization_sha256':sha(destination/'normalization.pkl')})
            verified_checkpoint_metadata(destination)
            protocol=course_protocol(old,parent.name);now=time.time()
            protocol.update(created_epoch=now,deadline_epoch=now+7200,
                recovery={'B1':{**deepcopy(old['recovery']['B1']),'checkpoint':'resume/B1','parent_batch':parent.name,
                    'parent_checkpoint':'resume/B1','lineage_controls':0,'retained_optimizer_steps':0,
                    'consumed_controls_lower_bound':max(old['recovery']['B1']['consumed_controls_lower_bound'],metadata['charged_control_steps_lower_bound'])}})
            protocol['basic_motion_learning']['initialization_changes']={'rollout':rollout,'learning_rate':learning,
                'target_KL':kl,'exploration':exploration,'reward_units':reward_change,'actor_and_critic_weights_exact_except_declared_log_std':True}
            for relative in ('navstack/fixed_reward_units.py','scripts/prepare_native_basic_motion_ppo.py'):
                shutil.copy2(ROOT/relative,batch/'source'/relative)
            protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            atomic_json(batch/'protocol.json',protocol)
            atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
            rows.append({'seed':row['seed'],'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json'),
                         'initial_model_sha256':sha(destination/'model.zip'),'PPO_updates_before_start':0})
            atomic_json(output/'status.json',{'phase':'PREPARING_ALL_FIVE_BASIC_COURSES','batches':rows})
        atomic_json(output/'status.json',{'phase':'ALL_FIVE_BASIC_PPO_COURSES_PREPARED_NOT_DRIVEN','batches':rows})
    except BaseException as error:
        atomic_json(output/'failure.json',{'error':repr(error),'batches':rows,'new_controls':0,'PPO_updates':0});raise
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.plan,a.output)

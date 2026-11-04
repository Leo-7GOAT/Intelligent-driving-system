"""Stage matched H/H0 development replicates from independently seeded pristine actors."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
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
from navstack.config import VehicleConfig
from navstack.experiments import signature,actor_vector
from navstack.independent_clip_ppo import load_ppo
from navstack.native_partitions import validate_native_partitions
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_native_ppo_configuration import profile
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.native_road_learning import training_config


def broader_protocol(old,inventory,seed):
    if seed not in (11,22,33,44,55) or old['seed']!=seed or old['methods']!=['H','H0']:
        raise ValueError('Require the complete canonical H/H0 seed identity')
    p=deepcopy(old)
    for name in ('jobs','evaluation_only','native_camera_evaluation','geometry_initialization',
                 'supervised_representation_gates','supervised_ppo_intervention','native_partition_initialization'):
        p.pop(name,None)
    for name in ('routes','native_case_manifest','native_validation'):p[name]=deepcopy(inventory[name])
    verification=validate_native_partitions(p)
    if verification['training_conditions']!=36 or verification['validation_conditions']!=24:
        raise ValueError('Require the existing36TRAIN/24development-validation distribution')
    p.update(methods=['H','H0'],controlled_budget_per_method=20480,
        method_budgets={'H':20480,'H0':20480},checkpoint_thresholds=list(range(1280,20481,1280)),
        deterministic_training_route_evaluation=True,freeze_observation_statistics=True,
        scope='Matched common-harness H/H0PPO development replication on36TRAIN/24known validation conditions;not final confirmation',
        broader_feedback_replication={'version':'matched_harness_broader_five_seed_v1','seed':seed,
            'requested_new_controls_each_method':20480,'policy_lineage_starts_at_zero_PPO':True,
            'additional_supervised_updates':0,'teacher_source':'Existing common12condition teacher and independently fitted actor initializations retained;no validation labels or new teacher data',
            'prior_broader_pilot':'Earlier single-seed broader results motivated this replication;they are not substituted for any of these new matched-harness seeds',
            'partition':verification,'stopping_rule':'All five seeds,both methods,one fixed training budget and every24validation case;no favorable checkpoint or seed selection',
            'endpoints':'Safety task completion,unsafe terminations,interventions,time and all failures;paired-seed descriptive intervals. This development replication cannot establish final confirmation.'},
        resource_limits={**old['resource_limits'],'wall_seconds':7200,'automatic_retries':0})
    return p


def prepare(execution,inventory_path):
    execution=Path(execution).resolve();inventory_path=Path(inventory_path).resolve()
    if execution.exists():raise FileExistsError('Preserve the existing preparation')
    inventory=read(inventory_path);pairs=read(ROOT/'protocols/native_feedback_development_pairs_20261031.json')['pairs']
    if [r['seed'] for r in pairs]!=[11,22,33,44,55]:raise ValueError('All five registered seeds are required')
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    execution.mkdir(parents=True);torch.set_num_threads(1)
    reference_profiles={};reference_statistics=None;actor_hashes={'H':set(),'H0':set()};created=[]
    try:
        for pair in pairs:
            old=read(ROOT/pair['h']/'protocol.json');p=broader_protocol(old,inventory,pair['seed'])
            vehicle=VehicleConfig(**read(ROOT/pair['h']/'calibration/calibration.json')['vehicle'])
            origins={};inputs={};costs={}
            for method,key in (('H','h'),('H0','h0')):
                parent=ROOT/pair[key];origin=parent/'resume'/method
                metadata=verified_checkpoint_metadata(origin);saved=read(origin/'config.json')
                if metadata['physics_steps'] or metadata['optimizer_steps'] or metadata['lineage_control_steps']:
                    raise ValueError('Only the original pristine actors may initialize this new distribution')
                if signature(training_config(p,method,vehicle))!=saved['signature']:
                    raise ValueError('Broader replication changes policy/task semantics')
                actual=profile(origin)
                if method in reference_profiles and actual!=reference_profiles[method]:
                    raise ValueError('Saved PPOrecipe differs across seeds')
                reference_profiles[method]=actual
                model=load_ppo(origin/'model.zip',device='cpu')
                digest=hashlib.sha256(actor_vector(model).detach().cpu().numpy().tobytes()).hexdigest()
                if digest in actor_hashes[method]:raise ValueError('A policy was duplicated instead of independently initialized')
                actor_hashes[method].add(digest)
                with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
                stats=(norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count)
                if reference_statistics is None:reference_statistics=tuple(np.copy(v) for v in stats)
                for a,b in zip(stats,reference_statistics):np.testing.assert_array_equal(a,b)
                audit=read(parent/method/'endpoint_raw_audit.json');summary=read(parent/method/'summary.json')
                for name,field in (('summary.json','summary_sha256'),('evaluation.json','evaluation_sha256')):
                    if sha(parent/method/name)!=audit[field]:raise ValueError('Canonical prior-cost evidence changed')
                costs[method]=max(metadata['charged_control_steps_lower_bound'],summary['charged_control_steps_lower_bound'])+audit['evaluation_control_frames_verified']
                inputs[method]={'canonical_batch':parent.name,'pristine_model_sha256':metadata['model_sha256'],
                    'pristine_metadata_sha256':sha(origin/'metadata.json'),'actor_parameter_sha256':digest,
                    'canonical_raw_audit_sha256':sha(parent/method/'endpoint_raw_audit.json'),
                    'prior_development_controls_charged_lower_bound':costs[method]}
                origins[method]=origin
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            batch=ROOT/'runs'/(stamp+'_native_matched_broader_feedback');batch.mkdir()
            (batch/'infrastructure').mkdir();(batch/'resume').mkdir()
            shutil.copytree(ROOT/pair['h']/'calibration',batch/'calibration')
            if sha(batch/'calibration/calibration.json')!=p['calibration_sha256']:raise ValueError('Physical calibration differs')
            for directory in ('navstack','scripts','tests'):
                for source in (ROOT/directory).rglob('*'):
                    if source.is_file() and '__pycache__' not in source.parts and source.suffix in ('.py','.ps1','.json'):
                        target=batch/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            for method,origin in origins.items():
                shutil.copytree(origin,batch/'resume'/method)
                if verified_checkpoint_metadata(batch/'resume'/method)!=verified_checkpoint_metadata(origin):
                    raise ValueError('Copied pristine checkpoint changed')
            teacher=ROOT/pair['h']/'measured_demonstrations.npz'
            if sha(teacher)!=p['extra_supervision']['dataset_sha256']:raise ValueError('Common teacher dataset differs')
            shutil.copy2(teacher,batch/'measured_demonstrations.npz');shutil.copy2(inventory_path,batch/'native_partition_inventory.json')
            atomic_json(batch/'canonical_input_protocol.json',old)
            p['broader_feedback_replication'].update(input_checkpoints=inputs,inventory_sha256=sha(inventory_path),
                common_low_level_source={name:sha(batch/'source'/name) for name in
                    ('navstack/env.py','navstack/planning.py','navstack/mpc.py','navstack/trajectory.py','navstack/shield.py','navstack/carla_bridge.py','scripts/carla_bridge37.py')},
                harness_note='The same current bridge,including pre-control walker clearance,is frozen for both methods and every seed before any PPOupdate. Earlier12condition and interrupted broader runs stay historical and charged.')
            p['recovery']={method:{'checkpoint':'resume/'+method,'parent_batch':inputs[method]['canonical_batch'],
                'parent_checkpoint':'resume/'+method,'lineage_controls':0,'consumed_controls_lower_bound':costs[method],
                'retained_optimizer_steps':0,'unknown_unflushed_parent_tail':read(ROOT/pair['h' if method=='H' else 'h0']/'protocol.json').get('recovery',{}).get(method,{}).get('unknown_unflushed_parent_tail',False)} for method in ('H','H0')}
            now=time.time();p.update(created_epoch=now,deadline_epoch=now+7200)
            p['source_hashes']={f.relative_to(batch/'source').as_posix():sha(f) for f in (batch/'source').rglob('*') if f.is_file() and '__pycache__' not in f.parts}
            atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
            created.append({'seed':pair['seed'],'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')})
            atomic_json(execution/'status.json',{'phase':'PREPARING_MATCHED_BROADER_FEEDBACK','batches':created})
        lowlevels=[read(ROOT/'runs'/r['batch']/'protocol.json')['broader_feedback_replication']['common_low_level_source'] for r in created]
        if any(value!=lowlevels[0] for value in lowlevels):raise ValueError('Common low-level source changed during cohort preparation')
        atomic_json(execution/'status.json',{'phase':'ALL_FIVE_MATCHED_BROADER_FEEDBACK_PREPARED_NOT_DRIVEN','batches':created,
            'distinct_actor_initializations_each_method':{m:len(v) for m,v in actor_hashes.items()},'common_low_level_source':lowlevels[0],
            'new_controls':0,'new_optimizer_updates':0,'research_acceptance':'UNMET'})
        print(json.dumps(read(execution/'status.json')),flush=True);return created
    except BaseException as error:
        atomic_json(execution/'failure.json',{'error':repr(error),'batches':created,'new_controls':0});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True);parser.add_argument('--inventory',required=True)
    args=parser.parse_args();prepare(args.execution,args.inventory)

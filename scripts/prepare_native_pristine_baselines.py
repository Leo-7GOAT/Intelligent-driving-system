"""Freeze five matched teacher-only controls for the broader PPO cohort."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from navstack.experiments import actor_vector
from navstack.independent_clip_ppo import load_ppo
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.run_native_broader_feedback_cohort import require_preparation


def baseline_protocol(old,parent):
    if (old['methods']!=['H','H0'] or old.get('evaluation_only')
            or old['method_budgets']!={'H':20480,'H0':20480}
            or len(old['native_case_manifest']['cases'])!=36
            or len(old['native_validation']['manifest']['cases'])!=24
            or old['recovery']['H']['lineage_controls'] or old['recovery']['H']['retained_optimizer_steps']):
        raise ValueError('Require the unchanged broader matched cohort and pristine H initializer')
    p=deepcopy(old)
    for name in ('jobs','freeze_observation_statistics','broader_feedback_replication','native_camera_evaluation','own_success_replay'):
        p.pop(name,None)
    p.update(methods=['H'],evaluation_only=True,controlled_budget_per_method=0,method_budgets={'H':0},
        checkpoint_thresholds=[],deterministic_training_route_evaluation=True,
        scope='Teacher-only pristine H on all24known broader development-validation conditions;zero PPO/new fits;paired attribution control,not final confirmation',
        recovery={'H':deepcopy(old['recovery']['H'])},
        pristine_policy_baseline={'version':'matched_teacher_only_24conditions_v1','paired_training_batch':parent,
            'teacher_supervised_steps_inherited':old['extra_supervision']['supervised_optimizer_steps_per_method'],
            'additional_supervised_updates':0,'additional_PPO_updates':0,
            'primary_descriptive_metric':'All24case safe task completion;paired trained-H minus pristine-H rates by seed',
            'secondary_metrics':['unsafe terminations','interventions','all-outcome simulation duration','mutually successful case travel time'],
            'selection':'Every registered seed and all24known cases,one fixed pristine endpoint;no validation labels or checkpoint selection',
            'limits':'Disentangles teacher initialization from subsequent PPO on this development distribution. No unseen final or independent-PPO-seed claim for teacher-only models.'},
        resource_limits={**old['resource_limits'],'wall_seconds':1800,'automatic_retries':0})
    p['extra_supervision']['scope']='Retained matched H teacher initialization only;no additional fitting or PPO in this baseline'
    return p


def prepare(cohort,output):
    cohort,output=Path(cohort).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve this registered baseline preparation')
    rows=require_preparation(read(cohort/'status.json'))
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    output.mkdir(parents=True);torch.set_num_threads(1);created=[];actors=set();common_source=None
    try:
        for row in rows:
            parent=ROOT/'runs'/row['batch'];old=read(parent/'protocol.json')
            original=parent/'queued_protocol.json' if (parent/'queued_protocol.json').exists() else parent/'protocol.json'
            if sha(original)!=row['protocol_sha256']:raise ValueError('Registered matched parent changed')
            if common_source is None:common_source=old['source_hashes']
            if old['source_hashes']!=common_source:raise ValueError('Matched complete worker source differs by seed')
            for name,digest in old['source_hashes'].items():
                if sha(parent/'source'/name)!=digest:raise ValueError('Frozen paired source changed')
            p=baseline_protocol(old,parent.name);origin=parent/'resume/H';metadata=verified_checkpoint_metadata(origin)
            if metadata['physics_steps'] or metadata['optimizer_steps'] or metadata['lineage_control_steps']:
                raise ValueError('Teacher-only source contains PPO training')
            model=load_ppo(origin/'model.zip',device='cpu')
            if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
                raise ValueError('Actual model must have zero PPO counters and empty optimizer')
            actor=hashlib.sha256(actor_vector(model).detach().cpu().numpy().tobytes()).hexdigest()
            expected=old['broader_feedback_replication']['input_checkpoints']['H']
            if actor!=expected['actor_parameter_sha256'] or actor in actors:
                raise ValueError('Pristine actor is different or duplicated across seeds')
            actors.add(actor)
            if sha(parent/'measured_demonstrations.npz')!=old['extra_supervision']['dataset_sha256']:
                raise ValueError('Retained teacher provenance changed')
            batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_pristine_H_baseline')
            batch.mkdir();(batch/'infrastructure').mkdir()
            shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(origin,batch/'resume/H')
            shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
            shutil.copy2(parent/'native_partition_inventory.json',batch/'native_partition_inventory.json')
            shutil.copy2(original,batch/'paired_original_protocol.json');shutil.copy2(Path(__file__),batch/'executed_baseline_preparer.py')
            if verified_checkpoint_metadata(batch/'resume/H')!=metadata:raise ValueError('Copied pristine checkpoint changed')
            now=time.time();p.update(created_epoch=now,deadline_epoch=now+1800)
            p['pristine_policy_baseline'].update(model_sha256=metadata['model_sha256'],
                normalization_sha256=metadata['normalization_sha256'],actor_parameter_sha256=actor,
                paired_original_protocol_sha256=row['protocol_sha256'],preparer_sha256=sha(Path(__file__)))
            atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
            created.append({'seed':row['seed'],'batch':batch.name,'paired_training_batch':parent.name,
                'protocol_sha256':sha(batch/'protocol.json'),'model_sha256':metadata['model_sha256'],
                'normalization_sha256':metadata['normalization_sha256'],'actor_parameter_sha256':actor})
            atomic_json(output/'status.json',{'phase':'PREPARING_PRISTINE_H_BASELINES','batches':created})
        atomic_json(output/'status.json',{'phase':'ALL_FIVE_PRISTINE_H_BASELINES_PREPARED_NOT_DRIVEN','batches':created,
            'distinct_pristine_actor_initializations':len(actors),'new_controls':0,'new_optimizer_updates':0,
            'independent_PPO_seeds_added':0,'research_acceptance':'UNMET'})
        return created
    except BaseException as error:
        atomic_json(output/'failure.json',{'error':repr(error),'batches':created});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.cohort,args.output)),flush=True)

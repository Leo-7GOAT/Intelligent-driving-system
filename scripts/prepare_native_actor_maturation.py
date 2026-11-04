"""Fixed additional TRAIN-only actor fitting for all five B1 seeds;zero PPO."""
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
from navstack.corrective_imitation import fit_corrective_actor
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_supervised_ppo import fresh_rollout
from scripts.prepare_native_training_view_pair import evaluation_input_protocol


def prepare(plan_path,output):
    plan_path,output=Path(plan_path),Path(output)
    plan=read(plan_path)
    if ([r['seed'] for r in plan['parents']]!=[11,22,33,44,55]
            or plan['additional_actor_steps']!=1024 or plan['actor_learning_rate']!=5e-5):
        raise ValueError('Require all five seeds and the fixed disclosed maturation recipe')
    if output.exists():raise FileExistsError('Preserve previous preparation')
    data_hashes=set();parents=[];tasks=[]
    for row in plan['parents']:
        parent=ROOT/'runs'/row['batch'];protocol=read(parent/'protocol.json')
        if protocol['seed']!=row['seed'] or protocol['methods']!=['B1']:
            raise ValueError('Parent seed/method differs')
        if (parent/'started.json').exists():require_finished_parent(parent)
        elif (parent/'B1').exists():raise ValueError('Unverified partial native lifecycle')
        origin=parent/'resume/B1';metadata=verified_checkpoint_metadata(origin)
        if metadata['physics_steps'] or metadata['optimizer_steps']:
            raise ValueError('Require the original zero-PPO actor,not trained selected weights')
        for name,digest in protocol['source_hashes'].items():
            if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed')
        dataset=parent/'measured_demonstrations.npz';digest=sha(dataset)
        if digest!=protocol['extra_supervision']['dataset_sha256']:raise ValueError('TRAIN data changed')
        data_hashes.add(digest)
        tasks.append({k:protocol.get(k) for k in ('routes','native_case_manifest','native_validation','calibration_sha256')})
        parents.append((row,parent,protocol,origin,metadata,dataset))
    if len(data_hashes)!=1 or any(t!=tasks[0] for t in tasks):
        raise ValueError('Require the same TRAIN rows and task distribution for every seed')
    output.mkdir(parents=True)
    shutil.copy2(plan_path,output/'registered_plan.json')
    shutil.copy2(Path(__file__),output/'executed_preparation.py')
    torch.set_num_threads(1);deadline=time.monotonic()+900;entries=[]
    atomic_json(output/'status.json',{'phase':'REGISTERED_FIXED_ALL_FIVE_FITS','batches':entries})
    try:
        for row,parent,old,origin,metadata,dataset in parents:
            healthy,reason=resources_ok(ROOT)
            if not healthy:raise RuntimeError(str(reason))
            batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_B1_actor_maturation')
            batch.mkdir();(batch/'infrastructure').mkdir()
            entries.append({'seed':row['seed'],'batch':batch.name})
            atomic_json(output/'status.json',{'phase':'FITTING_TRAIN_ACTOR','seed':row['seed'],'batches':entries})
            shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(parent/'calibration',batch/'calibration')
            shutil.copytree(origin,batch/'resume/B1')
            shutil.copy2(dataset,batch/dataset.name)
            # All gates use the already selected1280-action PPOcollection profile.
            # Rebuilding an EMPTY buffer does not update policy/critic/Adam state.
            from navstack.independent_clip_ppo import load_ppo
            model=load_ppo(origin/'model.zip',device='cpu')
            rollout=None
            if model.n_steps!=1280:model,rollout=fresh_rollout(origin/'model.zip',1280)
            if model.batch_size!=320:raise ValueError('Unexpected original minibatch profile')
            with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
            with np.load(dataset,allow_pickle=False) as d:
                if not d['train_mask'].all() or len(d['actions'])!=3496:
                    raise ValueError('Only the complete registered3496TRAIN rows are allowed')
                observations,actions,weights=[d[k].copy() for k in ('observations','actions','weights')]
            def heartbeat(progress):
                healthy,reason=resources_ok(ROOT)
                if not healthy or time.monotonic()>deadline:raise RuntimeError('Finite fit guard:'+str(reason))
                atomic_json(output/'fit_heartbeat.json',{'seed':row['seed'],'epoch':time.time(),**progress})
            fit=fit_corrective_actor(model,norm,observations,actions,weights,steps=1024,
                                    learning_rate=5e-5,seed=row['seed'],callback=heartbeat)
            destination=batch/'resume/B1';model.save(destination/'model.zip')
            for name in ('normalization.pkl','resume_state.pkl'):
                if sha(destination/name)!=sha(origin/name):raise ValueError('Initializer statistics/RNG changed')
            atomic_json(destination/'metadata.json',{**metadata,'model_sha256':sha(destination/'model.zip'),
                'supervised_optimizer_steps':old['extra_supervision']['supervised_optimizer_steps']+1024})
            verified_checkpoint_metadata(destination)
            protocol=evaluation_input_protocol(old)
            for name in ('initialization_gate_profile','supervised_representation_gates','supervised_ppo_intervention'):
                protocol.pop(name,None)
            now=time.time()
            prior=old['recovery']['B1']
            protocol.update(methods=['B1'],evaluation_only=True,controlled_budget_per_method=0,
                method_budgets={'B1':0},checkpoint_thresholds=[],rollout_actions=1280,
                ppo_epochs=model.n_epochs,ppo_batch_size=model.batch_size,created_epoch=now,deadline_epoch=now+1800,
                recovery={'B1':{**deepcopy(prior),'checkpoint':'resume/B1','parent_batch':parent.name,
                    'parent_checkpoint':'resume/B1','lineage_controls':0,'retained_optimizer_steps':0}},
                actor_maturation={'version':'all_five_TRAIN_actor1024_lr5e5_v1','parent':parent.name,
                    'initial_model_sha256':metadata['model_sha256'],'fit':fit,'empty_buffer_change':rollout,
                    'training_rows':3496,'validation_labels':0,'new_controls':0,'PPO_updates':0,
                    'limits':'All five seeds receive the same fixed additional actor budget. No seed/model selection or claim that TRAIN fit implies driving competence. Earlier failed actor/PPO branches remain charged separately.'},
                scope='New all-five B1TRAIN actor-maturation development version;zero PPO;all12known native conditions must be gated again')
            protocol['resource_limits']['wall_seconds']=1800
            protocol['extra_supervision']['supervised_optimizer_steps']+=1024
            shutil.copy2(Path(__file__),batch/'source/scripts'/Path(__file__).name)
            protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            atomic_json(batch/'protocol.json',protocol)
            atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
            entries[-1].update(protocol_sha256=sha(batch/'protocol.json'),model_sha256=sha(destination/'model.zip'))
        atomic_json(output/'status.json',{'phase':'ALL_FIVE_ACTORS_PREPARED_NOT_DRIVEN','batches':entries})
    except BaseException as error:
        atomic_json(output/'failure.json',{'error':repr(error),'batches':entries,'new_controls':0,'PPO_updates':0})
        raise
    return entries


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();prepare(args.plan,args.output)

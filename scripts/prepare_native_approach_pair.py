"""Matched replay/approach actor fits from one exact zero-PPO initialization."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import importlib.util
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
REPO=Path(__file__).resolve().parents[1]
RUNTIME=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',REPO)).resolve();sys.path.insert(0,str(RUNTIME))
import numpy as np
import torch
from navstack.experiments import write_json
from navstack.independent_clip_ppo import load_ppo
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
fit_path=REPO/'navstack/corrective_imitation.py'
fit_spec=importlib.util.spec_from_file_location('navstack._approach_actor_fit',fit_path)
fit_module=importlib.util.module_from_spec(fit_spec);fit_spec.loader.exec_module(fit_module)
fit_corrective_actor=fit_module.fit_corrective_actor


def prepare(parent,oracle,steps=512,weight=20.):
    parent,oracle=Path(parent).resolve(),Path(oracle).resolve()
    if RUNTIME!=parent/'source' or type(steps) is not int or not 64<=steps<=2000 or not 1<=weight<=50:
        raise ValueError('Require exact parent frozen runtime and bounded matched fit')
    require_finished_parent(parent);protocol=read(parent/'protocol.json');raw=read(parent/'B1/endpoint_raw_audit.json')
    if protocol['methods']!=['B1'] or protocol.get('evaluation_only'):raise ValueError('Require completed real B1PPO donor')
    for path,key in ((parent/'protocol.json','protocol_sha256'),(parent/'B1/config.json','config_sha256'),
        (parent/'B1/summary.json','summary_sha256'),(parent/'B1/evaluation.json','evaluation_sha256'),
        (parent/'B1/evaluation_trace.jsonl.gz','evaluation_trace_sha256')):
        if sha(path)!=raw[key]:raise ValueError('Donor raw binding differs')
    for name,digest in protocol['source_hashes'].items():
        if sha(RUNTIME/name)!=digest:raise ValueError('Donor frozen source changed')
    manifest=read(oracle/'manifest.json')
    if (manifest['raw_audit_sha256']!=sha(parent/'B1/endpoint_raw_audit.json') or not manifest['approach_profile']
        or manifest['new_controls'] or manifest['PPO_updates'] or manifest['supervised_updates']
        or sha(oracle/'corrective_training_queries.npz')!=manifest['dataset_sha256']
        or sha(oracle/'queries.jsonl')!=manifest['query_log_sha256']
        or sha(oracle/'executed_query_script.py')!=manifest['script_sha256']
        or sha(oracle/'executed_history_script.py')!=manifest['history_source_sha256']):
        raise ValueError('Training-only oracle provenance differs')
    records=[json.loads(line) for line in (oracle/'queries.jsonl').read_text(encoding='utf-8').splitlines()]
    available=[r for r in records if r['label_available']]
    keep=np.array([r.get('oracle_mode')=='reference' and r.get('controller_status')=='solved' for r in available])
    if len(available)!=manifest['available_labels'] or keep.sum()<8:raise ValueError('Insufficient reachable-approach queries')
    base_path=parent/'measured_demonstrations.npz'
    if sha(base_path)!=protocol['extra_supervision']['dataset_sha256']:raise ValueError('Original teacher dataset differs')
    with np.load(base_path,allow_pickle=False) as loaded:
        mask=loaded['train_mask'];base={k:loaded[k][mask].copy() for k in ('observations','actions','returns','groups')}
    with np.load(oracle/'corrective_training_queries.npz',allow_pickle=False) as loaded:
        correction={k:loaded[k][keep].copy() for k in ('observations','actions','groups')}
    np.testing.assert_array_equal(correction['actions'],np.array([r['label'] for r,k in zip(available,keep) if k],dtype=np.float32))
    origin=parent/'resume/B1';initial=verified_checkpoint_metadata(origin)
    if initial['physics_steps'] or initial['optimizer_steps']:raise ValueError('Fit starts from zero-PPO initializer,not trained donor weights')
    summary=read(parent/'B1/summary.json');charged=summary['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified']
    okay,reason=resources_ok(REPO)
    if not okay or shutil.disk_usage(REPO).free<12*1024**3:raise RuntimeError(str(reason))
    registration=parent/'reachable_approach_pair_registration.json'
    with ProcessLease(parent/'reachable_approach_pair_prepare.lock'):
        if registration.exists():raise FileExistsError('Do not repeat registered paired fits')
        pair=REPO/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_approach_pair')
        pair.mkdir();deadline=time.monotonic()+600
        write_json(pair/'registration.json',{'scope':__doc__,'parent':parent.name,'oracle_manifest_sha256':sha(oracle/'manifest.json'),
            'actor_fit_source_sha256':sha(fit_path),
            'steps_each':steps,'weight':weight,'wall_seconds':600,'base_training_rows':len(base['actions']),
            'extra_reference_rows':int(keep.sum()),'safe_wait_rows_not_fit':int((~keep).sum()),
            'selection':'All valid solved-reference oracle labels,including braking;no positive-action or outcome-based selection. Original teacher holdout rows remain excluded.',
            'new_controls':0,'PPO_updates':0,'postfit_gate':'All12known development conditions;no checkpoint picking;not independent validation'})
        torch.set_num_threads(1);result=[]
        try:
            for arm in ('replay','approach'):
                batch=REPO/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_'+arm+'_bc')
                batch.mkdir();(batch/'infrastructure').mkdir();shutil.copytree(RUNTIME,batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
                shutil.copytree(parent/'calibration',batch/'calibration');destination=batch/'resume/B1';shutil.copytree(origin,destination)
                model=load_ppo(origin/'model.zip',device='cpu')
                with (origin/'normalization.pkl').open('rb') as f:norm=pickle.load(f)
                observations=base['observations'];actions=base['actions'];returns=base['returns'];groups=base['groups'];weights=np.ones(len(actions))
                if arm=='approach':
                    observations=np.concatenate((observations,correction['observations']));actions=np.concatenate((actions,correction['actions']))
                    returns=np.r_[returns,np.zeros(len(correction['actions']),dtype=np.float32)];groups=np.r_[groups,correction['groups']]
                    weights=np.r_[weights,np.full(len(correction['actions']),weight)]
                def progress(row):
                    healthy,why=resources_ok(REPO)
                    if not healthy or time.monotonic()>=deadline:raise RuntimeError('Fit resource/deadline guard: '+str(why))
                    write_json(pair/'fit_heartbeat.json',{'epoch':time.time(),'arm':arm,**row})
                fit=fit_corrective_actor(model,norm,observations,actions,weights,steps=steps,learning_rate=1e-4,seed=protocol['seed'],callback=progress)
                model.save(destination/'model.zip')
                if sha(destination/'normalization.pkl')!=initial['normalization_sha256'] or sha(destination/'resume_state.pkl')!=initial['resume_state_sha256']:
                    raise ValueError('Fit changed normalization or RNG bytes')
                metadata={**initial,'model_sha256':sha(destination/'model.zip'),
                    'supervised_optimizer_steps':protocol['extra_supervision']['supervised_optimizer_steps']+steps,
                    'charged_control_steps_lower_bound':charged,'additional_actor_fit_steps':steps,
                    'note':'Fresh zero-PPO actor initializer plus matched offline replay/approach fit;donor training is input-data cost,not retained PPO state'}
                write_json(destination/'metadata.json',metadata);verified_checkpoint_metadata(destination)
                np.savez_compressed(batch/'measured_demonstrations.npz',observations=observations,actions=actions,returns=returns,
                    groups=groups,train_mask=np.ones(len(actions),dtype=bool),weights=weights)
                current=deepcopy(protocol)
                for key in ('jobs','freeze_observation_statistics','own_success_replay'):current.pop(key,None)
                now=time.time();current.update(evaluation_only=True,controlled_budget_per_method=0,method_budgets={'B1':0},
                    checkpoint_thresholds=[],created_epoch=now,deadline_epoch=now+1800,parent_batch=parent.name,
                    scope='Actor-only replay/approach B1 initializer;zero PPO updates;all12known conditions;not final confirmation',
                    approach_initialization={**read(pair/'registration.json'),'arm':arm,'fit':fit,
                        'base_teacher_dataset_sha256':sha(base_path),'donor_training_controls':summary['physics_steps'],
                        'donor_evaluation_controls':raw['evaluation_control_frames_verified'],'charged_input_controls_lower_bound':charged,
                        'critic_and_PPO_optimizer_unchanged':True,'normalizer_and_RNG_bytes_exact':True},
                    recovery={'B1':{'checkpoint':'resume/B1','parent_batch':parent.name,'parent_checkpoint':'resume/B1',
                        'lineage_controls':0,'consumed_controls_lower_bound':charged,'retained_optimizer_steps':0,'unknown_unflushed_parent_tail':False}},
                    resource_limits={**protocol['resource_limits'],'wall_seconds':1800})
                current['extra_supervision'].update(dataset_sha256=sha(batch/'measured_demonstrations.npz'),
                    supervised_optimizer_steps=metadata['supervised_optimizer_steps'],epochs=None,
                    holdout_scope='Only original teacher TRAIN rows plus optional reachable training-state labels;no fit holdout or independent validation claim',
                    scope='Disclosed actor-only replay and optional counterfactual B3/MPC approach labels;return placeholders never value targets')
                shutil.copy2(Path(__file__),batch/'source/scripts'/Path(__file__).name)
                shutil.copy2(fit_path,batch/'source/navstack/corrective_imitation.py')
                shutil.copytree(oracle,batch/'source/approach_evidence')
                current['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
                write_json(batch/'protocol.json',current);write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
                record={'arm':arm,'batch':batch.name,'model_sha256':metadata['model_sha256'],'fit':fit};result.append(record)
                write_json(pair/'completed_arms.json',result);print(json.dumps({'arm':arm,'batch':batch.name,'PPO_updates':0}),flush=True)
            write_json(registration,{'pair':pair.name,'arms':result});write_json(pair/'complete.json',{'arms':result,'new_controls':0,'PPO_updates':0})
            return pair
        except BaseException as exc:
            write_json(pair/'failure.json',{'error':repr(exc),'completed_arms':result,'new_controls':0,'PPO_updates':0});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--oracle',required=True)
    p.add_argument('--steps',type=int,default=512);p.add_argument('--weight',type=float,default=20.)
    a=p.parse_args();prepare(a.parent,a.oracle,a.steps,a.weight)

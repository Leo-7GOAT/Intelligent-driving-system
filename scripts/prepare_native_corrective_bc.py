"""Versioned offline corrective fit; a separate real-CARLA gate precedes PPO."""
import argparse
from datetime import datetime,timezone
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
from stable_baselines3 import PPO
from navstack.corrective_imitation import fit_corrective_actor
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import audit,read,sha
from scripts.evaluate_native_rule import require_finished_parent


def corrective_input_cost(protocol,base_manifest,donor_controls):
    """Charge new physical source batches once, retaining every ancestor cost."""
    known={row['batch'] for row in base_manifest['sources']}
    previous=protocol.get('corrective_initialization',{})
    known.update(previous.get('charged_input_batches',[]))
    known.update(previous[k] for k in ('parent','oracle_source') if previous.get(k))
    inherited=max(protocol['extra_supervision']['teacher_controls'],
                  protocol['recovery']['B1']['consumed_controls_lower_bound'])
    if type(inherited) is not int or inherited<0 or any(type(v) is not int or v<0 for v in donor_controls.values()):
        raise ValueError('Invalid physical corrective input ledger')
    return inherited+sum(v for k,v in donor_controls.items() if k not in known),sorted(known|set(donor_controls))


def prepare(parent,oracle,base_views,steps=544,weight=20.,learning_rate=1e-4):
    parent,oracle,base_views=map(lambda p:Path(p).resolve(),(parent,oracle,base_views))
    if (type(steps) is not int or not 1<=steps<=2000 or not np.isfinite(weight)
            or not 1<=weight<=50 or not np.isfinite(learning_rate) or not 0<learning_rate<=.01):
        raise ValueError('Invalid bounded corrective fit registration')
    require_finished_parent(parent)
    protocol=read(parent/'protocol.json')
    if protocol.get('methods')!=['B1'] or not protocol.get('evaluation_only'):
        raise ValueError('Require a complete zero-PPO direct initializer')
    origin=parent/'B1'/read(parent/'B1/latest_checkpoint.json')['folder']
    metadata=read(origin/'metadata.json')
    if metadata['physics_steps'] or metadata['optimizer_steps']:
        raise ValueError('Do not mix actual PPO training into an actor-only diagnostic')
    manifest=read(oracle/'manifest.json');exact=read(oracle/'exact_inputs.json')
    source=ROOT/'runs'/manifest['parent']
    if source.name!=manifest['parent'] or source.parent!=ROOT/'runs':raise ValueError('Invalid corrective source identity')
    audits={}
    for donor in {parent,source}:
        require_finished_parent(donor);verified=audit(donor,'B1')
        if verified!=read(donor/'B1/endpoint_raw_audit.json'):raise ValueError('Corrective parent raw audit changed')
        audits[donor.name]=verified
        guard=read(donor/'guard_final.json')
        if guard['exit_code']!=0 or guard['reason'] is not None or read(donor/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped':
            raise ValueError('Require normal owned parent lifecycle')
    if (manifest['raw_audit_sha256']!=sha(source/'B1/endpoint_raw_audit.json')
            or manifest['source_protocol_sha256']!=sha(source/'protocol.json')
            or manifest['new_controls'] or manifest['PPO_updates']
            or exact['new_controls'] or exact['PPO_updates']
            or exact['policy_actions_exact']!=manifest['available_labels']
            or sha(oracle/'queries.jsonl')!=manifest['query_log_sha256']
            or sha(oracle/'corrective_queries.npz')!=manifest['dataset_sha256']
            or sha(oracle/'corrective_queries_exact_inputs.npz')!=exact['exact_dataset_sha256']):
        raise ValueError('Corrective measured-input/oracle provenance changed')
    base_manifest=read(base_views/'manifest.json');base_path=base_views/'measured_demonstrations.npz'
    if (sha(base_path)!=base_manifest['dataset_sha256'] or base_manifest['teacher_unique_controls']!=protocol['extra_supervision']['teacher_controls']):
        raise ValueError('Original shared teacher dataset changed')
    source_protocol=read(source/'protocol.json')
    if (source_protocol.get('native_case_manifest')!=protocol.get('native_case_manifest')
            or source_protocol['calibration_sha256']!=protocol['calibration_sha256']):
        raise ValueError('Corrective conditions/calibration differ')
    for name,digest in protocol['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent frozen source changed')
        if name.startswith('navstack/') and (ROOT/name).read_text(encoding='utf-8')!=(parent/'source'/name).read_text(encoding='utf-8'):
            raise ValueError('Do not silently change the live observation/planning/control implementation')
    label='corrective_bc_'+exact['exact_dataset_sha256'][:12]+f'_steps{steps}_weight{weight:g}_lr{learning_rate:g}'
    registration=parent/(label+'_registration.json')
    with ProcessLease(parent/'corrective_prepare.lock'):
        if registration.exists():raise FileExistsError('Do not repeat a registered corrective fit')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_corrective_bc')
        batch.mkdir();deadline=time.monotonic()+600
        def health():
            okay,reason=resources_ok(ROOT)
            if not okay or time.monotonic()>=deadline or shutil.disk_usage(ROOT).free<12*1024**3:
                raise RuntimeError('Corrective fit resource/deadline guard: '+str(reason))
        write_json(batch/'fit_registration.json',{'scope':'Additional actor-only MSE fit on ALL original teacher rows plus weighted counterfactual corrective labels. No holdout accuracy, ordinary-PPO or driving-improvement claim.',
            'parent':parent.name,'oracle_source':source.name,'oracle_manifest_sha256':sha(oracle/'manifest.json'),
            'exact_inputs_sha256':sha(oracle/'exact_inputs.json'),'steps':steps,'corrective_weight':weight,
            'learning_rate':learning_rate,'wall_seconds':600,'seed':protocol['seed']})
        try:
            health();torch.set_num_threads(1)
            with np.load(base_path,allow_pickle=False) as data:
                base={key:data[key].copy() for key in ('observations','actions','returns','groups')}
            with np.load(oracle/'corrective_queries_exact_inputs.npz',allow_pickle=False) as data:
                correction={key:data[key].copy() for key in ('observations','actions','groups')}
            count=len(correction['actions'])
            if count!=manifest['available_labels'] or count<12:raise ValueError('Incomplete corrective labels')
            observations=np.concatenate((base['observations'],correction['observations']))
            actions=np.concatenate((base['actions'],correction['actions']))
            weights=np.r_[np.ones(len(base['actions'])),np.full(count,weight)]
            model=PPO.load(origin/'model.zip',device='cpu')
            with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
            def progress(row):
                health();write_json(batch/'fit_heartbeat.json',{'epoch':time.time(),**row});print(json.dumps(row),flush=True)
            fitted=fit_corrective_actor(model,norm,observations,actions,weights,steps=steps,
                learning_rate=learning_rate,seed=protocol['seed'],callback=progress)
            destination=batch/'resume/B1';shutil.copytree(origin,destination)
            model.save(destination/'model.zip')
            # Copying retains exact normalization and PPO RNG/optimizer inputs.
            assert sha(destination/'normalization.pkl')==sha(origin/'normalization.pkl')
            assert sha(destination/'resume_state.pkl')==sha(origin/'resume_state.pkl')
            teacher_controls=base_manifest['teacher_unique_controls']
            charged_controls,charged_batches=corrective_input_cost(protocol,base_manifest,
                {name:a['total_applied_controls_verified'] for name,a in audits.items()})
            previous_steps=protocol['extra_supervision']['supervised_optimizer_steps']
            metadata.update(model_sha256=sha(destination/'model.zip'),
                supervised_optimizer_steps=previous_steps+steps,charged_control_steps_lower_bound=charged_controls,
                corrective_actor_fit_steps=steps,note='Additional disclosed offline actor fit; critic, normalization bytes, PPO optimizer and starting RNG retained. New physical episode at evaluation.')
            write_json(destination/'metadata.json',metadata)
            np.savez_compressed(batch/'measured_demonstrations.npz',observations=observations,actions=actions,
                returns=np.r_[base['returns'],np.zeros(count,dtype=np.float32)],
                train_mask=np.ones(len(actions),dtype=bool),groups=np.r_[base['groups'],correction['groups']],weights=weights)
            (batch/'infrastructure').mkdir();shutil.copytree(parent/'calibration',batch/'calibration')
            for folder in ('navstack','scripts','tests'):
                for path in (ROOT/folder).rglob('*'):
                    if path.is_file() and '__pycache__' not in path.parts and (path.suffix in ('.py','.ps1') or 'fixtures' in path.parts):
                        target=batch/'source'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            evidence=batch/'source/corrective_evidence';shutil.copytree(oracle,evidence)
            now=time.time();protocol.pop('supervised_holdout_intervention',None)
            protocol.update(created_epoch=now,deadline_epoch=now+1800,parent_batch=parent.name,
                evaluation_only=True,controlled_budget_per_method=0,method_budgets={'B1':0},
                scope='Frozen direct B1 after additional counterfactual corrective actor supervision; zero PPO updates, known-condition development only',
                corrective_initialization={**read(batch/'fit_registration.json'),'fit':fitted,
                    'physical_teacher_controls':teacher_controls,'queried_learner_source_controls':manifest['source_raw_controls'],
                    'parent_selection_evaluation_controls':audits[parent.name]['evaluation_control_frames_verified'],
                    'charged_input_controls_lower_bound':charged_controls,'charged_input_batches':charged_batches,'corrective_queries':count,
                    'corrective_labels_are_executed_commands':False,'normalization_bytes_exact':True,
                    'PPO_RNG_bytes_exact':True,'parent_checkpoint_model_sha256':sha(origin/'model.zip')},
                recovery={'B1':{'checkpoint':'resume/B1','parent_batch':parent.name,
                    'parent_checkpoint':origin.name,'lineage_controls':0,'consumed_controls_lower_bound':charged_controls,
                    'retained_optimizer_steps':0,'unknown_unflushed_parent_tail':False}})
            protocol['extra_supervision'].update(dataset_sha256=sha(batch/'measured_demonstrations.npz'),
                supervised_optimizer_steps=previous_steps+steps,epochs=None,history=fitted['history'],
                holdout_scope='NONE: all supplied rows are training. Previous fit-only holdout is no longer a holdout. No independent validation claim.',
                scope='Original measured teacher labels plus counterfactual B3/MPC corrections at measured learner states; weighted actor-only fit. Corrective return placeholders are never value targets.')
            protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            write_json(batch/'protocol.json',protocol);write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
            with registration.open('x',encoding='utf-8') as stream:json.dump({'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')},stream,indent=2)
            print(batch,flush=True);return batch
        except BaseException as error:
            write_json(batch/'failure.json',{'error':str(error),'epoch':time.time(),'new_controls':0,'PPO_updates':0});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--oracle',required=True);p.add_argument('--base-views',required=True)
    p.add_argument('--steps',type=int,default=544);p.add_argument('--weight',type=float,default=20.);p.add_argument('--learning-rate',type=float,default=1e-4)
    a=p.parse_args();prepare(a.parent,a.oracle,a.base_views,a.steps,a.weight,a.learning_rate)

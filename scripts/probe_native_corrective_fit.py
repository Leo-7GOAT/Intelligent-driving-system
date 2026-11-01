"""Matched TRAIN-only actor fits from one real trained B1 endpoint; no driving."""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def require_replay_inputs(current,previous):
    import numpy as np
    for key in ('base_observations','base_actions','base_weights'):
        np.testing.assert_array_equal(current[key],previous[key])


def run(parent,oracle,output,replay_fit=None,require_retention=False):
    parent,oracle,output=map(lambda p:Path(p).resolve(),(parent,oracle,output))
    if any(name.startswith('navstack.') for name in sys.modules):
        raise RuntimeError('Use a fresh interpreter for the exact frozen native runtime')
    runtime=parent/'source';sys.path.insert(0,str(runtime))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    require_finished_parent(parent)
    healthy,reason=resources_ok(ROOT)
    if not healthy:raise RuntimeError(str(reason))
    protocol=read(parent/'protocol.json');folder=parent/'B1';raw=read(folder/'endpoint_raw_audit.json')
    if protocol['methods']!=['B1'] or protocol.get('evaluation_only'):
        raise ValueError('Require a completed actual B1 PPO training parent')
    for name,digest in protocol['source_hashes'].items():
        if sha(runtime/name)!=digest:raise ValueError('Frozen parent source changed:'+name)
    for path,key in ((parent/'protocol.json','protocol_sha256'),(folder/'config.json','config_sha256'),
                     (folder/'summary.json','summary_sha256'),(folder/'evaluation.json','evaluation_sha256')):
        if sha(path)!=raw[key]:raise ValueError('Parent raw audit binding changed')
    manifest=read(oracle/'manifest.json')
    if (manifest['raw_audit_sha256']!=sha(folder/'endpoint_raw_audit.json')
            or manifest.get('coverage_profile') not in ('first_episode_full_trajectory_stride80_v1','first_three_episodes_full_trajectory_stride40_v1')
            or manifest['new_controls'] or manifest['PPO_updates'] or manifest['supervised_updates']
            or manifest['labels_are_executed_commands'] is not False):
        raise ValueError('Require registered, counterfactual, TRAIN-only trajectory coverage')
    for name,key in [('corrective_training_queries.npz','dataset_sha256'),('queries.jsonl','query_log_sha256'),
                     ('executed_query_script.py','script_sha256'),('executed_history_script.py','history_source_sha256')]:
        if sha(oracle/name)!=manifest[key]:raise ValueError('Oracle provenance changed:'+name)
    records=[json.loads(line) for line in (oracle/'queries.jsonl').read_text(encoding='utf-8').splitlines()]
    available=[r for r in records if r['label_available']]
    keep=np.array([r.get('oracle_mode')=='reference' and r.get('controller_status')=='solved' for r in available])
    if len(available)!=manifest['available_labels'] or keep.sum()<8:
        raise ValueError('Insufficient solved-reference coverage')
    with np.load(oracle/'corrective_training_queries.npz',allow_pickle=False) as data:
        query={k:data[k].copy() for k in ('observations','actions','groups')}
    np.testing.assert_array_equal(query['actions'],np.array([r['label'] for r in available],dtype=np.float32))
    base_path=parent/'measured_demonstrations.npz'
    if sha(base_path)!=protocol['extra_supervision']['dataset_sha256']:
        raise ValueError('Original teacher data changed')
    with np.load(base_path,allow_pickle=False) as data:
        mask=data['train_mask']
        if mask.dtype!=bool:raise ValueError('Require explicit Boolean teacher TRAIN mask')
        base={k:data[k][mask].copy() for k in ('observations','actions')}
        weights=data['weights'][mask].copy() if 'weights' in data.files else np.ones(int(mask.sum()))
        base_groups=data['groups'][mask].copy() if require_retention else None
    latest=read(folder/'latest_checkpoint.json')['folder']
    if Path(latest).name!=latest:raise ValueError('Invalid checkpoint path')
    origin=folder/latest;metadata=verified_checkpoint_metadata(origin)
    if metadata['model_sha256']!=sha(folder/'model.zip'):
        raise ValueError('Training checkpoint and endpoint differ')
    fit_path=ROOT/'navstack/corrective_imitation.py'
    spec=importlib.util.spec_from_file_location('navstack._trained_corrective_fit',fit_path)
    fitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(fitter)
    reused=None
    if replay_fit is not None:
        replay_fit=Path(replay_fit).resolve();previous=read(replay_fit/'complete.json')
        if (previous['parent']!=parent.name or previous['parent_model_sha256']!=metadata['model_sha256']
                or previous['steps_each']!=512 or previous['learning_rate']!=1e-4
                or previous['fit_source_sha256']!=sha(fit_path)
                or sha(replay_fit/'fit_inputs.npz')!=previous['inputs_sha256']):
            raise ValueError('Prior replay has different parent,data or fitting implementation')
        with np.load(replay_fit/'fit_inputs.npz',allow_pickle=False) as d:
            require_replay_inputs({'base_observations':base['observations'],'base_actions':base['actions'],'base_weights':weights},d)
        reused=next(r for r in previous['results'] if r['arm']=='replay')
        if (sha(replay_fit/'replay/model.zip')!=reused['model_sha256']
                or sha(replay_fit/'replay/normalization.pkl')!=metadata['normalization_sha256']):
            raise ValueError('Completed replay artifact changed')
    if require_retention and reused is None:raise ValueError('Retention gate needs the exact existing replay reference')
    gate_module=None
    if require_retention:
        spec=importlib.util.spec_from_file_location('exact_retention_gate',ROOT/'scripts/probe_native_retained_correction.py')
        gate_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate_module)
    torch.set_num_threads(1);output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/'executed_probe.py');shutil.copy2(fit_path,output/'executed_fit.py')
    shutil.copytree(oracle,output/'oracle')
    np.savez_compressed(output/'fit_inputs.npz',base_observations=base['observations'],base_actions=base['actions'],
                        base_weights=weights,query_observations=query['observations'],query_actions=query['actions'],keep=keep)
    started=time.monotonic();deadline=started+600
    registration={'scope':__doc__,'parent':parent.name,'seed':protocol['seed'],'parent_model_sha256':metadata['model_sha256'],
        'parent_raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'oracle_manifest_sha256':sha(oracle/'manifest.json'),
        'steps_each':512,'learning_rate':1e-4,'correction_row_weight':20.,'teacher_TRAIN_rows':len(weights),
        'solved_reference_rows':int(keep.sum()),'excluded_wait_or_failed_rows':int((~keep).sum()),
        'selection':'All solved-reference labels from the registered outcome-independent TRAIN episode profile. All wait/brake-only queries excluded; no evaluation data.',
        'oracle_coverage_profile':manifest['coverage_profile'],'selected_training_episodes':manifest.get('selected_training_episodes'),
        'new_controls':0,'PPO_updates':0,'wall_seconds':600,'automatic_retries':0,
        'inputs_sha256':sha(output/'fit_inputs.npz'),'script_sha256':sha(Path(__file__)),'fit_source_sha256':sha(fit_path),
        'limits':'TRAIN loss is diagnostic only. Compare all four known basic cases in CARLA before adoption; no baseline qualification or generalization claim.'}
    registration.update(replay_fit_reused=reused is not None,new_supervised_updates=512 if reused is not None else 1024,
        reused_supervised_updates=512 if reused is not None else 0,require_basic_retention=require_retention)
    if reused is not None:registration['reused_replay_manifest_sha256']=sha(replay_fit/'complete.json')
    if require_retention:
        shutil.copy2(ROOT/'scripts/probe_native_retained_correction.py',output/'executed_retention_gate.py')
        registration['retention_gate_source_sha256']=sha(output/'executed_retention_gate.py')
    atomic_json(output/'registration.json',registration)
    results=[]
    try:
        for arm in ('replay','trajectory_correction'):
            model=load_ppo(origin/'model.zip',device='cpu')
            with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
            norm.training=False
            with (origin/'resume_state.pkl').open('rb') as stream:rng=pickle.load(stream)
            torch.set_rng_state(rng['torch'])
            def diagnose():
                model.policy.set_training_mode(False)
                with torch.no_grad():
                    prediction=model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(query['observations']))).distribution.mean.cpu().numpy()
                return {'reference_action_MSE':((prediction[keep]-query['actions'][keep])**2).mean(axis=0).tolist(),
                        'predictions_all_queries':prediction.tolist()}
            before=diagnose();observations=base['observations'];actions=base['actions'];sample_weights=weights
            if arm=='trajectory_correction':
                observations=np.concatenate((observations,query['observations'][keep]))
                actions=np.concatenate((actions,query['actions'][keep]))
                sample_weights=np.r_[sample_weights,np.full(int(keep.sum()),20.)]
            def progress(row):
                okay,why=resources_ok(ROOT)
                if not okay or time.monotonic()>deadline:raise RuntimeError('Bounded fit resource/deadline guard:'+str(why))
                atomic_json(output/'heartbeat.json',{'arm':arm,'epoch':time.time(),**row})
            destination=output/arm
            if arm=='replay' and reused is not None:
                model=load_ppo(replay_fit/'replay/model.zip',device='cpu');fit=deepcopy(reused['fit'])
                fit['reused_fit_manifest_sha256']=sha(replay_fit/'complete.json')
                shutil.copytree(replay_fit/'replay',destination)
            else:
                fit=fitter.fit_trained_corrective_actor(model,norm,observations,actions,sample_weights,
                    steps=512,learning_rate=1e-4,seed=protocol['seed'],callback=progress)
                shutil.copytree(origin,destination);model.save(destination/'model.zip')
            changed={**deepcopy(metadata),'model_sha256':sha(destination/'model.zip'),
                     'additional_actor_fit_steps':512,'auxiliary_fit_version':fit['version'],
                     'note':'Trained PPO endpoint plus512separate actor-only Adam updates;PPO weights changed,optimizer and counters retained;fresh rollout required before PPO'}
            atomic_json(destination/'metadata.json',changed);verified_checkpoint_metadata(destination)
            if sha(destination/'normalization.pkl')!=metadata['normalization_sha256'] or sha(destination/'resume_state.pkl')!=metadata['resume_state_sha256']:
                raise ValueError('Fit changed saved normalization or original continuation RNG bytes')
            results.append({'arm':arm,'fit':fit,'before':before,'after':diagnose(),
                            'model_sha256':changed['model_sha256'],'normalization_sha256':metadata['normalization_sha256']})
            if require_retention:
                model.policy.set_training_mode(False)
                with torch.no_grad():
                    base_prediction=model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(base['observations']))).distribution.mean.cpu().numpy()
                query_prediction=np.asarray(results[-1]['after']['predictions_all_queries'])
                if arm=='replay':
                    reference_base=base_prediction;reference_query=query_prediction
                    fixed=gate_module.retention_gate(reference_base,reference_base,base['actions'],base_groups,
                        reference_query[keep],reference_query[keep],query['actions'][keep])
                    atomic_json(output/'fixed_retention_gate_before_correction.json',fixed)
                else:gate=gate_module.retention_gate(base_prediction,reference_base,base['actions'],base_groups,
                    query_prediction[keep],reference_query[keep],query['actions'][keep])
            atomic_json(output/'completed_arms.json',results)
            print(json.dumps({'arm':arm,'before_MSE':before['reference_action_MSE'],
                              'after_MSE':results[-1]['after']['reference_action_MSE'],'new_controls':0,'PPO_updates':0}),flush=True)
        result={**registration,'results':results,'additional_supervised_steps_total':1024,
                'elapsed_seconds':time.monotonic()-started,'research_acceptance':'UNMET'}
        if require_retention:result['retention_gate']=gate
        atomic_json(output/('rejected.json' if require_retention and not gate['passed'] else 'complete.json'),result);return result
    except BaseException as error:
        atomic_json(output/'failure.json',{'error':repr(error),'completed_arms':[r['arm'] for r in results]});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','oracle','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--replay-fit');parser.add_argument('--require-basic-retention',action='store_true')
    args=parser.parse_args();run(args.parent,args.oracle,args.output,args.replay_fit,args.require_basic_retention)

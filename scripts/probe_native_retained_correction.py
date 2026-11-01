"""One fixed TRAIN retention-weight hypothesis after the rejected99-label fit."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def retention_gate(prediction,reference,labels,groups,query=None,query_reference=None,query_labels=None,*,basic_only=False):
    arrays=[np.asarray(a,dtype=np.float64) for a in (prediction,reference,labels)]
    a,b,y=arrays;groups=np.asarray(groups)
    if not basic_only:
        q,r,z=[np.asarray(v,dtype=np.float64) for v in (query,query_reference,query_labels)];arrays.extend((q,r,z))
    if (a.ndim!=2 or a.shape[1]!=2 or a.shape!=b.shape or a.shape!=y.shape
            or groups.shape!=(len(a),) or not set(range(4)).issubset(set(groups.tolist()))
            or not basic_only and (q.ndim!=2 or q.shape[1]!=2 or not len(q) or q.shape!=r.shape or q.shape!=z.shape)
            or any(not np.isfinite(x).all() for x in arrays)):
        raise ValueError('Require all four basic teacher TRAIN groups and finite paired predictions')
    rows=[]
    for group in range(4):
        mask=groups==group;actual=float(np.square(a[mask]-y[mask]).mean())
        baseline=float(np.square(b[mask]-y[mask]).mean());ceiling=1.25*baseline+1e-5
        rows.append({'group':group,'rows':int(mask.sum()),'MSE':actual,'replay_MSE':baseline,
                     'fixed_ceiling':ceiling,'passed':actual<=ceiling})
    if basic_only:return {'basic_teacher_groups':rows,'query_constraint_applied':False,'passed':all(r['passed'] for r in rows)}
    error=float(np.square(q-z).mean());before=float(np.square(r-z).mean())
    return {'basic_teacher_groups':rows,'query_MSE':error,'replay_query_MSE':before,
            'query_ceiling':.5*before,'passed':all(r['passed'] for r in rows) and error<=.5*before}


def backtrack_actor(actor,accepted,proposed,check):
    import torch
    for scale in (1.,.5,.25,.125,.0625,0.):
        with torch.no_grad():
            for parameter,previous,candidate in zip(actor,accepted,proposed):
                parameter.copy_(previous+scale*(candidate-previous))
        metrics=check()
        if all(r['passed'] for r in metrics['basic_teacher_groups']):
            return scale,metrics
    raise ValueError('Last accepted TRAINretention state no longer satisfies its fixed gate')


def run(parent_fit,output,replay_parent=None,executed_only=False):
    parent_fit,output=Path(parent_fit).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous fixed hypothesis')
    old=json.loads((parent_fit/'complete.json').read_text(encoding='utf-8'))
    parent=ROOT/'runs'/old['parent'];runtime=parent/'source';sys.path.insert(0,str(runtime))
    import torch
    from navstack.independent_clip_ppo import load_ppo,actor_critic_parameters
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    require_finished_parent(parent)
    p=read(parent/'protocol.json')
    if (old['solved_reference_rows']!=99 or old['steps_each']!=512
            or old['oracle_coverage_profile']!='first_three_episodes_full_trajectory_stride40_v1'
            or sha(parent_fit/'fit_inputs.npz')!=old['inputs_sha256']
            or sha(parent/'B1/endpoint_raw_audit.json')!=old['parent_raw_audit_sha256']):
        raise ValueError('Require the exact rejected99-label training fit')
    for name,digest in p['source_hashes'].items():
        if name.startswith('navstack/') and sha(runtime/name)!=digest:
            raise ValueError('Frozen policy or low-level implementation changed')
    fit_path=ROOT/'navstack/corrective_imitation.py'
    spec=importlib.util.spec_from_file_location('navstack._retained_fit',fit_path)
    fitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(fitter)
    with np.load(parent_fit/'fit_inputs.npz',allow_pickle=False) as d:inputs={k:d[k].copy() for k in d.files}
    teacher=parent/'measured_demonstrations.npz'
    if sha(teacher)!=p['extra_supervision']['dataset_sha256']:raise ValueError('Teacher dataset changed')
    with np.load(teacher,allow_pickle=False) as d:
        mask=d['train_mask'];groups=d['groups'][mask].copy()
        np.testing.assert_array_equal(inputs['base_observations'],d['observations'][mask])
        np.testing.assert_array_equal(inputs['base_actions'],d['actions'][mask])
    replay=next(r for r in old['results'] if r['arm']=='replay')
    if sha(parent_fit/'replay/model.zip')!=replay['model_sha256']:raise ValueError('Replay reference changed')
    training_parent=parent
    hard=replay_parent is not None
    if hard and executed_only:raise ValueError('Keep the two hypotheses separate')
    if hard:
        parent=Path(replay_parent).resolve();require_finished_parent(parent)
        if (read(parent/'protocol.json')['methods']!=['B1']
                or read(parent/'protocol.json')['source_hashes']!=p['source_hashes']
                or read(parent/'B1/evaluation.json')['model_sha256']!=sha(parent/'resume/B1/model.zip')):
            raise ValueError('Require the completed unchanged replay endpoint')
        origin=parent/'resume/B1'
    else:origin=parent/'B1'/read(parent/'B1/latest_checkpoint.json')['folder']
    metadata=verified_checkpoint_metadata(origin)
    if not hard and metadata['model_sha256']!=old['parent_model_sha256']:raise ValueError('Different trained parent')
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu')
    reference=load_ppo(parent_fit/'replay/model.zip',device='cpu')
    if hard and (not fitter._exact(model.policy.state_dict(),reference.policy.state_dict())
            or not fitter._exact(model.policy.optimizer.state_dict(),reference.policy.optimizer.state_dict())
            or (model.num_timesteps,model._n_updates)!=(reference.num_timesteps,reference._n_updates)):
        raise ValueError('Replay checkpoint has different actor,critic,PPO state or counters')
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    if sha(origin/'normalization.pkl')!=replay['normalization_sha256']:raise ValueError('Different normalization')
    norm.training=False;keep=inputs['keep'];base=inputs['base_observations'];labels=inputs['base_actions']
    queries=inputs['query_observations'];query_labels=inputs['query_actions']
    def predict(policy,observations):
        mode=policy.policy.training;policy.policy.set_training_mode(False);x=norm.normalize_obs(observations)
        try:
            with torch.no_grad():return np.concatenate([policy.policy.get_distribution(torch.as_tensor(x[i:i+256])).distribution.mean.cpu().numpy() for i in range(0,len(x),256)])
        finally:policy.policy.set_training_mode(mode)
    reference_base=predict(reference,base);reference_query=predict(reference,queries)
    # Compute and lock all ceilings before the new fit. There is no best-step selection.
    fixed_gate=retention_gate(reference_base,reference_base,labels,groups,reference_query[keep],reference_query[keep],query_labels[keep],basic_only=executed_only)
    output.mkdir(parents=True);shutil.copy2(__file__,output/'executed_probe.py')
    shutil.copy2(ROOT/'navstack/corrective_imitation.py',output/'executed_fit.py')
    shutil.copytree(parent_fit/'oracle',output/'oracle');shutil.copytree(parent_fit/'replay',output/'replay')
    weights=inputs['base_weights'].copy()
    if not hard and not executed_only:weights[np.isin(groups,[0,1,2,3])]*=8
    physical=(groups>=0)&(groups<1024)
    if executed_only and (set(groups[physical].tolist())!=set(range(12)) or not np.all(inputs['base_weights'][physical]==1)):
        raise ValueError('Require the original twelve measured teacher groups with unit weights')
    np.savez_compressed(output/'fit_inputs.npz',**{**inputs,'base_weights':weights,
        'original_base_weights':inputs['base_weights'],'base_groups':groups,
        'fit_train_mask':physical if executed_only else np.ones(len(groups),dtype=bool)})
    registration={k:v for k,v in old.items() if k not in ('results','elapsed_seconds')}
    registration.update(scope=__doc__,parent_fit_manifest_sha256=sha(parent_fit/'complete.json'),
        basic_teacher_weight_multiplier=1 if hard or executed_only else 8,retention_group_ids=[0,1,2,3],
        fixed_gate_before_fit=fixed_gate,steps_each=512,new_supervised_updates=512,
        reused_supervised_updates=512,replay_fit_reused=True,
        replay_fit_inputs_sha256=old['inputs_sha256'],inputs_sha256=sha(output/'fit_inputs.npz'),
        script_sha256=sha(output/'executed_probe.py'),fit_source_sha256=sha(output/'executed_fit.py'),
        selection='One final512step endpoint;TRAINper-group retention gate before any native probe. No new queries,holdout labels,checkpoint selection or retries.')
    if hard:
        registration.update(scope='TRAIN-only constrained actor fine-tuning from the completed replay policy;no PPO or driving',
            parent=parent.name,parent_model_sha256=metadata['model_sha256'],
            parent_raw_audit_sha256=sha(parent/'B1/endpoint_raw_audit.json'),
            teacher_training_parent=training_parent.name,
            retained_correction_profile='basic_teacher_group_backtracking_v1',
            additional_supervised_steps_total=512,reused_supervised_updates=0,
            backtracking_scales=[1.,.5,.25,.125,.0625,0.],
            selection='Fixed512Adam proposals;at callbacks1,64,...512project actor displacement toward last accepted state. Every basic TRAINgroup must satisfy its prelocked ceiling. Final query reduction gate;no evaluation-based selection.')
    if executed_only:
        registration.update(scope='TRAIN-only executed-teacher data repair from the same actual PPOparent;no new counterfactual labels',
            executed_teacher_only_profile='verified_measured_groups0to11_v1',additional_supervised_steps_total=512,
            reused_supervised_updates=0,teacher_TRAIN_rows=int(physical.sum()),
            excluded_counterfactual_rows=int((~physical).sum()),
            excluded_counterfactual_weight=float(inputs['base_weights'][~physical].sum()),
            solved_reference_rows=0,query_rows_used=0,
            selection='Fixed teacher provenance mask groups0..11,all its TRAINrows,512actor Adam steps. Same prelocked four-group retention ceilings;no rejected-query fitting criterion or heldout data.')
    atomic_json(output/'registration.json',registration);started=time.monotonic()
    actor,_=actor_critic_parameters(model.policy)
    accepted=[v.detach().clone() for v in actor];blocks=[]
    def progress(row):
        nonlocal accepted
        okay,reason=resources_ok(ROOT)
        if not okay or time.monotonic()-started>300:raise RuntimeError('Bounded fit stopped:'+str(reason))
        if hard:
            proposed=[v.detach().clone() for v in actor]
            def check():return retention_gate(predict(model,base),reference_base,labels,groups,
                predict(model,queries)[keep],reference_query[keep],query_labels[keep])
            scale,metrics=backtrack_actor(actor,accepted,proposed,check)
            accepted=[v.detach().clone() for v in actor]
            blocks.append({'Adam_proposals':row['additional_supervised_steps'],'accepted_displacement_scale':scale,
                'basic_teacher_groups':metrics['basic_teacher_groups'],'query_MSE':metrics['query_MSE']})
        atomic_json(output/'heartbeat.json',{'epoch':time.time(),**row})
    fitted=fitter.fit_trained_corrective_actor(model,norm,base[physical] if executed_only else np.concatenate((base,queries[keep])),
        labels[physical] if executed_only else np.concatenate((labels,query_labels[keep])),
        weights[physical] if executed_only else np.r_[weights,np.full(int(keep.sum()),20.)],
        steps=512,learning_rate=1e-4,seed=p['seed'],callback=progress)
    prediction=predict(model,queries);gate=retention_gate(predict(model,base),reference_base,labels,groups,
        prediction[keep],reference_query[keep],query_labels[keep],basic_only=executed_only)
    arm='teacher_only_cleaning' if executed_only else 'trajectory_correction'
    destination=output/arm;shutil.copytree(origin,destination);model.save(destination/'model.zip')
    inherited_fit_steps=metadata.get('additional_actor_fit_steps',0) if hard else 0
    metadata.update(model_sha256=sha(destination/'model.zip'),additional_actor_fit_steps=512,
        inherited_actor_fit_steps=inherited_fit_steps,cumulative_actor_fit_steps=inherited_fit_steps+512,
        auxiliary_fit_version=fitted['version'],retention_weight_profile='verified_measured_groups0to11_v1' if executed_only else 'basic_teacher_group_backtracking_v1' if hard else 'basic_groups0to3_weight8_v1')
    atomic_json(destination/'metadata.json',metadata);verified_checkpoint_metadata(destination)
    result={**registration,'results':([] if hard or executed_only else [deepcopy(replay)])+[{'arm':arm,'fit':fitted,
        'before':deepcopy(replay['after']) if hard else next(r for r in old['results'] if r['arm']=='trajectory_correction')['before'],
        'after':{'reference_action_MSE':np.square(prediction[keep]-query_labels[keep]).mean(axis=0).tolist(),
                 'predictions_all_queries':prediction.tolist()},'model_sha256':metadata['model_sha256'],
        'normalization_sha256':metadata['normalization_sha256']}],
        'retention_gate':gate,'actor_backtracking_blocks':blocks,
        'auxiliary_Adam_state_advanced_for_every_proposal':hard,
        'elapsed_seconds':time.monotonic()-started,'research_acceptance':'UNMET'}
    # A rejected fit cannot be consumed by the native probe preparer.
    atomic_json(output/('complete.json' if gate['passed'] else 'rejected.json'),result)
    print(json.dumps({'gate':gate,'new_supervised_updates':512,'new_controls':0,'PPO_updates':0}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-fit',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--replay-parent',help='Explicit completed replay endpoint for fixed TRAIN-group backtracking')
    parser.add_argument('--executed-only',action='store_true',help='Fit only the original physically executed teacher TRAINgroups;exclude all counterfactual labels')
    args=parser.parse_args();run(args.parent_fit,args.output,args.replay_parent,args.executed_only)

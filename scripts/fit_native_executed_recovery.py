"""One fixed actor fit using audited, actually executed TRAIN teacher controls."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def executed_recovery_arrays(data, expected_rows):
    keep=np.asarray(data['fit_train_mask'])
    sources=np.asarray(data['teacher_executed'])
    if (keep.dtype!=np.bool_ or sources.dtype!=np.bool_ or keep.shape!=sources.shape
            or np.any(keep&~sources) or int(keep.sum())!=expected_rows):
        raise ValueError('Counterfactual proposals cannot enter the fit')
    extra=data['observations'][keep].copy();labels=data['actions'][keep].copy();groups=data['groups'][keep].copy()
    if not len(extra) or set(groups.tolist())!=set(range(2048,2052)):
        raise ValueError('Require executed teacher labels from every registered TRAIN road')
    return extra,labels,groups


def episode_balanced_weights(groups):
    groups=np.asarray(groups)
    if groups.ndim!=1 or not len(groups):raise ValueError('Require nonempty episode groups')
    weights=np.zeros(len(groups),dtype=np.float64)
    for group in np.unique(groups):weights[groups==group]=1/np.count_nonzero(groups==group)
    return weights


def actor_fit_lineage(metadata,new_steps):
    previous=metadata.get('cumulative_actor_fit_steps',metadata.get('additional_actor_fit_steps',0))
    if type(previous) is not int or previous<0 or type(new_steps) is not int or new_steps<=0:
        raise ValueError('Require explicit nonnegative inherited and positive new actor-update counts')
    return {'inherited_actor_fit_steps':previous,'additional_actor_fit_steps':new_steps,
            'cumulative_actor_fit_steps':previous+new_steps}


def require_probe_fit(result):
    unprojected=result.get('fit_profile')=='executed_original_TRAIN_no_projection_v1'
    if result['new_supervised_Adam_proposals']!=512:
        raise ValueError('Require the fixed512proposal endpoint')
    if unprojected:
        if (result.get('route_profile')!='original_TRAIN4_v1'
                or result.get('TRAIN_retention_screen_applied') is not False
                or result.get('retention_reference_model_sha256')!='4f20570844ff17da318c601893be8523ddf504193d21775f9553a3d5c8530ca0'):
            raise ValueError('Require the explicitly registered matched projection ablation')
    elif not result['retention_gate']['passed']:
        raise ValueError('Require fixed endpoint TRAINretention gate')
    return unprojected


def run(batch,output,no_projection=False):
    batch,output=Path(batch).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior fixed recovery fit')
    sys.path.insert(0,str(batch/'source'))
    import torch
    from navstack.independent_clip_ppo import load_ppo,actor_critic_parameters
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    require_finished_parent(batch)
    p=read(batch/'protocol.json');collection=read(batch/'B1/collection.json');audit=read(batch/'B1/collection_raw_audit.json')
    if (audit['collection_result_sha256']!=sha(batch/'B1/collection.json')
            or audit['protocol_sha256']!=sha(batch/'protocol.json')
            or not audit['all_measured_observations_bit_exact'] or not audit['all_learner_proposals_bit_exact']
            or audit['dataset_sha256']!=sha(batch/'B1/collection.npz') or audit['not_policy_evaluation'] is not True):
        raise ValueError('Require complete native recovery audit before fitting')
    teacher=batch/'measured_demonstrations.npz'
    if sha(teacher)!=p['extra_supervision']['dataset_sha256']:raise ValueError('Original teacher data changed')
    with np.load(teacher,allow_pickle=False) as d:
        keep=d['train_mask']&(d['groups']>=0)&(d['groups']<1024)
        base=d['observations'][keep].copy();labels=d['actions'][keep].copy();groups=d['groups'][keep].copy()
        if set(groups.tolist())!=set(range(12)) or len(base)!=3467:raise ValueError('Require all original physically executed teacher TRAIN rows')
    with np.load(batch/'B1/collection.npz',allow_pickle=False) as d:
        extra,extra_labels,extra_groups=executed_recovery_arrays(d,audit['eligible_executed_teacher_labels'])
    origin=batch/'resume/B1';metadata=verified_checkpoint_metadata(origin)
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu')
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False
    modules={}
    for key,path in [('fitter',ROOT/'navstack/corrective_imitation.py'),('gate',ROOT/'scripts/probe_native_retained_correction.py')]:
        spec=importlib.util.spec_from_file_location('navstack._executed_recovery_'+key,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules[key]=module
    def predict(obs):
        mode=model.policy.training;model.policy.set_training_mode(False)
        try:
            x=norm.normalize_obs(obs)
            with torch.no_grad():return np.concatenate([model.policy.get_distribution(torch.as_tensor(x[i:i+256])).distribution.mean.cpu().numpy() for i in range(0,len(x),256)])
        finally:model.policy.set_training_mode(mode)
    reference=predict(base);extra_before=predict(extra)
    original_profile=p['native_recovery_collection'].get('route_profile')=='original_TRAIN4_v1'
    if no_projection and not original_profile:
        raise ValueError('Projection ablation is locked to the audited all4original-TRAINcollection')
    reference_sha=metadata['model_sha256']
    if original_profile:
        # Keep the first attempt's exact ceilings. Do not relax them after a
        # rejected model merely because it is the new collection policy.
        reference_path=ROOT/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay/model.zip'
        reference_sha=sha(reference_path)
        if reference_sha!='4f20570844ff17da318c601893be8523ddf504193d21775f9553a3d5c8530ca0':
            raise ValueError('Original fixed retention reference changed')
        replay=load_ppo(reference_path,device='cpu');replay.policy.set_training_mode(False)
        x=norm.normalize_obs(base)
        with torch.no_grad():reference=np.concatenate([replay.policy.get_distribution(torch.as_tensor(x[i:i+256])).distribution.mean.cpu().numpy() for i in range(0,len(x),256)])
    all_groups=np.r_[groups,extra_groups]
    # Equal total probability per original teacher or new collection episode.
    # All eligible controls are included, including the timed-out episode.
    weights=episode_balanced_weights(all_groups)
    output.mkdir(parents=True)
    for source,name in [(Path(__file__),'executed_fit.py'),(ROOT/'navstack/corrective_imitation.py','executed_actor_helper.py'),
                        (ROOT/'scripts/probe_native_retained_correction.py','executed_retention_helper.py')]:shutil.copy2(source,output/name)
    np.savez_compressed(output/'fit_inputs.npz',observations=np.concatenate((base,extra)),actions=np.concatenate((labels,extra_labels)),groups=all_groups,weights=weights)
    fixed=modules['gate'].retention_gate(reference,reference,labels,groups,basic_only=True)
    registration={'scope':__doc__,'collection_batch':batch.name,'collection_audit_sha256':sha(batch/'B1/collection_raw_audit.json'),
        'source_model_sha256':metadata['model_sha256'],'source_normalization_sha256':metadata['normalization_sha256'],
        'original_executed_teacher_TRAIN_rows':len(base),'new_executed_teacher_TRAIN_rows':len(extra),
        'excluded_counterfactual_collection_rows':collection['collection_controls']-len(extra),
        'new_PPO_controls':0,'new_PPO_updates':0,'steps':512,'learning_rate':1e-4,'seed':22,
        'sampling':'Equal total probability per each of12original and4new TRAINepisodes;all eligible executed controls,no outcome filter.',
        'selection':'One final512proposal endpoint,retaining all4prelocked basic TRAINceilings;no validation-based choice or retries.',
        'basic_gate_before_fit':fixed,'backtracking_scales':[1.,.5,.25,.125,.0625,0.],
        'fit_inputs_sha256':sha(output/'fit_inputs.npz'),
        'source_hashes':{name:sha(output/name) for name in ('executed_fit.py','executed_actor_helper.py','executed_retention_helper.py')}}
    if original_profile:
        registration.update(route_profile='original_TRAIN4_v1',retention_reference_model_sha256=reference_sha,
            fixed_retention_ceilings_unchanged_from_first_attempt=True,
            ancestry='Starts from the rejected new-road fit only to test executed original-TRAINrecovery;no promotion implied,no extra validation labels.')
    if no_projection:
        registration.update(fit_profile='executed_original_TRAIN_no_projection_v1',TRAIN_retention_screen_applied=False,
            scope='Matched512step actor fit on the SAMEactual original-TRAINteacher data,parent,seed and weights;only displacement projection removed. Driving qualification unchanged.',
            selection='One final512step endpoint,all valid executed teacher labels,no validation selection. Old TRAINMSEceilings reported diagnostically,not the driving qualification.',
            backtracking_scales=[],matched_projected_fit='runs/B1_seed22_original_TRAIN_recovery_fit_20261101')
    atomic_json(output/'registration.json',registration)
    actor,_=actor_critic_parameters(model.policy);accepted=[v.detach().clone() for v in actor];blocks=[];started=time.monotonic()
    def progress(row):
        nonlocal accepted
        okay,reason=resources_ok(ROOT)
        if not okay or time.monotonic()-started>300:raise RuntimeError('Bounded recovery fit stopped:'+str(reason))
        proposed=[v.detach().clone() for v in actor]
        check=lambda:modules['gate'].retention_gate(predict(base),reference,labels,groups,basic_only=True)
        if no_projection:scale,gate=1.,check()
        else:scale,gate=modules['gate'].backtrack_actor(actor,accepted,proposed,check)
        accepted=[v.detach().clone() for v in actor]
        blocks.append({'Adam_proposals':row['additional_supervised_steps'],'accepted_scale':scale,'gate':gate})
        atomic_json(output/'heartbeat.json',{'epoch':time.time(),**row})
    fit=modules['fitter'].fit_trained_corrective_actor(model,norm,np.concatenate((base,extra)),np.concatenate((labels,extra_labels)),
        weights,steps=512,learning_rate=1e-4,seed=22,callback=progress)
    gate=modules['gate'].retention_gate(predict(base),reference,labels,groups,basic_only=True)
    destination=output/'checkpoint';shutil.copytree(origin,destination);model.save(destination/'model.zip')
    metadata.update(model_sha256=sha(destination/'model.zip'),**actor_fit_lineage(metadata,512),
        auxiliary_fit_version=fit['version'],auxiliary_fit_profile='executed_original_TRAIN_no_projection_v1' if no_projection else 'executed_native_TRAIN_recovery_group_balanced_v1',
        note='Recorded trained parent plus512new separate actor Adam proposals;projection profile is explicit. PPOoptimizer,critic,counters and RMS retained;inherited and cumulative actor costs are explicit.')
    atomic_json(destination/'metadata.json',metadata);verified_checkpoint_metadata(destination)
    result={**registration,'fit':fit,'retention_gate':gate,'backtracking_blocks':blocks,
        'new_supervised_Adam_proposals':512,'new_controls_during_fit':0,'model_sha256':metadata['model_sha256'],
        'normalization_sha256':metadata['normalization_sha256'],'elapsed_seconds':time.monotonic()-started,
        'new_TRAIN_teacher_MSE_before':float(np.square(extra_before-extra_labels).mean()),
        'new_TRAIN_teacher_MSE_after':float(np.square(predict(extra)-extra_labels).mean()),'research_acceptance':'UNMET'}
    atomic_json(output/('complete.json' if no_projection or gate['passed'] else 'rejected.json'),result)
    print(json.dumps({k:result[k] for k in ('model_sha256','new_TRAIN_teacher_MSE_before','new_TRAIN_teacher_MSE_after','retention_gate')}),flush=True)


def prepare_probe(fit):
    sys.path.insert(0,str(ROOT))
    from navstack.research import atomic_json
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    fit=Path(fit).resolve();result=read(fit/'complete.json');parent=ROOT/'runs'/result['collection_batch']
    if (fit/'probe_registration.json').exists():raise FileExistsError('Preserve prior recovery probe')
    require_finished_parent(parent);old=read(parent/'protocol.json')
    unprojected=require_probe_fit(result)
    for name,digest in result['source_hashes'].items():
        if sha(fit/name)!=digest:raise ValueError('Executed fit source changed')
    if sha(fit/'fit_inputs.npz')!=result['fit_inputs_sha256']:raise ValueError('Fit inputs changed')
    metadata=verified_checkpoint_metadata(fit/'checkpoint')
    if metadata['model_sha256']!=result['model_sha256']:raise ValueError('Fitted checkpoint changed')
    original=ROOT/'runs/20261101T011350364179Z_native_B1_basic_lr_extension'
    require_finished_parent(original)
    if any(sha(parent/'source'/name)!=digest for name,digest in old['source_hashes'].items()):raise ValueError('Frozen collection source changed')
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_recovery_pure_policy_probe')
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(fit/'checkpoint',batch/'resume/B1')
    shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    evidence=batch/'fit_evidence';evidence.mkdir()
    for name in ('registration.json','complete.json','fit_inputs.npz','executed_fit.py','executed_actor_helper.py','executed_retention_helper.py'):
        shutil.copy2(fit/name,evidence/name)
    p=json.loads(json.dumps(old));p.pop('native_recovery_collection');now=time.time()
    p.update(routes=read(original/'protocol.json')['routes'],native_case_manifest=read(original/'protocol.json')['native_case_manifest'],
        deterministic_training_route_evaluation=True,created_epoch=now,deadline_epoch=now+1800,
        scope='Pure unchanged B1model.predict after executed TRAINrecovery actor fit;all4original basic roads,zero teacher runtime or added PPO,not final confirmation.',
        resource_limits={**old['resource_limits'],'wall_seconds':1800},
        executed_native_recovery_actor_fit={'complete_sha256':sha(fit/'complete.json'),'new_supervised_Adam_proposals':512,
            'new_TRAIN_collection_controls':read(parent/'B1/collection.json')['collection_controls'],
            'teacher_in_policy_evaluation':False,'basic_gate':'At least3/4successANDzero collision/bounds;unchanged.'})
    p['recovery']['B1'].update(parent_batch=parent.name,parent_checkpoint='resume/B1',
        consumed_controls_lower_bound=old['recovery']['B1']['consumed_controls_lower_bound']+read(parent/'B1/collection.json')['collection_controls'])
    p['extra_supervision']['supervised_optimizer_steps']+=512
    if unprojected:
        control=read(ROOT/'runs/B1_seed22_original_TRAIN_recovery_fit_20261101/probe_registration.json')
        completed=ROOT/'runs'/control['batch'];require_finished_parent(completed)
        cost=read(completed/'B1/endpoint_raw_audit.json')['total_applied_controls_verified']
        p['recovery']['B1']['consumed_controls_lower_bound']+=cost
        p['executed_native_recovery_actor_fit'].update(fit_profile=result['fit_profile'],TRAIN_retention_screen_applied=False,
            matched_projected_probe=completed.name,other_arm_probe_controls_charged=cost,
            driving_gate_unchanged=True)
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
    atomic_json(fit/'probe_registration.json',{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')})
    print(batch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch');p.add_argument('--output');p.add_argument('--prepare-probe')
    p.add_argument('--no-projection',action='store_true',help='Explicit matched actual-teacher projection ablation;not a driving-gate relaxation')
    args=p.parse_args()
    if args.prepare_probe:prepare_probe(args.prepare_probe)
    elif args.batch and args.output:run(args.batch,args.output,args.no_projection)
    else:p.error('Require --batch/--output or --prepare-probe')

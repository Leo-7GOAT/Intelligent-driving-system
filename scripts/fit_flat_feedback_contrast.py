"""Fixed actor fit of pose responses, preserving the prior measured-state means."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def exact(left,right):
    import torch
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(exact(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return type(left) is type(right) and len(left)==len(right) and all(exact(a,b) for a,b in zip(left,right))
    return left==right


def run(output,continue_fit=None,steps=512):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve the single registered contrastive-feedback fit')
    if (continue_fit is None and steps!=512) or (continue_fit is not None and steps!=1024):
        raise ValueError('Only the first512step fit or declared1024step continuation is supported')
    parent=ROOT/'runs/20261101T011350364179Z_native_B1_basic_lr_extension'
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Fresh frozen runtime required')
    sys.path.insert(0,str(parent/'source'))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo,actor_critic_parameters
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    diagnostic=ROOT/'runs/native_B1_feedback_gain_queries_20261101.json';d=read(diagnostic)
    queries=ROOT/'runs'/d['query_dataset']['name']
    if sha(queries)!=d['query_dataset']['sha256']:raise ValueError('Pose queries changed')
    teacher=ROOT/'runs'/d['teacher_batch'];raw=read(teacher/'raw_audit.json')
    paths=list((teacher/'infrastructure').glob('*_responses.jsonl.gz'))
    if len(paths)!=1 or sha(paths[0])!=raw['bridge_response_sha256']:raise ValueError('Teacher raw inventory differs')
    acknowledged=set()
    with gzip.open(paths[0],'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line).get('response',{})
            if 'vehicle_control' not in r:continue
            if (all(abs(r['vehicle_control'][k]-r['applied_vehicle_control'][k])<=1e-5 for k in ('throttle','brake','steer'))
                    and r['applied_vehicle_control']['manual_gear_shift'] is True
                    and r['applied_vehicle_control']['gear']==1 and r['applied_vehicle_control']['hand_brake'] is False):
                acknowledged.add((r['episode_id'],r['frame']))
    with gzip.open(teacher/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    identities={(r['case_id'],t['backend_frame']):t['backend_evidence']['episode_id'] for r in rows for t in r['info']['trace']}
    with np.load(queries,allow_pickle=False) as arrays:
        qobs=arrays['observations'].copy();mpc=arrays['MPC_steerings'].copy()
        cases=arrays['case_ids'].copy();frames=arrays['frames'].copy();eps=arrays['epsilons'].copy()
    mask=np.asarray([all((identities[int(case),int(frame)],int(frame)-j) in acknowledged for j in (0,1)) for case,frame in zip(cases,frames)])
    qobs,mpc,cases,eps=qobs[mask],mpc[mask],cases[mask],eps[mask]
    if set(cases.tolist())!=set(range(4)):raise ValueError('Need qualified query states on all4TRAIN roads')
    cfg=EnvConfig.from_dict(read(parent/'B1/config.json')['env'])
    origin=ROOT/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay';metadata=verified_checkpoint_metadata(origin)
    if metadata['model_sha256']!=d['model_sha256']:raise ValueError('Query policy differs')
    with np.load(parent/'measured_demonstrations.npz',allow_pickle=False) as arrays:
        keep=arrays['train_mask']&(arrays['groups']>=0)&(arrays['groups']<1024)
        obs=arrays['observations'][keep].copy();labels=arrays['actions'][keep].copy();groups=arrays['groups'][keep].copy()
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu');model.policy.set_training_mode(False)
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False;stats=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),float(norm.obs_rms.count))
    x=torch.as_tensor(norm.normalize_obs(obs));qx=torch.as_tensor(norm.normalize_obs(qobs.reshape(-1,4705))).reshape(-1,3,4705)
    def predict(z):return model.policy.get_distribution(z).distribution.mean
    with torch.no_grad():
        anchors=torch.cat([predict(x[i:i+256]) for i in range(0,len(x),256)]).clone()
        initial=predict(qx.flatten(0,1)).reshape(-1,3,2).clone()
    continuation=None
    if continue_fit is not None:
        continue_fit=Path(continue_fit).resolve();continuation=read(continue_fit/'rejected.json')
        if (continuation['steps']!=512 or continuation['source_model_sha256']!=metadata['model_sha256']
                or continuation['query_sha256']!=sha(queries)):
            raise ValueError('Continue only the preserved512step endpoint on identical TRAINinputs')
        metadata=verified_checkpoint_metadata(continue_fit/'checkpoint')
        if metadata['model_sha256']!=continuation['model_sha256']:raise ValueError('Continuation model changed')
        model=load_ppo(continue_fit/'checkpoint/model.zip',device='cpu');model.policy.set_training_mode(False)
    target=torch.as_tensor((mpc[:,1:]-mpc[:,:1])/cfg.vehicle.max_steer)
    scales=torch.empty(len(eps),1)
    for epsilon in np.unique(eps):
        active=eps==epsilon;scales[active]=float(np.sqrt(np.square(target.numpy()[active]).mean()))
    if not torch.isfinite(scales).all() or torch.any(scales<=1e-4):raise ValueError('Invalid TRAIN-only response scales')
    actors,critics=actor_critic_parameters(model.policy)
    old_state=deepcopy(model.policy.optimizer.state_dict());counters=(model.num_timesteps,model._n_updates)
    frozen={k:v.clone() for k,v in model.policy.state_dict().items() if 'value_net' in k or 'policy_net_road' in k or k=='log_std'}
    fitted=[p for name,p in model.policy.named_parameters() if ('policy_net' in name or 'action_net' in name) and 'policy_net_road' not in name]
    for name,p in model.policy.named_parameters():p.requires_grad_(any(p is q for q in fitted))
    output.mkdir();shutil.copy2(Path(__file__),output/'executed_fit.py');shutil.copy2(diagnostic,output/'query_diagnostic.json')
    registration={'scope':__doc__,'script_sha256':sha(Path(__file__)),'query_sha256':sha(queries),
        'diagnostic_sha256':sha(diagnostic),'source_model_sha256':metadata['model_sha256'],
        'source_normalization_sha256':metadata['normalization_sha256'],'anchor_rows':len(x),
        'pose_triplets':len(qx),'excluded_unconfirmed_triplets':int((~mask).sum()),
        'hypothetical_poses_are_executed':False,'anchor_targets':'Previous policy means on original measured TRAINinputs;self-distillation,not newly executed teacher labels.',
        'steps':steps,'seed':22,'learning_rate':1e-4,'new_controls':0,'new_PPO_updates':0,
        'loss':'Mean squared one-sided pose-response error divided by TRAINteacher response RMS peraxis,plus original mean retention error divided by[.05,.02].',
        'screen':'All4basic teacher steering RMS<=.01rad;both pose-response RMSerrors reduced>=50%;anchor acceleration normalized RMS<=.05. Unchanged native>=3/4 AND no collision/bounds.',
        'frozen':'Road encoder,critic,log_std,retained PPOAdam,counters,RMS;original direct network architecture.',
        'previous_rejected_ridge_fit_sha256':sha(ROOT/'runs/B1_seed22_pose_feedback_fit_20261101/rejected.json')}
    if continuation:
        registration.update(continuation_parent_model_sha256=metadata['model_sha256'],inherited_contrastive_Adam_steps=512,
            continuation_parent_result_sha256=sha(continue_fit/'rejected.json'),
            continuation_reason='TRAINloss remains decreasing;first endpoint narrowly fails lateral>=50%reduction. One fixed1024additional-step endpoint;no threshold change or intermediate selection.',
            auxiliary_Adam_restart_explicit=True,
            optimizer_restart_reason='First auxiliary optimizer was not serialized;this continuation explicitly starts fresh auxiliary Adam. Original PPOAdam remains exact.',
            retention_anchor_model_sha256=d['model_sha256'])
    atomic_json(output/'registration.json',registration)
    np.savez_compressed(output/'fit_inputs.npz',anchor_observations=obs,anchor_means=anchors.numpy(),pose_observations=qobs,
        MPC_steerings=mpc,epsilons=eps,case_ids=cases,frames=frames[mask],response_scales=scales.numpy())
    optimizer=torch.optim.Adam(fitted,lr=1e-4);rng=np.random.default_rng(22);prob=np.zeros(len(groups))
    for group in np.unique(groups):prob[groups==group]=1/(len(np.unique(groups))*np.count_nonzero(groups==group))
    retention_scale=torch.tensor([.05,.02]);started=time.monotonic();history=[]
    for step in range(1,steps+1):
        if time.monotonic()-started>300:raise TimeoutError('Fixed contrastive fit CPU budget expired')
        indices=rng.choice(len(x),size=128,replace=True,p=prob)
        means=predict(x[indices]);responses=predict(qx.flatten(0,1)).reshape(-1,3,2)
        residual=(responses[:,1:,1]-responses[:,:1,1]-target)/scales
        anchor_loss=((means-anchors[indices])/retention_scale).square().mean();response_loss=residual.square().mean()
        loss=anchor_loss+response_loss
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(fitted,.5,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%64==0:
            okay,reason=resources_ok(ROOT)
            if not okay:raise RuntimeError(str(reason))
            entry={'steps':step,'anchor_loss':float(anchor_loss.detach()),'pose_loss':float(response_loss.detach())};history.append(entry)
            atomic_json(output/'heartbeat.json',entry);print(json.dumps(entry),flush=True)
    for p in model.policy.parameters():p.requires_grad_(True)
    if (not exact(old_state,model.policy.optimizer.state_dict()) or counters!=(model.num_timesteps,model._n_updates)
            or any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in frozen.items())):raise ValueError('Frozen learning state changed')
    np.testing.assert_array_equal(stats[0],norm.obs_rms.mean);np.testing.assert_array_equal(stats[1],norm.obs_rms.var)
    if stats[2]!=norm.obs_rms.count:raise ValueError('RMS count changed')
    with torch.no_grad():
        final=torch.cat([predict(x[i:i+256]) for i in range(0,len(x),256)]).numpy()
        last=predict(qx.flatten(0,1)).reshape(-1,3,2)
    retained=[]
    for case in range(4):
        active=groups==case;retained.append({'case_id':case,'steering_RMS_rad':float(np.sqrt(np.square(final[active,1]-labels[active,1]).mean())*cfg.vehicle.max_steer)})
    gain=[]
    for epsilon in np.unique(eps):
        active=eps==epsilon
        before=(initial[:,1:,1]-initial[:,:1,1]-target).numpy()[active]
        after=(last[:,1:,1]-last[:,:1,1]-target).numpy()[active]
        gain.append({'axis':'lateral_metres' if epsilon>.05 else 'yaw_radians','triplets':int(active.sum()),
            'RMS_difference_error_before':float(np.sqrt(np.square(before).mean())),
            'RMS_difference_error_after':float(np.sqrt(np.square(after).mean()))})
    acceleration_rms=float(np.sqrt(np.square(final[:,0]-anchors.numpy()[:,0]).mean()))
    passed=all(r['steering_RMS_rad']<=.01 for r in retained) and all(r['RMS_difference_error_after']<=.5*r['RMS_difference_error_before'] for r in gain) and acceleration_rms<=.05
    checkpoint=output/'checkpoint';shutil.copytree(continue_fit/'checkpoint' if continuation else origin,checkpoint);model.save(checkpoint/'model.zip')
    torch.save(optimizer.state_dict(),output/'auxiliary_Adam.pt')
    metadata.update(model_sha256=sha(checkpoint/'model.zip'),additional_actor_fit_steps=steps,
        cumulative_actor_fit_steps=metadata.get('cumulative_actor_fit_steps',metadata.get('additional_actor_fit_steps',0))+steps,
        auxiliary_fit_profile='TRAINpose_response_contrast_v1',note='Original trained PPOlineage plus separately charged contrastive actor Adam updates. Fresh actual PPOrollout required;counterfactual pose supervision is explicit.')
    atomic_json(checkpoint/'metadata.json',metadata);verified_checkpoint_metadata(checkpoint)
    result={**registration,'model_sha256':metadata['model_sha256'],'fit_inputs_sha256':sha(output/'fit_inputs.npz'),
        'new_auxiliary_Adam_updates':steps,'auxiliary_Adam_sha256':sha(output/'auxiliary_Adam.pt'),
        'new_controls':0,'teacher_retention':retained,'pose_response':gain,
        'acceleration_anchor_RMS':acceleration_rms,'offline_screen_passed':passed,'history':history,
        'elapsed_seconds':time.monotonic()-started,'research_acceptance':'UNMET'}
    atomic_json(output/('complete.json' if passed else 'rejected.json'),result)
    print(json.dumps({k:result[k] for k in ('model_sha256','teacher_retention','pose_response','acceleration_anchor_RMS','offline_screen_passed')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--continue-fit');p.add_argument('--steps',type=int,default=512);a=p.parse_args();run(a.output,a.continue_fit,a.steps)

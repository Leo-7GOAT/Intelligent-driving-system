"""One fixed reflection-augmented actor fit after the failed real PPO course."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_flat_reflection import reflect_raw
from scripts.fit_flat_feedback_contrast import exact


def run(output,continue_fit=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve the registered reflection fit')
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Fresh frozen runtime required')
    parent=ROOT/'runs/20261101T094700825994Z_native_B1_feedback_PPO'
    sys.path.insert(0,str(parent/'source'))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo,actor_critic_parameters
    from navstack.research import resources_ok,atomic_json
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    require_finished_parent(parent);p=read(parent/'protocol.json');result=read(ROOT/'runs/native_B1_feedback_PPO_20261101.json')
    if result['endpoint']['basic_gate_passed'] or result['new_actual_PPO_controls']!=10240:
        raise ValueError('Require the complete preserved failed basic PPO endpoint')
    origin=parent/'B1'/read(parent/'B1/latest_checkpoint.json')['folder'];metadata=verified_checkpoint_metadata(origin)
    if metadata['model_sha256']!=result['final_model_sha256']:raise ValueError('PPO endpoint differs')
    cfg=EnvConfig.from_dict(read(parent/'B1/config.json')['env'])
    continuation=None
    if continue_fit is not None:
        continue_fit=Path(continue_fit).resolve();continuation=read(continue_fit/'rejected.json')
        if continuation['source_model_sha256']!=metadata['model_sha256'] or continuation['steps']!=512:
            raise ValueError('Only one declared512step exact-Adam continuation is supported')
        origin=continue_fit/'checkpoint';metadata=verified_checkpoint_metadata(origin)
        if metadata['model_sha256']!=continuation['model_sha256']:raise ValueError('Reflection continuation changed')
    teacher=ROOT/'runs/20261027T080111Z_native_rule';tr=read(teacher/'raw_audit.json')
    rawpath=teacher/'infrastructure'/tr['bridge_response_stream'] if 'bridge_response_stream' in tr else next((teacher/'infrastructure').glob('*_responses.jsonl.gz'))
    if sha(rawpath)!=tr['bridge_response_sha256']:raise ValueError('Teacher raw stream changed')
    good=set()
    with gzip.open(rawpath,'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line).get('response',{})
            if 'vehicle_control' not in r:continue
            a=r['applied_vehicle_control']
            if all(abs(r['vehicle_control'][k]-a[k])<=1e-5 for k in ('throttle','brake','steer')) and a['manual_gear_shift'] is True and a['gear']==1 and a['hand_brake'] is False:
                good.add((r['episode_id'],r['frame']))
    with gzip.open(teacher/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    eligibility=[]
    for case in range(12):
        ticks=[t for r in rows if r['case_id']==case for t in r['info']['trace']]
        for i,t in enumerate(ticks):
            if i%5==4:continue
            episode=t['backend_evidence']['episode_id'];frame=t['backend_frame']
            eligibility.append((episode,frame) in good and (i==0 or (episode,frame-1) in good))
    with np.load(parent/'measured_demonstrations.npz',allow_pickle=False) as arrays:
        keep=arrays['train_mask']&(arrays['groups']>=0)&(arrays['groups']<1024)
        obs=arrays['observations'][keep].copy();labels=arrays['actions'][keep].copy();groups=arrays['groups'][keep].copy()
    eligibility=np.asarray(eligibility,dtype=bool)
    if eligibility.shape!=(len(obs),) or eligibility.sum()!=3424:raise ValueError('Acknowledged TRAIN inventory differs')
    obs,labels,groups=obs[eligibility],labels[eligibility],groups[eligibility]
    mirrored=reflect_raw(obs);np.testing.assert_array_equal(reflect_raw(mirrored),obs)
    mirrored_labels=labels.copy();mirrored_labels[:,1]*=-1
    inputs=np.concatenate((obs,mirrored));targets=np.concatenate((labels,mirrored_labels));allgroups=np.r_[groups,groups+12]
    previous=ROOT/'runs/B1_seed22_feedback_contrast_extension_20261101';fit=read(previous/'complete.json')
    if sha(previous/'fit_inputs.npz')!=fit['fit_inputs_sha256']:raise ValueError('Prior response queries changed')
    with np.load(previous/'fit_inputs.npz',allow_pickle=False) as arrays:
        q=arrays['pose_observations'].copy();mpc=arrays['MPC_steerings'].copy();eps=arrays['epsilons'].copy()
    qmirror=reflect_raw(q.reshape(-1,4705)).reshape(q.shape);q=np.concatenate((q,qmirror));mpc=np.concatenate((mpc,-mpc));eps=np.r_[eps,eps]
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu');model.policy.set_training_mode(False)
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False;stats=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),float(norm.obs_rms.count))
    x=torch.as_tensor(norm.normalize_obs(inputs));y=torch.as_tensor(targets)
    qx=torch.as_tensor(norm.normalize_obs(q.reshape(-1,4705))).reshape(-1,3,4705)
    target=torch.as_tensor((mpc[:,1:]-mpc[:,:1])/cfg.vehicle.max_steer);scales=torch.empty(len(eps),1)
    for e in np.unique(eps):scales[eps==e]=float(np.sqrt(np.square(target.numpy()[eps==e]).mean()))
    actor,critic=actor_critic_parameters(model.policy);fitted=[v for k,v in model.policy.named_parameters() if 'policy_net' in k or 'action_net' in k]
    frozen={k:v.clone() for k,v in model.policy.state_dict().items() if 'value_net' in k or k=='log_std'}
    retained=deepcopy(model.policy.optimizer.state_dict());counters=(model.num_timesteps,model._n_updates)
    for v in model.policy.parameters():v.requires_grad_(any(v is a for a in fitted))
    output.mkdir();shutil.copy2(__file__,output/'executed_fit.py');shutil.copy2(ROOT/'scripts/diagnose_flat_reflection.py',output/'executed_reflection.py')
    registration={'scope':__doc__,'script_sha256':sha(Path(__file__)),'reflection_source_sha256':sha(ROOT/'scripts/diagnose_flat_reflection.py'),
        'source_model_sha256':metadata['model_sha256'],'source_normalization_sha256':metadata['normalization_sha256'],
        'source_PPO_report_sha256':sha(ROOT/'runs/native_B1_feedback_PPO_20261101.json'),
        'acknowledged_teacher_TRAINrows':len(obs),'virtual_reflected_rows':len(mirrored),'virtual_pose_triplets':len(q),
        'hypothetical_rows_are_executed':False,'new_controls':0,'new_PPO_updates':0,'steps':512,'learning_rate':1e-4,'seed':22,
        'loss':'Episode-and-reflection balanced teacher MSE divided by[.1,.02],plus original and reflected pose-response MSE divided by TRAINtarget response RMS.',
        'screen':'All4basic original steering RMS<=.01rad and reflected<=.02rad;original basic acceleration normalized RMS<=.1;all pose-response RMS errors<=1.25times the preceding accepted offline pose-fit values. Native gate unchanged.',
        'fixed':'Direct architecture,critic,log_std,PPOAdam,counters,RMS;actor road encoder now trainable with the augmented images.',
        'limits':'Nominal reflection symmetry is a virtual augmentation assumption,not new measured CARLA demonstrations or final conditions.'}
    atomic_json(output/'registration.json',registration)
    np.savez_compressed(output/'fit_inputs.npz',observations=inputs,actions=targets,groups=allgroups,pose_observations=q,MPC_steerings=mpc,epsilons=eps)
    optimizer=torch.optim.Adam(fitted,lr=1e-4);rng=np.random.default_rng(22);weights=np.zeros(len(allgroups))
    for group in np.unique(allgroups):weights[allgroups==group]=1/(len(np.unique(allgroups))*np.count_nonzero(allgroups==group))
    if continuation:
        if sha(continue_fit/'auxiliary_Adam.pt')!=continuation['auxiliary_Adam_sha256']:raise ValueError('Auxiliary Adam changed')
        optimizer.load_state_dict(torch.load(continue_fit/'auxiliary_Adam.pt',map_location='cpu',weights_only=True))
        for _ in range(512):rng.choice(len(x),256,replace=True,p=weights)
        registration.update(inherited_reflection_Adam_steps=512,auxiliary_Adam_and_sampler_resumed_exact=True,
            continuation_parent_result_sha256=sha(continue_fit/'rejected.json'),
            continuation_reason='Both response screens narrowly fail while TRAINloss remains decreasing;one fixed512additional-step endpoint with identical ceilings,no intermediate selection.')
        atomic_json(output/'registration.json',registration)
    def prediction(z):return model.policy.get_distribution(z).distribution.mean
    history=[];started=time.monotonic()
    for step in range(1,513):
        if time.monotonic()-started>300:raise TimeoutError('Reflection fit CPU budget exhausted')
        indices=rng.choice(len(x),256,replace=True,p=weights)
        means=prediction(x[indices]);response=prediction(qx.flatten(0,1)).reshape(-1,3,2)
        point=((means-y[indices])/torch.tensor([.1,.02])).square().mean()
        derivative=((response[:,1:,1]-response[:,:1,1]-target)/scales).square().mean();loss=point+derivative
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(fitted,.5,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%64==0:
            okay,reason=resources_ok(ROOT)
            if not okay:raise RuntimeError(str(reason))
            row={'steps':step,'teacher_loss':float(point.detach()),'response_loss':float(derivative.detach())};history.append(row)
            atomic_json(output/'heartbeat.json',row);print(json.dumps(row),flush=True)
    for v in model.policy.parameters():v.requires_grad_(True)
    if not exact(retained,model.policy.optimizer.state_dict()) or counters!=(model.num_timesteps,model._n_updates) or any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in frozen.items()):raise ValueError('Frozen PPOlearning state changed')
    np.testing.assert_array_equal(stats[0],norm.obs_rms.mean);np.testing.assert_array_equal(stats[1],norm.obs_rms.var)
    if stats[2]!=norm.obs_rms.count:raise ValueError('Normalization changed')
    with torch.no_grad():
        final=torch.cat([prediction(x[i:i+256]) for i in range(0,len(x),256)]).numpy();response=prediction(qx.flatten(0,1)).reshape(-1,3,2)
    metrics=[]
    for group in list(range(4))+list(range(12,16)):
        mask=allgroups==group;error=np.sqrt(np.square(final[mask]-targets[mask]).mean(0))
        metrics.append({'case_id':group%12,'reflected':group>=12,'acceleration_RMS':float(error[0]),'steering_RMS_rad':float(error[1]*cfg.vehicle.max_steer)})
    responses=[]
    for e in np.unique(eps):
        error=(response[:,1:,1]-response[:,:1,1]-target).numpy()[eps==e]
        axis='lateral_metres' if e>.05 else 'yaw_radians';ceiling=1.25*next(r['RMS_difference_error_after'] for r in fit['pose_response'] if r['axis']==axis)
        responses.append({'axis':axis,'normalized_steering_response_RMS':float(np.sqrt(np.square(error).mean())),'ceiling':ceiling})
    passed=all(r['steering_RMS_rad']<=(.02 if r['reflected'] else .01) and (r['reflected'] or r['acceleration_RMS']<=.1) for r in metrics) and all(r['normalized_steering_response_RMS']<=r['ceiling'] for r in responses)
    checkpoint=output/'checkpoint';shutil.copytree(origin,checkpoint);model.save(checkpoint/'model.zip');torch.save(optimizer.state_dict(),output/'auxiliary_Adam.pt')
    metadata.update(model_sha256=sha(checkpoint/'model.zip'),additional_actor_fit_steps=512,pose_response_updates_inherited=1536,
        replay_actor_fit_steps_inherited=512,reflection_feedback_steps=1024 if continuation else 512,auxiliary_fit_profile='reflection_augmented_pose_feedback_v1',
        note='Actual PPO30720lineage plus512separate actor Adam steps on acknowledged TRAINlabels and explicit virtual reflections. PPOoptimizer,counters and RMSretained;fresh rollout required before PPO.')
    atomic_json(checkpoint/'metadata.json',metadata);verified_checkpoint_metadata(checkpoint)
    result={**registration,'model_sha256':metadata['model_sha256'],'fit_inputs_sha256':sha(output/'fit_inputs.npz'),
        'new_auxiliary_Adam_updates':512,'auxiliary_Adam_sha256':sha(output/'auxiliary_Adam.pt'),
        'teacher_metrics':metrics,'pose_response':responses,'offline_screen_passed':passed,'history':history,
        'elapsed_seconds':time.monotonic()-started,'research_acceptance':'UNMET'}
    atomic_json(output/('complete.json' if passed else 'rejected.json'),result)
    print(json.dumps({k:result[k] for k in ('model_sha256','teacher_metrics','pose_response','offline_screen_passed')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--continue-fit');a=p.parse_args();run(a.output,a.continue_fit)

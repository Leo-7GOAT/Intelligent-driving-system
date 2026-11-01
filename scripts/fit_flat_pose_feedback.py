"""One fixed ten-coefficient steering fit with explicit hypothetical pose queries."""
import argparse
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve the single registered pose-feedback fit')
    parent=ROOT/'runs/20261101T011350364179Z_native_B1_basic_lr_extension'
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Fresh frozen runtime required')
    sys.path.insert(0,str(parent/'source'))
    import navstack
    navstack.__path__.append(str(ROOT/'navstack'))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo
    from navstack.pose_feedback_policy import promote,pose_features
    from navstack.subgoals import route_frame_candidates
    from navstack.vehicle import VehicleState
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    cfg=EnvConfig.from_dict(read(parent/'B1/config.json')['env'])
    if cfg.route_candidate_geometry!='route_offsets_v1':
        raise ValueError('Inspect candidate geometry before fitting:'+cfg.route_candidate_geometry)
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    diagnostic=ROOT/'runs/native_B1_feedback_gain_diagnostic_20261101.json';d=read(diagnostic)
    teacher=ROOT/'runs'/d['teacher_batch']
    if sha(teacher/'evaluation_trace.jsonl.gz')!=d['teacher_trace_sha256']:raise ValueError('Diagnostic teacher changed')
    origin=ROOT/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay'
    metadata=verified_checkpoint_metadata(origin)
    if metadata['model_sha256']!=d['model_sha256']:raise ValueError('Diagnostic policy changed')
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu');model.policy.set_training_mode(False)
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False
    paths=list((teacher/'infrastructure').glob('*_responses.jsonl.gz'))
    if len(paths)!=1:raise ValueError('Require the complete single teacher raw stream')
    responses={}
    with gzip.open(paths[0],'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line).get('response',{})
            if 'vehicle_control' in r:responses[r['episode_id'],r['frame']]=r
    def matched(tick):
        e=tick['backend_evidence'];r=responses[e['episode_id'],tick['backend_frame']]
        return (all(abs(r['vehicle_control'][k]-r['applied_vehicle_control'][k])<=1e-5 for k in ('throttle','brake','steer'))
            and r['applied_vehicle_control']['manual_gear_shift'] is True
            and r['applied_vehicle_control']['gear']==1 and r['applied_vehicle_control']['hand_brake'] is False)
    with gzip.open(teacher/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    ticks={case:[t for row in rows if row['case_id']==case for t in row['info']['trace']] for case in range(12)}
    with np.load(parent/'measured_demonstrations.npz',allow_pickle=False) as arrays:
        keep=arrays['train_mask']&(arrays['groups']>=0)&(arrays['groups']<1024)
        obs=arrays['observations'][keep].copy();labels=arrays['actions'][keep].copy();groups=arrays['groups'][keep].copy()
    eligibility=[]
    for case in range(12):
        ts=ticks[case];selected=[i for i in range(len(ts)) if i%5!=4]
        if np.count_nonzero(groups==case)!=len(selected):raise ValueError('TRAIN mask/frame mapping differs')
        eligibility.extend(matched(ts[i]) and (i==0 or matched(ts[i-1])) for i in selected)
    eligibility=np.asarray(eligibility,dtype=bool)
    normalized=norm.normalize_obs(obs)
    with torch.no_grad():
        predictions=np.concatenate([model.policy.get_distribution(torch.as_tensor(normalized[i:i+256])).distribution.mean.numpy() for i in range(0,len(obs),256)])
    features=pose_features(torch.as_tensor(normalized)).numpy().astype(np.float64)
    physical=cfg.vehicle.max_steer
    base_x=features[eligibility];base_y=(labels[:,1]-predictions[:,1])[eligibility]
    base_groups=groups[eligibility]
    base_weights=np.zeros(len(base_x))
    for case in range(12):base_weights[base_groups==case]=.5/(12*np.count_nonzero(base_groups==case))
    routes={case:read(teacher/('scene_%04d.json'%case))['scene']['native_layout']['path'] for case in range(4)}
    def geometry(state,case):
        candidates=route_frame_candidates(state,routes[case],cfg.vehicle,cfg.route_goal_candidate)
        values=[1.];co,si=math.cos(state.yaw),math.sin(state.yaw)
        for index in (5,15,25):
            target,_=candidates[index];dx,dy=target[0]-state.x,target[1]-state.y
            values.extend(((co*dx+si*dy)/15,(-si*dx+co*dy)/15,math.sin(target[2]-state.yaw)))
        return np.asarray(values)
    query_x=[];query_y=[];query_records=[]
    for r in d['records']:
        case,index=r['case_id'],r['control_index'];ts=ticks[case]
        if not matched(ts[index]) or (index and not matched(ts[index-1])):continue
        state=VehicleState(**r['state']);base=geometry(state,case);e=r['epsilon']
        for sign,key in ((-1,'minus_steerings'),(1,'plus_steerings')):
            perturbed=(replace(state,x=state.x-sign*e*math.sin(state.yaw),y=state.y+sign*e*math.cos(state.yaw))
                if r['axis']=='lateral_metres' else replace(state,yaw=state.yaw+sign*e))
            diff=geometry(perturbed,case)-base
            p,m=r[key]
            target=((m-r['base_MPC_steering'])-(p-r['base_policy_steering']))/physical
            query_x.append(diff);query_y.append(target);query_records.append({'case_id':case,'control_index':index,'axis':r['axis'],'sign':sign})
    qx=np.asarray(query_x);qy=np.asarray(query_y)
    x=np.concatenate((base_x,qx));y=np.r_[base_y,qy];weights=np.r_[base_weights,np.full(len(qx),.5/len(qx))]
    output.mkdir();started=time.monotonic()
    registration={'scope':__doc__,'diagnostic_sha256':sha(diagnostic),'source_model_sha256':sha(origin/'model.zip'),
        'source_normalization_sha256':sha(origin/'normalization.pkl'),'new_controls':0,'new_PPO_updates':0,
        'new_Adam_updates':0,'solver':'One weighted ridge solve;lambda=.001 on all10coefficients,no search/retries.',
        'weights':'Half total weight on episode-balanced acknowledged teacher TRAINrows;half on fixed-reference pose differences.',
        'real_teacher_rows':int(eligibility.sum()),'excluded_unconfirmed_or_previous_unconfirmed_rows':int((~eligibility).sum()),
        'hypothetical_pose_differences':len(qx),'hypothetical_labels_are_executed':False,
        'screen':'All4basic teacher steering RMS<=.01rad;both pose gain RMSerrors reduced>=50%. Actual basic gate remains>=3/4 AND no collision/bounds.',
        'fixed_parameters':'All original actor/critic/log_std/PPOAdam/counters/RMS;only10new learned steering coefficients.',
        'script_sha256':sha(Path(__file__)),'policy_source_sha256':sha(ROOT/'navstack/pose_feedback_policy.py')}
    atomic_json(output/'registration.json',registration)
    np.savez_compressed(output/'fit_inputs.npz',features=x,targets=y,weights=weights,teacher_groups=base_groups)
    coefficients=np.linalg.solve(x.T@(weights[:,None]*x)+.001*np.eye(10),x.T@(weights*y))
    before={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    intervention=promote(model)
    with torch.no_grad():model.policy.action_net.policy_net_pose.copy_(torch.as_tensor(coefficients,dtype=torch.float32))
    if any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items()):raise ValueError('Original learned parameters changed')
    retained=[]
    for case in range(4):
        mask=eligibility&(groups==case);delta=features[mask]@coefficients
        retained.append({'case_id':case,'rows':int(mask.sum()),
            'steering_RMS_before_rad':float(np.sqrt(np.square(predictions[mask,1]-labels[mask,1]).mean())*physical),
            'steering_RMS_after_rad':float(np.sqrt(np.square(predictions[mask,1]+delta-labels[mask,1]).mean())*physical)})
    gains=[]
    for axis in ('lateral_metres','yaw_radians'):
        indices=[i for i,r in enumerate(query_records) if r['axis']==axis];error_before=qy[indices]*physical;error_after=(qy[indices]-qx[indices]@coefficients)*physical
        gains.append({'axis':axis,'difference_RMS_error_before':float(np.sqrt(np.square(error_before).mean())),
            'difference_RMS_error_after':float(np.sqrt(np.square(error_after).mean()))})
    passed=all(r['steering_RMS_after_rad']<=.01 for r in retained) and all(r['difference_RMS_error_after']<=.5*r['difference_RMS_error_before'] for r in gains)
    checkpoint=output/'checkpoint';shutil.copytree(origin,checkpoint);model.save(checkpoint/'model.zip')
    reloaded=load_ppo(checkpoint/'model.zip',device='cpu')
    if any(not torch.equal(v,reloaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items()):raise ValueError('New policy reload differs')
    metadata.update(model_sha256=sha(checkpoint/'model.zip'),pose_feedback_profile=intervention,
        new_pose_feedback_ridge_solves=1,new_pose_feedback_parameters=10,
        note='Original PPOcheckpoint plus10separately supervised steering residual coefficients;not new PPO. Original tensors and Adam state retained;new head Adam state initially empty.')
    atomic_json(checkpoint/'metadata.json',metadata);verified_checkpoint_metadata(checkpoint)
    shutil.copy2(Path(__file__),output/'executed_fit.py');shutil.copy2(ROOT/'navstack/pose_feedback_policy.py',output/'executed_policy.py')
    result={**registration,'coefficients':coefficients.tolist(),'model_sha256':metadata['model_sha256'],
        'fit_inputs_sha256':sha(output/'fit_inputs.npz'),'teacher_retention':retained,'pose_difference_fit':gains,
        'offline_screen_passed':passed,'native_driving_gate_passed':False,'elapsed_seconds':time.monotonic()-started,
        'limits':'Pose differences are counterfactual held-reference teacher queries;the baseline was trained under legacy delivery. Actual qualification must use an explicit new sync-delivery transfer. No health or PPOvalue claim.'}
    atomic_json(output/('complete.json' if passed else 'rejected.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();run(a.output)

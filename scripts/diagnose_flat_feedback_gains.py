"""Fixed-reference MPC versus flat-policy pose feedback on recorded TRAIN roads.

Perturbed poses are hypothetical inputs, never integrated or executed in CARLA.
They are not claimed to be measured demonstrations or used for fitting here.
"""
import argparse
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import pickle
import sys
import time


def analyze(root,query_output=None):
    root=Path(root).resolve()
    parent=root/'runs/20261101T011350364179Z_native_B1_basic_lr_extension'
    teacher=root/'runs/20261027T080111Z_native_rule'
    if any(k.startswith('navstack.') for k in sys.modules):
        raise RuntimeError('Use a fresh frozen-runtime interpreter')
    sys.path.insert(0,str(parent/'source'))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.env import rule_action
    from navstack.independent_clip_ppo import load_ppo
    from navstack.measured_demo import measured_flat_env,advance_measured_flat,measured_label
    from navstack.mpc import MPC
    from navstack.vehicle import Command,VehicleState
    from scripts.audit_native_policy_sequence import observation_env
    from scripts.audit_native_method_endpoint import read,sha

    tp=read(teacher/'protocol.json')
    for name in ('env.py','planning.py','trajectory.py','mpc.py'):
        if sha(parent/'source/navstack'/name)!=sha(teacher/'source/navstack'/name):
            raise ValueError('Teacher runtime differs:'+name)
    cfg=EnvConfig.from_dict(read(parent/'B1/config.json')['env'])
    tc=EnvConfig.from_dict(tp['config'])
    origin=root/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay'
    if sha(origin/'model.zip')!='4f20570844ff17da318c601893be8523ddf504193d21775f9553a3d5c8530ca0':
        raise ValueError('Fixed policy changed')
    torch.set_num_threads(1)
    model=load_ppo(origin/'model.zip',device='cpu');model.policy.set_training_mode(False)
    snapshot={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False
    rms=norm.obs_rms;stats=(rms.mean.copy(),rms.var.copy(),float(rms.count))
    def policy_steer(observation,previous):
        x=torch.as_tensor(norm.normalize_obs(observation[None,:]))
        with torch.no_grad():a=model.policy.get_distribution(x).distribution.mean.numpy()[0]
        return float(np.clip(a[1]*cfg.vehicle.max_steer,
            max(-cfg.vehicle.max_steer,previous.steering-cfg.vehicle.steer_rate*cfg.control_dt),
            min(cfg.vehicle.max_steer,previous.steering+cfg.vehicle.steer_rate*cfg.control_dt)))
    def mpc_steer(state,reference,previous):
        result=MPC(tc.vehicle,tc.control_dt,tc.mpc_horizon).step(state,reference,previous)
        if result.status!='solved':raise ValueError('Counterfactual MPC not solved')
        return result.command.steering
    with gzip.open(teacher/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        rows=[json.loads(line) for line in stream]
    records=[];golden_errors=[];query_inputs=[];query_labels=[];query_eps=[];query_cases=[];query_frames=[];started=time.monotonic()
    for case in range(4):
        selected=[r for r in rows if r['case_id']==case]
        initial=selected[0]['info']['initial_scenario'];scene=read(teacher/('scene_%04d.json'%case))['scene']
        if initial['dynamic'] or scene['static']:raise ValueError('Only unobstructed basic TRAIN roads')
        env=observation_env(tc,scene,initial);flat=measured_flat_env(cfg,scene,initial)
        env.world.road_collision_backend=tc.road_collision_backend
        env.reference=None;env.controller=MPC(tc.vehicle,tc.control_dt,tc.mpc_horizon)
        env._rule_plan_cache=None;env.episode={'plan_times':[],'plan_failures':0}
        index=0
        try:
            for row in selected:
                if row['info']['mode']!='reference':break
                if int(rule_action(env))!=row['rule_proposal']:raise ValueError('Held proposal differs')
                mode,_=env._prepare_reference(int(row['policy_action']))
                if mode!='reference':raise ValueError('Held reference differs')
                for tick in row['info']['trace']:
                    if time.monotonic()-started>300:raise TimeoutError('Finite feedback diagnostic expired')
                    s=env.state;previous=env.plant.previous_command
                    golden=env.controller.step(s,env.reference,previous)
                    error=max(abs(getattr(golden.command,k)-tick['requested_command'][k]) for k in ('acceleration','steering'))
                    if golden.status!='solved' or error>1e-5:raise ValueError('Held MPC does not reproduce recorded command')
                    golden_errors.append(error)
                    if index%40==20 and .5<=s.v<=2 and math.dist((s.x,s.y),env.goal[:2])>3:
                        if flat.state!=s or flat.plant.previous_command!=previous:raise ValueError('Teacher and flat measured state differ')
                        flat.plant.state=s;base_observation=flat._observation();base_policy=policy_steer(base_observation,previous)
                        base_mpc=mpc_steer(s,env.reference,previous)
                        for axis,epsilon in (('lateral_metres',.1),('yaw_radians',.02)):
                            controls=[];inputs=[base_observation]
                            for sign in (-1,1):
                                perturbed=(replace(s,x=s.x-sign*epsilon*math.sin(s.yaw),y=s.y+sign*epsilon*math.cos(s.yaw))
                                    if axis=='lateral_metres' else replace(s,yaw=s.yaw+sign*epsilon))
                                flat.plant.state=perturbed
                                observation=flat._observation();inputs.append(observation)
                                controls.append((policy_steer(observation,previous),mpc_steer(perturbed,env.reference,previous)))
                            flat.plant.state=s
                            pg,mg=(np.asarray(controls[1])-controls[0])/(2*epsilon)
                            records.append({'case_id':case,'control_index':index,'frame_before':tick['backend_frame']-1,
                                'axis':axis,'epsilon':epsilon,'state':vars(s),'base_policy_steering':base_policy,
                                'base_MPC_steering':base_mpc,'policy_gain':float(pg),'MPC_gain':float(mg),
                                'minus_steerings':list(controls[0]),'plus_steerings':list(controls[1])})
                            query_inputs.append(inputs);query_labels.append([base_mpc,controls[0][1],controls[1][1]])
                            query_eps.append(epsilon);query_cases.append(case);query_frames.append(tick['backend_frame'])
                        flat.plant.state=s
                    action=measured_label(flat,tick);advance_measured_flat(flat,tick,action)
                    env.plant.state=VehicleState(**tick['state']);env.plant.previous_command=Command(**tick['command'])
                    env.plant.evidence=tick.get('backend_evidence',{});index+=1
        finally:flat.close();env.close()
        print(json.dumps({'case_id':case,'golden_prefix_controls':index,'pose_queries':sum(r['case_id']==case for r in records)}),flush=True)
    if any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in snapshot.items()):raise ValueError('Policy changed')
    np.testing.assert_array_equal(stats[0],rms.mean);np.testing.assert_array_equal(stats[1],rms.var)
    if stats[2]!=rms.count:raise ValueError('Normalizer changed')
    summaries=[]
    for axis in ('lateral_metres','yaw_radians'):
        selected=[r for r in records if r['axis']==axis]
        p=np.array([r['policy_gain'] for r in selected]);m=np.array([r['MPC_gain'] for r in selected]);active=np.abs(m)>.01
        summaries.append({'axis':axis,'queries':len(selected),'active_MPC_queries':int(active.sum()),
            'policy_gain_median':float(np.median(p)),'MPC_gain_median':float(np.median(m)),
            'opposite_sign_active_queries':int(np.count_nonzero(p[active]*m[active]<0)),
            'policy_over_MPC_abs_gain_median_active':float(np.median(np.abs(p[active]/m[active]))),
            'gain_RMS_difference':float(np.sqrt(np.square(p-m).mean()))})
    query_proof=None
    if query_output is not None:
        query_output=Path(query_output)
        if query_output.exists():raise FileExistsError('Preserve prior pose-query data')
        with query_output.open('xb') as stream:
            np.savez_compressed(stream,observations=np.asarray(query_inputs,dtype=np.float32),
                MPC_steerings=np.asarray(query_labels,dtype=np.float32),epsilons=np.asarray(query_eps,dtype=np.float32),
                case_ids=np.asarray(query_cases,dtype=np.int64),frames=np.asarray(query_frames,dtype=np.int64))
        query_proof={'name':query_output.name,'sha256':sha(query_output),'triplets':len(query_inputs),
            'order':'base,minus,plus','hypothetical_labels_are_executed':False}
    return {'scope':__doc__,'script_sha256':sha(Path(__file__)),
        'teacher_batch':teacher.name,'teacher_protocol_sha256':sha(teacher/'protocol.json'),
        'teacher_trace_sha256':sha(teacher/'evaluation_trace.jsonl.gz'),
        'model_sha256':sha(origin/'model.zip'),'normalization_sha256':sha(origin/'normalization.pkl'),
        'selection':'All four original TRAINroads;reference prefix control_index modulo40 equals20,speed[.5,2],goal distance>3m. No outcome-based point selection.',
        'golden_prefix_controls':len(golden_errors),'golden_max_command_error':max(golden_errors),
        'summaries':summaries,'records':records,'query_dataset':query_proof,'new_controls':0,'new_optimizer_updates':0,
        'limits':'Local finite differences around measured teacher states with fixed timed reference and previous command. Pose perturbations are hypothetical;no causal driving gain,closed-loop stability or qualification is established.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output',required=True)
    p.add_argument('--query-output');a=p.parse_args()
    result=analyze(a.root,a.query_output)
    with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result['summaries']),flush=True)

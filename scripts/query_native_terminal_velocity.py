"""Held-reference terminal velocity queries around measured basic TRAIN states."""
import argparse
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def query(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve terminal query registration')
    parent=ROOT/'runs/20261101T100704463570Z_native_reflection_feedback_probe'
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Fresh frozen runtime required')
    sys.path.insert(0,str(parent/'source'))
    import numpy as np
    from navstack.config import EnvConfig
    from navstack.env import rule_action
    from navstack.measured_demo import measured_flat_env,measured_label,advance_measured_flat
    from navstack.mpc import MPC
    from navstack.vehicle import VehicleState,Command
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.audit_native_policy_sequence import observation_env
    from scripts.evaluate_native_rule import require_finished_parent
    require_finished_parent(parent)
    teacher=ROOT/'runs/20261027T080111Z_native_rule';tp=read(teacher/'protocol.json');raw=read(teacher/'raw_audit.json')
    for name in ('env.py','planning.py','trajectory.py','mpc.py'):
        if sha(parent/'source/navstack'/name)!=sha(teacher/'source/navstack'/name):raise ValueError('Teacher runtime differs:'+name)
    cfg=EnvConfig.from_dict(read(parent/'B1/config.json')['env']);tc=EnvConfig.from_dict(tp['config'])
    paths=list((teacher/'infrastructure').glob('*_responses.jsonl.gz'))
    if len(paths)!=1 or sha(paths[0])!=raw['bridge_response_sha256']:raise ValueError('Teacher raw stream differs')
    good=set()
    with gzip.open(paths[0],'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line).get('response',{})
            if 'vehicle_control' in r and all(abs(r['vehicle_control'][k]-r['applied_vehicle_control'][k])<=1e-5 for k in ('throttle','brake','steer')):
                good.add((r['episode_id'],r['frame']))
    with gzip.open(teacher/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    output.mkdir();shutil_source=Path(__file__);(output/'executed_query.py').write_bytes(shutil_source.read_bytes())
    registration={'scope':__doc__,'script_sha256':sha(Path(__file__)),'teacher_trace_sha256':sha(teacher/'evaluation_trace.jsonl.gz'),
        'teacher_protocol_sha256':sha(teacher/'protocol.json'),'selection':'All4original TRAINroads,reference-prefix control_index modulo5 equals0,goal distance<=3m,current+prior returned controls match requests.',
        'queried_speeds_m_s':[.25,.75,1.25,1.75],'new_controls':0,'new_optimizer_updates':0,
        'hypothetical_velocities_are_executed':False,'wall_seconds':300,
        'limits':'Fixed measured pose,time,reference and previous command. Current speed alone is counterfactual;no plant integration or new driving.'}
    atomic_json(output/'registration.json',registration)
    inputs=[];labels=[];groups=[];records=[];errors=[];started=time.monotonic();queried=0
    for case in range(4):
        selected=[r for r in rows if r['case_id']==case];initial=selected[0]['info']['initial_scenario']
        scene=read(teacher/('scene_%04d.json'%case))['scene']
        if initial['dynamic'] or scene['static']:raise ValueError('Terminal queries limited to unobstructed basic TRAINroads')
        env=observation_env(tc,scene,initial);flat=measured_flat_env(cfg,scene,initial)
        env.world.road_collision_backend=tc.road_collision_backend
        env.reference=None;env.controller=MPC(tc.vehicle,tc.control_dt,tc.mpc_horizon)
        env._rule_plan_cache=None;env.episode={'plan_times':[],'plan_failures':0};index=0
        try:
            for row in selected:
                if row['info']['mode']!='reference':break
                if int(rule_action(env))!=row['rule_proposal']:raise ValueError('Held proposal differs')
                mode,_=env._prepare_reference(int(row['policy_action']))
                if mode!='reference':raise ValueError('Held reference differs')
                for tick in row['info']['trace']:
                    if time.monotonic()-started>300:raise TimeoutError('Finite terminal query budget expired')
                    state=env.state;previous=env.plant.previous_command;golden=env.controller.step(state,env.reference,previous)
                    error=max(abs(getattr(golden.command,k)-tick['requested_command'][k]) for k in ('acceleration','steering'))
                    if golden.status!='solved' or error>1e-5:raise ValueError('Golden terminal context differs')
                    errors.append(error);identity=tick['backend_evidence']['episode_id'];frame=tick['backend_frame']
                    if index%5==0 and math.dist((state.x,state.y),env.goal[:2])<=3 and all((identity,frame-j) in good for j in (0,1)):
                        for speed in (.25,.75,1.25,1.75):
                            hypothetical=replace(state,v=speed);flat.plant.state=hypothetical
                            answer=MPC(tc.vehicle,tc.control_dt,tc.mpc_horizon).step(hypothetical,env.reference,previous);queried+=1
                            record={'case_id':case,'control_index':index,'frame_before':frame-1,'speed':speed,'status':answer.status}
                            if answer.status=='solved':
                                a=float(np.clip(answer.command.acceleration,-cfg.vehicle.max_decel,cfg.vehicle.max_accel))
                                label=a/(cfg.vehicle.max_accel if a>=0 else cfg.vehicle.max_decel)
                                inputs.append(flat._observation());labels.append(label);groups.append(case)
                                record['normalized_acceleration_label']=label
                            records.append(record)
                        flat.plant.state=state
                    action=measured_label(flat,tick);advance_measured_flat(flat,tick,action)
                    env.plant.state=VehicleState(**tick['state']);env.plant.previous_command=Command(**tick['command'])
                    env.plant.evidence=tick.get('backend_evidence',{});index+=1
        finally:env.close();flat.close()
        okay,reason=resources_ok(ROOT)
        if not okay:raise RuntimeError(str(reason))
        print(json.dumps({'case_id':case,'queries_so_far':queried,'available_labels':len(labels)}),flush=True)
    if set(groups)!=set(range(4)):raise ValueError('Every original TRAINroad must have terminal queries')
    np.savez_compressed(output/'terminal_velocity_queries.npz',observations=np.asarray(inputs,dtype=np.float32),
        actions=np.asarray(labels,dtype=np.float32),groups=np.asarray(groups,dtype=np.int64))
    result={**registration,'queries':queried,'available_labels':len(labels),'golden_prefix_controls':len(errors),
        'golden_max_command_error':max(errors),'dataset_sha256':sha(output/'terminal_velocity_queries.npz'),
        'records':records,'elapsed_seconds':time.monotonic()-started}
    atomic_json(output/'manifest.json',result);print(json.dumps({'queries':queried,'labels':len(labels)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();query(a.output)

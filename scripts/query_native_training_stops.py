"""Counterfactual B3/MPC labels at training-only B1 stopping states;no execution."""
import argparse
from dataclasses import asdict
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import shutil
ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navstack.config import EnvConfig
from navstack.env import NavigationEnv,rule_action
from navstack.imitation import command_action
from navstack.measured_demo import measured_flat_env
from navstack.mpc import MPC
from navstack.vehicle import Command,brake
from scripts.audit_native_method_endpoint import read,sha
spec=importlib.util.spec_from_file_location('exact_flat_history',Path(__file__).with_name('audit_flat_terminal_policy_drift.py'))
history=importlib.util.module_from_spec(spec);spec.loader.exec_module(history)


def query(batch,oracle_config,output,approach=False):
    batch,oracle_config,output=map(Path,(batch,oracle_config,output));folder=batch/'B1'
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+900
    protocol=read(batch/'protocol.json');raw=read(folder/'endpoint_raw_audit.json')
    for path,key in ((batch/'protocol.json','protocol_sha256'),(folder/'config.json','config_sha256')):
        if sha(path)!=raw[key]:raise ValueError('Raw source binding differs')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:raise ValueError('Use exact frozen native runtime')
    cfg=EnvConfig.from_dict(read(folder/'config.json')['env']);teacher=EnvConfig.from_dict(read(oracle_config)['env'])
    if (teacher.method!='H' or cfg.method!='B1' or teacher.vehicle!=cfg.vehicle or teacher.control_dt!=cfg.control_dt
        or teacher.reward!=cfg.reward or teacher.road_geometry_version!=cfg.road_geometry_version
        or teacher.planner.max_seconds is not None or teacher.goal_tolerance!=cfg.goal_tolerance):
        raise ValueError('Oracle task/geometry/vehicle/reward differs')
    offsets=(180,120,60,20,0) if approach else (60,20,0)
    maximum=12*len(offsets)
    registration={'scope':__doc__,'batch':batch.name,'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'oracle_config_sha256':sha(oracle_config),'script_sha256':sha(Path(__file__)),
        'history_source_sha256':sha(Path(history.__file__)),'selection':'First training state per registered condition with time>=5s,abs(speed)<.05m/s and goal distance in(goal_tolerance,4m);fixed preceding offsets. No evaluation states or outcome-based selection.',
        'offsets_controls':list(offsets),'approach_profile':approach,
        'maximum_queries':maximum,'wall_seconds':900,'new_controls':0,'PPO_updates':0,'supervised_updates':0}
    (output/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__),output/'executed_query_script.py');shutil.copy2(Path(history.__file__),output/'executed_history_script.py')
    records=[];observations=[];actions=[];groups=[]
    try:
        responses=batch/'infrastructure'/raw['bridge_response_stream']
        if sha(responses)!=raw['bridge_response_sha256']:raise ValueError('Bridge bytes differ')
        begins={}
        with gzip.open(responses,'rt',encoding='utf-8') as f:
            for line in f:
                response=json.loads(line).get('response',{})
                if 'scene' in response:begins[response['episode_id']]=response
        episodes={}
        for name,digest in raw['transition_hashes'].items():
            if sha(folder/name)!=digest:raise ValueError('Training transitions differ')
            with gzip.open(folder/name,'rt',encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line);episodes.setdefault(row['episode_id'],[]).append(row)
        if sum(len(r['trace']) for rows in episodes.values() for r in rows)!=raw['training_control_frames_verified']:
            raise ValueError('Training frame inventory differs')
        completed={r['episode_id']:r for r in (json.loads(line) for line in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines())}
        selected={}
        for episode,rows in sorted(episodes.items()):
            if episode not in completed:continue
            begin=begins[rows[0]['trace'][0]['backend_evidence']['episode_id']];condition=begin['scene']['condition']
            if condition in selected:continue
            goal=begin['scene']['goal'];state=begin['state']
            for index,row in enumerate(rows):
                distance=math.hypot(state['x']-goal[0],state['y']-goal[1])
                if state['time']>=5 and abs(state['v'])<.05 and cfg.goal_tolerance<distance<4.:
                    selected[condition]=(episode,sorted({max(0,index-offset) for offset in offsets}));break
                state=row['trace'][-1]['state']
        torch.set_num_threads(1)
        for condition,(episode,indices) in sorted(selected.items()):
            rows=episodes[episode];begin=begins[rows[0]['trace'][0]['backend_evidence']['episode_id']]
            env=measured_flat_env(cfg,begin['scene'],completed[episode]['initial_scenario'],begin=begin)
            try:
                for index,row in enumerate(rows[:max(indices)+1]):
                    if time.monotonic()>deadline:raise TimeoutError('Counterfactual query budget expired')
                    if index in indices:
                        state=asdict(env.state);record={'condition':condition,'training_episode':episode,'control_index':index,
                            'frame_before':env._measured_frame,'episode_id':env._measured_episode,'state':state,'label_available':False}
                        expert=NavigationEnv(teacher);expert.world=env.world;expert.plant=env.plant;expert.goal=env.goal
                        expert.navigation_route=env.navigation_route;expert.last_behavior=np.zeros(4);expert.execution_outcome=np.zeros(6,dtype=np.float32)
                        expert.reference=None;expert._reference_selection=None;expert._rule_plan_cache=None
                        expert.episode={'plan_times':[],'plan_failures':0}
                        before=time.monotonic()
                        if not env.world.free((env.state.x,env.state.y,env.state.yaw),cfg.vehicle):
                            record['rejection']='recorded_pose_not_free'
                        else:
                            action=rule_action(expert);mode,plan=expert._prepare_reference(action)
                            record.update(oracle_action=int(action),oracle_mode=mode,plan=plan)
                            if mode=='reference':
                                control=MPC(cfg.vehicle,cfg.control_dt,cfg.mpc_horizon).step(env.state,expert.reference,env.plant.previous_command)
                                command=control.command;valid=control.status=='solved';record['controller_status']=control.status
                            else:command=brake(env.state,cfg.vehicle);valid=mode in ('wait','dynamic_brake','parking_brake')
                            if valid:
                                rate=cfg.vehicle.steer_rate*cfg.control_dt;previous=env.plant.previous_command.steering
                                projected=Command(float(np.clip(command.acceleration,-cfg.vehicle.max_decel,cfg.vehicle.max_accel)),
                                    float(np.clip(command.steering,max(-cfg.vehicle.max_steer,previous-rate),min(cfg.vehicle.max_steer,previous+rate))))
                                label=command_action(asdict(projected),cfg.vehicle)
                                observations.append(env._observation());actions.append(label);groups.append(1024+episode)
                                record.update(label_available=True,label=label.tolist(),oracle_requested_command=asdict(command),projected_command=asdict(projected))
                            else:record['rejection']='no_executable_oracle_command'
                        record['query_seconds']=time.monotonic()-before
                        if asdict(env.state)!=state:raise ValueError('Offline oracle changed measured state')
                        records.append(record)
                        with (output/'queries.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
                        print(json.dumps({'condition':condition,'index':index,'available':record['label_available'],
                            'mode':record.get('oracle_mode'),'acceleration':record.get('projected_command',{}).get('acceleration'),'seconds':record['query_seconds']}),flush=True)
                    if len(row['trace'])!=1:raise ValueError('Require hold1 direct controls')
                    history.advance_recorded_flat(env,row['trace'][0],row['policy_action'])
            finally:env.close()
        if not records or len(records)>maximum:raise ValueError('Invalid bounded query inventory')
        dataset=output/'corrective_training_queries.npz'
        np.savez_compressed(dataset,observations=np.asarray(observations,dtype=np.float32),actions=np.asarray(actions,dtype=np.float32),groups=np.asarray(groups,dtype=np.int64))
        manifest={**registration,'queries':len(records),'available_labels':len(actions),'selected_conditions':sorted(selected),
            'completed_training_episodes':len(completed),'unfinished_training_episodes_retained':len(episodes)-len(completed),
            'dataset_sha256':sha(dataset),'query_log_sha256':sha(output/'queries.jsonl'),'elapsed_seconds':time.monotonic()-started,
            'labels_are_executed_commands':False}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');return manifest
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps({'error':repr(exc),'completed_queries':len(records),'new_controls':0,'PPO_updates':0},indent=2),encoding='utf-8');raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('batch','oracle-config','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--approach',action='store_true',help='Also query6/9seconds before the first stopping state;all original queries remain included')
    a=p.parse_args();query(a.batch,a.oracle_config,a.output,a.approach)

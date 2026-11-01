"""Persistent2Hz reference/20Hz MPC labels at the exact prior TRAIN query states."""
import argparse
from dataclasses import asdict
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import shutil
import numpy as np


class HeldTeacher:
    """Keep controller and reference context while measured states advance externally."""
    def __init__(self,expert,controller,rule,brake):
        self.expert,self.controller,self.rule,self.brake=expert,controller,rule,brake
        self.period=round(expert.config.decision_dt/expert.config.control_dt)
        if self.period<1 or abs(self.period*expert.config.control_dt-expert.config.decision_dt)>1e-9:
            raise ValueError('Teacher cadence must be a whole number of measured controls')
        self.previous=-1;self.action=None;self.mode=None;self.plan=None

    def query(self,index):
        if index!=self.previous+1:raise ValueError('Advance every measured control exactly once')
        before=vars(self.expert.state).copy();previous=vars(self.expert.plant.previous_command).copy()
        if index%self.period==0:
            self.action=int(self.rule(self.expert))
            self.mode,self.plan=self.expert._prepare_reference(self.action)
        if self.mode=='reference':
            result=self.controller.step(self.expert.state,self.expert.reference,self.expert.plant.previous_command)
            command,status=result.command,result.status
        else:
            command=self.brake(self.expert.state,self.expert.config.vehicle);status=self.mode
        if vars(self.expert.state)!=before or vars(self.expert.plant.previous_command)!=previous:
            raise ValueError('Teacher changed measured plant state or acknowledged command')
        self.previous=index
        return command,status


def query(batch,previous_oracle,teacher_config,context_audit,output):
    batch,previous_oracle,teacher_config,context_audit,output=map(lambda p:Path(p).resolve(),
        (batch,previous_oracle,teacher_config,context_audit,output))
    if output.exists():raise FileExistsError('Preserve prior persistent teacher data')
    if any(name.startswith('navstack.') for name in sys.modules):
        raise RuntimeError('Use a fresh interpreter for the exact frozen runtime')
    runtime=batch/'source';sys.path.insert(0,str(runtime))
    import torch
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv,rule_action
    from navstack.imitation import command_action
    from navstack.measured_demo import measured_flat_env
    from navstack.mpc import MPC
    from navstack.vehicle import Command,brake
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    require_finished_parent(batch);folder=batch/'B1';protocol=read(batch/'protocol.json');raw=read(folder/'endpoint_raw_audit.json')
    old=read(previous_oracle/'manifest.json');audit=read(context_audit)
    if (protocol.get('evaluation_only') or old['batch']!=batch.name
            or old['coverage_profile']!='first_three_episodes_full_trajectory_stride40_v1'
            or old['raw_audit_sha256']!=sha(folder/'endpoint_raw_audit.json')
            or old['oracle_config_sha256']!=sha(teacher_config)
            or audit['teacher_batch']!=protocol['extra_supervision']['teacher_batch']
            or {r['case_id'] for r in audit['results']}!=set(range(4))
            or any(r['held_max_command_error']>1e-5 for r in audit['results'])):
        raise ValueError('Require exact prior TRAIN inputs and complete golden teacher-context validation')
    for path,key in ((batch/'protocol.json','protocol_sha256'),(folder/'config.json','config_sha256')):
        if sha(path)!=raw[key]:raise ValueError('Raw-bound collection inputs changed')
    for name,digest in audit['runtime_source_manifest'].items():
        if sha(runtime/'navstack'/name)!=digest:raise ValueError('Golden reference implementation differs')
    for name,digest in protocol['source_hashes'].items():
        if sha(runtime/name)!=digest:raise ValueError('Frozen source changed:'+name)
    for name,key in [('queries.jsonl','query_log_sha256'),('corrective_training_queries.npz','dataset_sha256')]:
        if sha(previous_oracle/name)!=old[key]:raise ValueError('Previous query inputs changed')
    cfg=EnvConfig.from_dict(read(folder/'config.json')['env']);teacher=EnvConfig.from_dict(read(teacher_config)['env'])
    if (cfg.method!='B1' or teacher.method!='H' or teacher.vehicle!=cfg.vehicle or teacher.control_dt!=cfg.control_dt
            or teacher.decision_dt!=cfg.decision_dt or teacher.reward!=cfg.reward
            or teacher.road_geometry_version!=cfg.road_geometry_version or teacher.planner.max_seconds is not None):
        raise ValueError('Teacher vehicle,interface or cadence differs')
    if len(protocol['native_case_manifest']['cases'])!=4 or any(c['hazard'] is not None for c in protocol['native_case_manifest']['cases']):
        raise ValueError('This persistent oracle supports only the four unobstructed basic TRAIN roads')
    history_path=previous_oracle/'executed_history_script.py'
    if sha(history_path)!=old['history_source_sha256']:raise ValueError('Measured history implementation differs')
    os.environ['NAVSTACK_AUDIT_SOURCE_ROOT']=str(runtime)
    spec=importlib.util.spec_from_file_location('exact_persistent_flat_history',history_path)
    history=importlib.util.module_from_spec(spec);spec.loader.exec_module(history)
    previous=[json.loads(line) for line in (previous_oracle/'queries.jsonl').read_text(encoding='utf-8').splitlines()]
    with np.load(previous_oracle/'corrective_training_queries.npz',allow_pickle=False) as d:
        old_observations=d['observations'].copy()
    if not all(r['label_available'] for r in previous) or len(old_observations)!=len(previous):
        raise ValueError('Exact paired observation check requires the complete121previous query states')
    points={};old_indices={}
    for i,r in enumerate(previous):
        key=(r['condition'],r['training_episode']);points.setdefault(key,[]).append(r['control_index'])
        old_indices[key+(r['control_index'],)]=i
    responses=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(responses)!=raw['bridge_response_sha256']:raise ValueError('Measured bridge stream changed')
    begins={}
    with gzip.open(responses,'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line).get('response',{})
            if 'scene' in r:begins[r['episode_id']]=r
    episodes={}
    for name,digest in raw['transition_hashes'].items():
        if sha(folder/name)!=digest:raise ValueError('TRAIN transition bytes changed')
        with gzip.open(folder/name,'rt',encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line);episodes.setdefault(r['episode_id'],[]).append(r)
    if sum(len(r['trace']) for rows in episodes.values() for r in rows)!=raw['training_control_frames_verified']:
        raise ValueError('Incomplete measured TRAIN inventory')
    completed={r['episode_id']:r for r in map(json.loads,(folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines())}
    output.mkdir(parents=True);shutil.copy2(__file__,output/'executed_query_script.py')
    shutil.copy2(history_path,output/'executed_history_script.py')
    registration={**old,'scope':__doc__,'script_sha256':sha(Path(__file__)),
        'teacher_reference_context':'persistent_2Hz_plan_20Hz_MPC_v1',
        'previous_oracle_manifest_sha256':sha(previous_oracle/'manifest.json'),
        'context_audit_sha256':sha(context_audit),'stored_points_fixed_before_queries':len(previous),
        'new_controls':0,'PPO_updates':0,'supervised_updates':0,'wall_seconds':900}
    for key in ('queries','available_labels','dataset_sha256','query_log_sha256','elapsed_seconds'):
        registration.pop(key,None)
    (output/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
    records=[];observations=[];actions=[];groups=[];started=time.monotonic();teacher_controls=0;teacher_seconds=0.
    torch.set_num_threads(1)
    for (condition,episode),indices in sorted(points.items()):
        if episode not in completed:raise ValueError('Selected TRAIN episode is incomplete')
        rows=episodes[episode];begin=begins[rows[0]['trace'][0]['backend_evidence']['episode_id']]
        if begin['scene']['condition']!=condition or begin['dynamic']:raise ValueError('Different or dynamic condition')
        env=measured_flat_env(cfg,begin['scene'],completed[episode]['initial_scenario'],begin=begin)
        expert=NavigationEnv(teacher);expert.world=env.world;expert.plant=env.plant;expert.goal=env.goal
        expert.navigation_route=env.navigation_route;expert.last_behavior=np.zeros(4);expert.execution_outcome=np.zeros(6,dtype=np.float32)
        expert.reference=None;expert._reference_selection=None;expert._rule_plan_cache=None;expert.episode={'plan_times':[],'plan_failures':0}
        expert.controller=MPC(teacher.vehicle,teacher.control_dt,teacher.mpc_horizon)
        oracle=HeldTeacher(expert,expert.controller,rule_action,brake)
        try:
            for index,row in enumerate(rows[:max(indices)+1]):
                if time.monotonic()-started>900:raise TimeoutError('Finite persistent query budget expired')
                before=time.monotonic();command,status=oracle.query(index)
                elapsed=time.monotonic()-before;teacher_controls+=1;teacher_seconds+=elapsed
                if index in indices:
                    old_row=previous[old_indices[condition,episode,index]]
                    if env._measured_frame!=old_row['frame_before'] or vars(env.state)!=old_row['state']:
                        raise ValueError('Different measured query state or frame')
                    observation=env._observation();np.testing.assert_array_equal(observation,old_observations[old_indices[condition,episode,index]])
                    available=status=='solved' or oracle.mode in ('wait','dynamic_brake','parking_brake')
                    record={**{k:old_row[k] for k in ('condition','training_episode','control_index','frame_before','episode_id','state')},
                        'label_available':available,'oracle_action':oracle.action,'oracle_mode':oracle.mode,
                        'plan':oracle.plan,'controller_status':status,'query_seconds':elapsed,
                        'reference_start_time':getattr(expert.reference,'start_time',None)}
                    if available:
                        limit=cfg.vehicle.steer_rate*cfg.control_dt;prev=env.plant.previous_command.steering
                        projected=Command(float(np.clip(command.acceleration,-cfg.vehicle.max_decel,cfg.vehicle.max_accel)),
                            float(np.clip(command.steering,max(-cfg.vehicle.max_steer,prev-limit),min(cfg.vehicle.max_steer,prev+limit))))
                        label=command_action(asdict(projected),cfg.vehicle)
                        record.update(label=label.tolist(),oracle_requested_command=asdict(command),projected_command=asdict(projected))
                        observations.append(observation);actions.append(label);groups.append(1024+episode)
                    else:record['rejection']='no_executable_persistent_oracle_command'
                    records.append(record)
                    with (output/'queries.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
                if len(row['trace'])!=1:raise ValueError('Require exactly one recorded control per flat action')
                history.advance_recorded_flat(env,row['trace'][0],row['policy_action'])
        finally:env.close();expert.close()
        print(json.dumps({'condition':condition,'episode':episode,'stored_queries':len(records),'teacher_controls':teacher_controls}),flush=True)
    actual=[tuple(r[k] for k in ('condition','training_episode','control_index','frame_before','episode_id')) for r in records]
    expected=[tuple(r[k] for k in ('condition','training_episode','control_index','frame_before','episode_id')) for r in previous]
    if actual!=expected:raise ValueError('Persistent context changed the declared query inventory')
    dataset=output/'corrective_training_queries.npz'
    np.savez_compressed(dataset,observations=np.asarray(observations,dtype=np.float32),actions=np.asarray(actions,dtype=np.float32),groups=np.asarray(groups,dtype=np.int64))
    result={**registration,'queries':len(records),'available_labels':len(actions),
        'dataset_sha256':sha(dataset),'query_log_sha256':sha(output/'queries.jsonl'),
        'all_previous_measured_observations_bit_identical':True,'teacher_control_queries':teacher_controls,
        'teacher_query_seconds':teacher_seconds,'elapsed_seconds':time.monotonic()-started,'labels_are_executed_commands':False}
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('batch','previous-oracle','teacher-config','context-audit','output'):parser.add_argument('--'+key,required=True)
    a=parser.parse_args();query(a.batch,a.previous_oracle,a.teacher_config,a.context_audit,a.output)

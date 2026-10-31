"""Compare initial/final B1 at every audited last-decision observation,offline."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navstack.config import EnvConfig
from navstack.independent_clip_ppo import load_ppo
from navstack.measured_demo import measured_flat_env,measured_label,_actor_snapshot
from navstack.vehicle import Command,VehicleState
from navstack.world import MovingObstacle
from scripts.audit_native_action_availability import verify_source
from scripts.audit_native_method_endpoint import read,sha


def advance_recorded_flat(env,tick,action):
    """Retain the original policy float32 action,not inverse-rounded teacher labels."""
    values=np.clip(np.asarray(action,dtype=np.float32),-1,1);v=env.config.vehicle
    requested={'acceleration':float(values[0]*(v.max_accel if values[0]>=0 else v.max_decel)),
               'steering':float(values[1]*v.max_steer)}
    if requested!=tick['requested_command']:raise ValueError('Policy action does not reproduce the recorded request')
    measured_label(env,tick)  # Independent frame/clock/actor/acknowledgement checks.
    before=env.state
    env.last_behavior=np.array([*values,0.,0.])
    env.plant.state=VehicleState(**tick['state']);env.plant.previous_command=Command(**tick['command'])
    env.plant.evidence=tick['backend_evidence']
    _,states=_actor_snapshot(env.plant.evidence.get('native_dynamic_actors',[] if not env._measured_actor_ids else None),env._measured_actor_ids)
    env.world.dynamic=[MovingObstacle(**state) for state in states];env._measured_frame=env.plant.evidence['frame']
    env.execution_outcome=np.array([0.,0.,0.,0.,0.,(env.state.v-before.v)/v.max_speed],dtype=np.float32)


def analyze(batch,output):
    batch,output=Path(batch),Path(output);folder=batch/'B1'
    if output.exists():raise FileExistsError('Preserve previous terminal-policy diagnosis')
    started=time.monotonic();protocol=read(batch/'protocol.json');raw=read(folder/'endpoint_raw_audit.json')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/'):verify_source(ROOT/name,batch/'source'/name,digest)
    for path,key in ((batch/'protocol.json','protocol_sha256'),(folder/'config.json','config_sha256'),
        (folder/'evaluation_trace.jsonl.gz','evaluation_trace_sha256'),(folder/'evaluation.json','evaluation_sha256')):
        if sha(path)!=raw[key]:raise ValueError('Endpoint input no longer matches raw audit')
    config=EnvConfig.from_dict(read(folder/'config.json')['env']);models=[];norms=[];identities=[]
    for location in (batch/'resume/B1',folder/read(folder/'latest_checkpoint.json')['folder']):
        metadata=read(location/'metadata.json')
        for name,key in (('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')):
            if sha(location/name)!=metadata[key]:raise ValueError('Checkpoint bytes differ')
        models.append(load_ppo(location/'model.zip',device='cpu'))
        with (location/'normalization.pkl').open('rb') as stream:norms.append(pickle.load(stream))
        norms[-1].training=False
        identities.append({k:metadata[k] for k in ('model_sha256','normalization_sha256','physics_steps','optimizer_steps')})
    if models[0].num_timesteps or models[0]._n_updates or models[0].policy.optimizer.state:
        raise ValueError('Require exact zero-PPO initialization')
    for key in ('mean','var','count'):np.testing.assert_array_equal(getattr(norms[0].obs_rms,key),getattr(norms[1].obs_rms,key))
    stream_path=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(stream_path)!=raw['bridge_response_sha256']:raise ValueError('Raw bridge stream differs')
    begins={}
    with gzip.open(stream_path,'rt',encoding='utf-8') as stream:
        for line in stream:
            response=json.loads(line).get('response',{})
            if 'scene' in response:begins[response['episode_id']]=response
    torch.set_num_threads(1);env=None;current=None;results=[];controls=0
    try:
        with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                if time.monotonic()-started>180:raise TimeoutError('Bounded offline reconstruction expired')
                row=json.loads(line);ticks=row['info']['trace']
                if row['case_id']!=current:
                    if env is not None:env.close()
                    current=row['case_id'];begin=begins[ticks[0]['backend_evidence']['episode_id']]
                    env=measured_flat_env(config,begin['scene'],row['info']['initial_scenario'],begin=begin)
                if row['info']['termination_reason'] is not None:
                    observation=env._observation()[None]
                    normalized=[norm.normalize_obs(observation) for norm in norms]
                    np.testing.assert_array_equal(*normalized)
                    predictions=[model.predict(obs,deterministic=True)[0][0] for model,obs in zip(models,normalized)]
                    np.testing.assert_array_equal(predictions[1],row['policy_action'])
                    distributions=[]
                    with torch.no_grad():
                        for model,obs in zip(models,normalized):
                            d=model.policy.get_distribution(torch.as_tensor(obs)).distribution
                            mean=d.mean.numpy()[0];std=d.stddev.numpy()[0]
                            distributions.append({'mean':mean.tolist(),'std':std.tolist(),
                                'positive_acceleration_probability':.5*math.erfc(float(-mean[0]/(std[0]*math.sqrt(2))))})
                    results.append({'case_id':current,'reason':row['info']['termination_reason'],
                        'measured_before_last_state':vars(env.state),'measured_before_last_frame':env._measured_frame,
                        'initial_action':predictions[0].tolist(),'final_action':predictions[1].tolist(),
                        'recorded_action':row['policy_action'],'initial_distribution':distributions[0],
                        'final_distribution':distributions[1]})
                if len(ticks)!=1:raise ValueError('Require one native control per flat action')
                advance_recorded_flat(env,ticks[0],row['policy_action']);controls+=1
    finally:
        if env is not None:env.close()
    expected={r['case_id'] for r in read(folder/'evaluation.json')['episodes']}
    if len(results)!=len(expected) or {r['case_id'] for r in results}!=expected or controls!=raw['evaluation_control_frames_verified']:
        raise ValueError('Incomplete endpoint/control inventory')
    report={'scope':'Every known-condition last decision reconstructed from measured states/history;initial-policy counterfactual is not new closed-loop performance',
        'batch':batch.name,'script_sha256':sha(Path(__file__)),'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'checkpoints_initial_final':identities,'measured_control_history_verified':controls,
        'last_decision_actions_reproduced':len(results),'normalization_statistics_identical':True,
        'results':results,'new_physical_controls':0,'optimizer_updates':0,'wall_seconds':time.monotonic()-started}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps([{'case':r['case_id'],'reason':r['reason'],'initial':r['initial_action'],
        'final':r['final_action'],'p_positive_initial':r['initial_distribution']['positive_acceleration_probability'],
        'p_positive_final':r['final_distribution']['positive_acceleration_probability']} for r in results]));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.batch,a.output)

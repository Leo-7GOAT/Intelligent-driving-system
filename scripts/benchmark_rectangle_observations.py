"""Compare complete flat-policy observations at fixed-stride measured states."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',REPO)).resolve();sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navstack.config import EnvConfig
from navstack.independent_clip_ppo import load_ppo
from navstack.measured_demo import measured_flat_env
from navstack.road_grid_batch import ExactRoadBatch
from scripts.audit_native_method_endpoint import read,sha


def local_module(name,filename):
    spec=importlib.util.spec_from_file_location(name,REPO/'scripts'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


prototype=local_module('rectangle_prototype','experimental_rectangle_kernel.py')
history=local_module('exact_flat_history','audit_flat_terminal_policy_drift.py')


def benchmark(batch,output,stride=100):
    batch,output=Path(batch),Path(output);folder=batch/'B1'
    if output.exists() or not 20<=stride<=500:raise ValueError('Use new output and fixed20..500control stride')
    protocol=read(batch/'protocol.json');raw=read(folder/'endpoint_raw_audit.json')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:raise ValueError('Use exact archived runtime')
    for path,key in ((batch/'protocol.json','protocol_sha256'),(folder/'config.json','config_sha256'),
            (folder/'evaluation.json','evaluation_sha256'),(folder/'evaluation_trace.jsonl.gz','evaluation_trace_sha256')):
        if sha(path)!=raw[key]:raise ValueError('Audited input differs')
    responses=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(responses)!=raw['bridge_response_sha256']:raise ValueError('Native response stream differs')
    begins={}
    with gzip.open(responses,'rt',encoding='utf-8') as stream:
        for line in stream:
            response=json.loads(line).get('response',{})
            if 'scene' in response:begins[response['episode_id']]=response
    config=EnvConfig.from_dict(read(folder/'config.json')['env']);torch.set_num_threads(1)
    evaluation=read(folder/'evaluation.json');model=load_ppo(folder/'model.zip',device='cpu')
    if sha(folder/'model.zip')!=evaluation['model_sha256']:raise ValueError('Evaluated model differs')
    with (folder/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    if sha(folder/'normalization.pkl')!=evaluation['normalization_sha256']:raise ValueError('Normalization differs')
    norm.training=False;stats=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),float(norm.obs_rms.count))
    records=[];controls=0;current=None;envs={};started=time.monotonic();construction=[]
    try:
        with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                if time.monotonic()-started>600:raise TimeoutError('Bounded observation comparison expired')
                row=json.loads(line);ticks=row['info']['trace']
                if len(ticks)!=1:raise ValueError('Require actual one-control flat decisions')
                if row['case_id']!=current:
                    for env in envs.values():env.close()
                    begin=begins[ticks[0]['backend_evidence']['episode_id']];current=row['case_id'];index=0
                    envs={};sizes={};costs={}
                    for label,kernel in (('original_cells',ExactRoadBatch),('rectangle_partition',prototype.RectangleRoadBatch)):
                        env=measured_flat_env(config,begin['scene'],row['info']['initial_scenario'],begin=begin)
                        t=time.perf_counter();env.world._road_batch=kernel(env.world.road);costs[label]=time.perf_counter()-t
                        sizes[label]=len(env.world._road_batch.rectangles);envs[label]=env
                    construction.append({'case_id':current,'seconds':costs,'rectangles':sizes})
                if index%stride==0 or row['info']['termination_reason'] is not None:
                    observations={};times={}
                    for label in (list(envs) if len(records)%2==0 else list(reversed(envs))):
                        t=time.perf_counter();observations[label]=envs[label]._observation();times[label]=time.perf_counter()-t
                    np.testing.assert_array_equal(observations['original_cells'],observations['rectangle_partition'])
                    normalized=[norm.normalize_obs(observations[label][None]) for label in envs]
                    np.testing.assert_array_equal(*normalized)
                    actions=[model.predict(obs,deterministic=True)[0][0] for obs in normalized]
                    np.testing.assert_array_equal(*actions);np.testing.assert_array_equal(actions[0],row['policy_action'])
                    records.append({'case_id':current,'decision':index,'frame_before':envs['original_cells']._measured_frame,
                        'termination_reason':row['info']['termination_reason'],'observation_seconds':times,
                        'policy_action_exact':True,'all_raw_observation_values_exact':True})
                for env in envs.values():history.advance_recorded_flat(env,ticks[0],row['policy_action'])
                index+=1;controls+=1
    finally:
        for env in envs.values():env.close()
    if controls!=raw['evaluation_control_frames_verified'] or {r['case_id'] for r in records}!={r['case_id'] for r in evaluation['episodes']}:
        raise ValueError('Incomplete all-case measured history')
    np.testing.assert_array_equal(stats[0],norm.obs_rms.mean);np.testing.assert_array_equal(stats[1],norm.obs_rms.var)
    if stats[2]!=norm.obs_rms.count:raise ValueError('Normalization statistics changed')
    timings={}
    for label in envs:
        values=[r['observation_seconds'][label] for r in records]
        timings[label]={'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'max':max(values),'count':len(values)}
    report={'scope':__doc__,'batch':batch.name,'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'script_sha256':sha(Path(__file__)),'prototype_sha256':sha(REPO/'scripts/experimental_rectangle_kernel.py'),
        'history_helper_sha256':sha(REPO/'scripts/audit_flat_terminal_policy_drift.py'),
        'selection':'Every fixed-stride control plus first/terminal control of every case,without success/failure exclusion',
        'stride':stride,'native_histories_verified':controls,'sampled_observations':len(records),
        'all_inputs_and_deterministic_actions_exact':True,'normalization_frozen':True,
        'observation_timings':timings,'construction':construction,'samples':records,
        'limits':'Offline interleaved sampled-state compute only;not full closed-loop speed or universal numeric equivalence. Kernel construction separately reported. No runtime enabled.',
        'new_controls':0,'optimizer_updates':0,'production_behavior_changed':False,'wall_seconds':time.monotonic()-started}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','construction')}));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stride',type=int,default=100)
    args=parser.parse_args();benchmark(args.batch,args.output,args.stride)

"""Reconstruct every direct-policy endpoint action from its measured state history."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys


def audit(batch):
    batch=Path(batch).resolve();runtime=batch/'source';folder=batch/'B1'
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Use a fresh frozen-runtime interpreter')
    sys.path.insert(0,str(runtime))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo
    from navstack.measured_demo import measured_flat_env
    from navstack.native_cases import verify_registered_native_begin
    from navstack.vehicle import VehicleState,Command
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    require_finished_parent(batch);p=read(batch/'protocol.json');raw=read(folder/'endpoint_raw_audit.json')
    for name,digest in p['source_hashes'].items():
        if sha(runtime/name)!=digest:raise ValueError('Frozen source changed:'+name)
    cfg=EnvConfig.from_dict(read(folder/'config.json')['env'])
    if cfg.method!='B1' or cfg.flat_hold!=1 or cfg.flat_longitudinal_mode!='acceleration':raise ValueError('Require direct hold1 acceleration policy')
    evaluation=read(folder/'evaluation.json')
    for name,key in (('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')):
        if sha(folder/name)!=evaluation[key]:raise ValueError('Frozen endpoint changed')
    if sha(folder/'evaluation_trace.jsonl.gz')!=raw['evaluation_trace_sha256']:raise ValueError('Endpoint trace changed')
    path=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(path)!=raw['bridge_response_sha256']:raise ValueError('Measured bridge stream changed')
    begins={}
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line).get('response',{})
            if 'scene' in r:begins[r['episode_id']]=r
    with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    torch.set_num_threads(1);model=load_ppo(folder/'model.zip',device='cpu');model.policy.set_training_mode(False)
    before={k:v.clone() for k,v in model.policy.state_dict().items()}
    with (folder/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False;stats=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),float(norm.obs_rms.count))
    records=[]
    for case in sorted({r['case_id'] for r in rows}):
        selected=[r for r in rows if r['case_id']==case];first=selected[0]['info']['trace'][0]
        begin=begins[first['backend_evidence']['episode_id']];verify_registered_native_begin(begin,p,case)
        if begin['dynamic']:raise ValueError('This direct sequence auditor is limited to basic static TRAINroads')
        env=measured_flat_env(cfg,begin['scene'],selected[0]['info']['initial_scenario'],begin=begin)
        count=0
        try:
            for row in selected:
                if len(row['info']['trace'])!=1:raise ValueError('Missing or duplicated direct control')
                tick=row['info']['trace'][0]
                if tick['backend_frame']!=env._measured_frame+1:raise ValueError('Measured frame history discontinuity')
                observation=norm.normalize_obs(env._observation()[None,:])
                predicted=model.predict(observation,deterministic=True)[0][0]
                np.testing.assert_array_equal(predicted,np.asarray(row['policy_action'],dtype=np.float32))
                old_speed=env.state.v;env.last_behavior=np.array([*row['policy_action'],0.,0.])
                env.plant.state=VehicleState(**tick['state']);env.plant.previous_command=Command(**tick['command'])
                env.plant.evidence=tick['backend_evidence'];env._measured_frame=tick['backend_frame']
                env.execution_outcome=np.array([0.,0.,0.,0.,0.,(env.state.v-old_speed)/cfg.vehicle.max_speed],dtype=np.float32)
                count+=1
        finally:env.close()
        record={'case_id':case,'policy_actions_bit_exact':count};records.append(record);print(json.dumps(record),flush=True)
    if sum(r['policy_actions_bit_exact'] for r in records)!=raw['evaluation_control_frames_verified']:
        raise ValueError('Action inventory differs from actual-frame audit')
    if any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen policy changed')
    np.testing.assert_array_equal(stats[0],norm.obs_rms.mean);np.testing.assert_array_equal(stats[1],norm.obs_rms.var)
    if stats[2]!=norm.obs_rms.count:raise ValueError('Frozen normalization count changed')
    return {'scope':__doc__,'batch':batch.name,'script_sha256':sha(Path(__file__)),
        'model_sha256':evaluation['model_sha256'],'normalization_sha256':evaluation['normalization_sha256'],
        'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'protocol_sha256':sha(batch/'protocol.json'),
        'cases':records,'total_bit_exact_actions':sum(r['policy_actions_bit_exact'] for r in records),
        'new_controls':0,'new_optimizer_updates':0,
        'limits':'Reconstructs the saved endpoint from measured states/history;does not replay or validate historical training policies and is not a new physical evaluation.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.batch)
    with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)

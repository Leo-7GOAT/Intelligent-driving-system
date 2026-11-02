"""Audit the completed TRAIN-only route-feedback PPO course, without inventing evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha,read_prefix,_verify_ticks
from scripts.audit_native_control_delivery import analyze as delivery_audit
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def verify_adam(before,after,expected):
    if type(expected) is not int or expected<=0 or before['param_groups']!=after['param_groups']:
        raise ValueError('PPO parameter groups or positive step budget differ')
    ids=[i for g in before['param_groups'] for i in g['params']]
    if len(ids)!=len(set(ids)) or set(after['state'])!=set(ids):raise ValueError('Missing or duplicate PPO states')
    empty=set(ids)-set(before['state'])
    if empty!=set(before['param_groups'][-1]['params']) or len(empty)!=5:
        raise ValueError('Only the five registered route residual states may start empty')
    for i in ids:
        initial=float(before['state'][i]['step']) if i in before['state'] else 0.
        if float(after['state'][i]['step'])-initial!=expected:
            raise ValueError('Every original/new Adam state must advance by recorded actual steps')
    return dict(original_states=len(before['state']),new_route_states=len(empty),step_increment_each=expected)


def audit(batch):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from navstack.native_cases import verify_registered_native_begin
    batch=Path(batch).resolve();require_finished_parent(batch);folder=batch/'B1'
    p=read(batch/'protocol.json');summary=read(folder/'summary.json')
    if (p['route_feedback_PPO']['new_PPO_controls']!=5120 or p.get('evaluation_only')
            or p.get('native_validation') or p['deterministic_training_route_evaluation']
            or len(p['native_case_manifest']['cases'])!=48 or not p['freeze_observation_statistics']
            or summary['physics_steps']!=5120 or summary['stopped_reason'] is not None
            or (folder/'evaluation.json').exists()):raise ValueError('Require exactly the registered TRAIN-only course')
    for name,digest in p['source_hashes'].items():
        if sha(batch/'source'/name)!=digest:raise ValueError('Frozen source changed:'+name)
    checkpoint=folder/read(folder/'latest_checkpoint.json')['folder'];meta=verified_checkpoint_metadata(checkpoint)
    for key in ('physics_steps','optimizer_steps','lineage_control_steps','charged_control_steps_lower_bound'):
        if meta[key]!=summary[key]:raise ValueError('Checkpoint and actual summary disagree:'+key)
    actual,ack,owners,begins={},{},{},{};streams=[]
    for path in sorted((batch/'infrastructure').glob('*_responses.jsonl.gz')):
        rows,error=read_prefix(path)
        if error:raise ValueError('Incomplete bridge stream')
        streams.append(dict(name=path.name,sha256=sha(path)))
        for wrapper in rows:
            r=wrapper.get('response',{});episode=r.get('episode_id')
            if episode is not None and 'request_id' in r:
                key=(episode,r['request_id'])
                if key in ack and ack[key]!=r:raise ValueError('Conflicting acknowledgement')
                ack[key]=r
            if 'reset_ticks' in r:
                if episode in begins:raise ValueError('Duplicate reset')
                verify_registered_native_begin(r,p);begins[episode]=r
            if 'vehicle_control' in r:
                key=(episode,r['frame'])
                if key in actual:raise ValueError('Duplicate actual frame')
                actual[key]=r;owners[key]=path.name
    used,previous,episodes=set(),{},set();count=0;outcomes=Counter();transition_hashes={}
    for path in sorted(folder.glob('transitions*.gz')):
        rows,error=read_prefix(path)
        if error:raise ValueError('Incomplete transitions')
        transition_hashes[path.name]=sha(path)
        for row in rows:
            count+=_verify_ticks(row['trace'],actual,ack,owners,used,previous,episodes,'ff_state_v1')
            if row.get('termination_reason'):outcomes[row['termination_reason']]+=1
    if count!=5120 or used!=set(actual) or summary['world_ticks']!=count+sum(r['reset_ticks'] for r in begins.values()):
        raise ValueError('Actual control/reset inventory differs')
    for (episode,_),r in actual.items():
        if episode not in begins or [(x['actor_id'],x['actor_type']) for x in r.get('native_dynamic_actors',[])]!=[
                (x['actor_id'],x['actor_type']) for x in begins[episode].get('native_dynamic_actors',[])]:
            raise ValueError('Measured actor identities changed')
    optimization,error=read_prefix(folder/'optimization.jsonl')
    if error or not optimization or any(optimization[-1][k]!=summary[k] for k in ('physics_steps','optimizer_steps','optimization_phases')):
        raise ValueError('Actual optimization records disagree')
    torch.set_num_threads(1)
    initial=load_ppo(batch/'resume/B1/model.zip',device='cpu');final=load_ppo(checkpoint/'model.zip',device='cpu')
    if final.num_timesteps-initial.num_timesteps!=5120:raise ValueError('PPO control counter differs')
    adam=verify_adam(initial.policy.optimizer.state_dict(),final.policy.optimizer.state_dict(),summary['optimizer_steps'])
    changed=dict(actor=0,critic=0)
    for name,value in initial.policy.state_dict().items():
        if not torch.equal(value,final.policy.state_dict()[name]):changed['critic' if 'value_net' in name else 'actor']+=1
    if not all(changed.values()):raise ValueError('Actor and critic must really update')
    with (batch/'resume/B1/normalization.pkl').open('rb') as stream:a=pickle.load(stream)
    with (checkpoint/'normalization.pkl').open('rb') as stream:b=pickle.load(stream)
    np.testing.assert_array_equal(a.obs_rms.mean,b.obs_rms.mean);np.testing.assert_array_equal(a.obs_rms.var,b.obs_rms.var)
    if a.obs_rms.count!=b.obs_rms.count:raise ValueError('Frozen RMS count changed')
    delivery=delivery_audit(batch)
    if not delivery['strict_delivery_gate_passed']:raise ValueError('Engine delivery gate failed')
    return dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json'),script_sha256=sha(Path(__file__)),
        new_actual_PPO_controls=count,new_actual_Adam_steps=summary['optimizer_steps'],saved_Adam=adam,
        all_actual_frames_verified=True,strict_engine_delivery=delivery,observation_RMS_exact=True,
        changed_network_tensors=changed,training_outcomes=dict(outcomes),new_auxiliary_steps=0,
        model_sha256=meta['model_sha256'],normalization_sha256=meta['normalization_sha256'],
        transition_hashes=transition_hashes,bridge_streams=streams,world_ticks=summary['world_ticks'],
        endpoint_validation='PENDING;training outcomes are stochastic training,not frozen-policy evaluation',research_acceptance='UNMET')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.batch)
    with Path(args.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ('batch','new_actual_PPO_controls','new_actual_Adam_steps')}),flush=True)

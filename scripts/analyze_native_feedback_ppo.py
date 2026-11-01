"""Audit real PPO learning after the explicitly disclosed pose-response prior."""
import argparse
import json
import math
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_native_control_delivery import analyze as audit_delivery
from scripts.evaluate_native_rule import require_finished_parent


def adam_step_increments(before,after,expected):
    if type(expected) is not int or expected<=0 or not before or before.keys()!=after.keys():
        raise ValueError('Require the same nonempty Adam states and positive actual update count')
    increments=[]
    for key,old in before.items():
        start,end=float(old['step']),float(after[key]['step'])
        if not all(math.isfinite(v) and v.is_integer() for v in (start,end)) or end-start!=expected:
            raise ValueError('Every Adam state must advance by the actual recorded optimizer count')
        increments.append(int(end-start))
    return {'parameter_states':len(increments),'step_increment_each':expected}


def analyze(batch):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    batch=Path(batch).resolve();require_finished_parent(batch);folder=batch/'B1'
    p=read(batch/'protocol.json');spec=p['native_feedback_PPO'];summary=read(folder/'summary.json')
    raw=read(folder/'endpoint_raw_audit.json');evaluation=read(folder/'evaluation.json');inference=read(folder/'measured_start_inference.json')
    if (summary['physics_steps']!=spec['new_PPO_controls'] or raw['protocol_sha256']!=sha(batch/'protocol.json')
            or raw['training_control_frames_verified']!=summary['physics_steps']
            or summary['stopped_reason'] is not None):raise ValueError('Require complete real basic-course training')
    if sha(batch/'resume/B1/model.zip')!=spec['source_model_sha256']:raise ValueError('Initial pose-response prior differs')
    initial=load_ppo(batch/'resume/B1/model.zip',device='cpu');final=load_ppo(folder/'model.zip',device='cpu')
    if final.num_timesteps-initial.num_timesteps!=summary['physics_steps']:
        raise ValueError('Saved PPO control counter differs')
    proof=adam_step_increments(initial.policy.optimizer.state_dict()['state'],final.policy.optimizer.state_dict()['state'],summary['optimizer_steps'])
    changes={'actor':0,'critic':0}
    for key,value in initial.policy.state_dict().items():
        if not torch.equal(value,final.policy.state_dict()[key]):changes['critic' if 'value_net' in key else 'actor']+=1
    if not all(changes.values()):raise ValueError('PPO must actually change actor and critic')
    with (batch/'resume/B1/normalization.pkl').open('rb') as stream:old_norm=pickle.load(stream)
    with (folder/'normalization.pkl').open('rb') as stream:new_norm=pickle.load(stream)
    np.testing.assert_array_equal(old_norm.obs_rms.mean,new_norm.obs_rms.mean)
    np.testing.assert_array_equal(old_norm.obs_rms.var,new_norm.obs_rms.var)
    if old_norm.obs_rms.count!=new_norm.obs_rms.count:raise ValueError('Frozen normalization count changed')
    delivery=audit_delivery(batch)
    if not delivery['strict_delivery_gate_passed']:raise ValueError('Delivery not confirmed throughout training/evaluation')
    episodes=evaluation['episodes']
    if sorted(e['case_id'] for e in episodes)!=[0,1,2,3]:raise ValueError('All4TRAINroads required')
    successes=sum(e['success'] for e in episodes);unsafe=sum(e['termination_reason'] in ('collision','out_of_bounds') for e in episodes)
    if (sha(folder/'model.zip')!=evaluation['model_sha256'] or inference['model_sha256']!=evaluation['model_sha256']
            or inference['batch_protocol_sha256']!=sha(batch/'protocol.json') or len(inference['results'])!=4):
        raise ValueError('Measured-start inference not bound to endpoint')
    return {'scope':__doc__,'script_sha256':sha(Path(__file__)),'batch':batch.name,
        'initial_model_sha256':spec['source_model_sha256'],'final_model_sha256':evaluation['model_sha256'],
        'initial_normalization_sha256':spec['normalization_sha256'],'final_normalization_sha256':evaluation['normalization_sha256'],
        'new_actual_PPO_controls':summary['physics_steps'],'new_actual_Adam_steps':summary['optimizer_steps'],
        'actual_saved_Adam_proof':proof,'changed_network_tensors':changes,
        'new_PPO_epoch_updates':final._n_updates-initial._n_updates,
        'new_supervised_updates_during_PPO':0,'inherited_pose_response_Adam_steps':1536,
        'initial_Gaussian_std':initial.policy.log_std.detach().exp().tolist(),
        'final_Gaussian_std':final.policy.log_std.detach().exp().tolist(),
        'observation_RMS_mean_variance_count_exact':True,'low_level_fixed':'batch_sync_v1',
        'all_engine_controls_acknowledged':True,'delivery_by_phase':delivery['by_phase'],
        'all_actual_raw_controls':raw['total_applied_controls_verified'],'measured_start_actions_verified':4,
        'lineage_controls':summary['lineage_control_steps'],'charged_controls_lower_bound':summary['charged_control_steps_lower_bound'],
        'training_terminal_outcomes':raw['training_termination_reasons'],
        'endpoint':{'successes':successes,'cases':4,'unsafe_cases':unsafe,'basic_gate_passed':successes>=3 and unsafe==0,
            'case_outcomes':[{'case_id':e['case_id'],'termination':e['termination_reason'],'controls':e['physics_steps']} for e in episodes]},
        'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'protocol_sha256':sha(batch/'protocol.json'),
        'evaluation_trace_sha256':sha(folder/'evaluation_trace.jsonl.gz'),
        'limits':'One development seed,warm-started actual PPOafter extra counterfactual supervision. All4conditions are known TRAINroads,not final tests or a healthy five-seed/full-task baseline. Not isolated proof of PPO superiority.',
        'research_acceptance':'UNMET'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.batch)
    with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ('new_actual_PPO_controls','new_actual_Adam_steps','endpoint')}),flush=True)

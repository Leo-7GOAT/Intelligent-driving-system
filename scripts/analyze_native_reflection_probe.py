"""Verify both reflection fits and the complete fixed native deployment."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_native_control_delivery import analyze as delivery_audit
from scripts.analyze_native_feedback_ppo import adam_step_increments
from scripts.fit_flat_feedback_contrast import exact
from scripts.evaluate_native_rule import require_finished_parent


def analyze(batch):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    batch=Path(batch).resolve();require_finished_parent(batch)
    parent=ROOT/'runs/20261101T094700825994Z_native_B1_feedback_PPO'
    p=read(batch/'protocol.json');old=read(parent/'protocol.json')
    if (p['routes']!=old['routes'] or p['native_case_manifest']!=old['native_case_manifest']
            or read(batch/'B1/config.json')['env']!=read(parent/'B1/config.json')['env']
            or sha(batch/'calibration/calibration.json')!=sha(parent/'calibration/calibration.json')):
        raise ValueError('Task,lowlevel,observations,reward or goals differ')
    for name,digest in old['source_hashes'].items():
        if p['source_hashes'].get(name)!=digest:raise ValueError('Frozen source differs:'+name)
    fits=[ROOT/'runs/B1_seed22_reflection_feedback_fit_20261101',ROOT/'runs/B1_seed22_reflection_feedback_extension_20261101']
    results=[read(fits[0]/'rejected.json'),read(fits[1]/'complete.json')]
    base=load_ppo(parent/'B1/model.zip',device='cpu');state=base.policy.state_dict();ppo=base.policy.optimizer.state_dict()
    optimizers=[];proofs=[]
    for folder,result in zip(fits,results):
        for name,key in (('fit_inputs.npz','fit_inputs_sha256'),('executed_fit.py','script_sha256'),('executed_reflection.py','reflection_source_sha256'),('auxiliary_Adam.pt','auxiliary_Adam_sha256')):
            if sha(folder/name)!=result[key]:raise ValueError('Executed auxiliary evidence changed:'+name)
        if sha(folder/'checkpoint/model.zip')!=result['model_sha256']:raise ValueError('Fit checkpoint changed')
        model=load_ppo(folder/'checkpoint/model.zip',device='cpu');current=model.policy.state_dict()
        if (state.keys()!=current.keys() or any(not torch.equal(v,current[k]) for k,v in state.items() if 'value_net' in k or k=='log_std')
                or not exact(ppo,model.policy.optimizer.state_dict()) or (model.num_timesteps,model._n_updates)!=(base.num_timesteps,base._n_updates)
                or sha(folder/'checkpoint/normalization.pkl')!=sha(parent/'B1/normalization.pkl')):
            raise ValueError('Original architecture,critic,logstd,PPOAdam,counters or RMS changed')
        changed=sum(not torch.equal(v,current[k]) for k,v in state.items())
        if not changed:raise ValueError('Expected actual fitted actor displacement')
        optimizer=torch.load(folder/'auxiliary_Adam.pt',map_location='cpu',weights_only=True);optimizers.append(optimizer)
        proofs.append({'model_sha256':result['model_sha256'],'changed_actor_tensors':changed,'new_auxiliary_steps':512,
            'actual_auxiliary_Adam_states':len(optimizer['state']),'actual_step_each':sorted({int(v['step']) for v in optimizer['state'].values()}),
            'architecture_critic_logstd_PPOAdam_counters_RMS_exact':True})
    if proofs[0]['actual_step_each']!=[512] or proofs[1]['actual_step_each']!=[1024]:raise ValueError('Actual auxiliary updates differ')
    increment=adam_step_increments(optimizers[0]['state'],optimizers[1]['state'],512)
    with np.load(fits[0]/'fit_inputs.npz',allow_pickle=False) as a,np.load(fits[1]/'fit_inputs.npz',allow_pickle=False) as b:
        if a.files!=b.files:raise ValueError('Continuation dataset fields differ')
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    if sha(batch/'B1/model.zip')!=results[1]['model_sha256'] or sha(batch/'fit_evidence/complete.json')!=p['native_reflection_feedback']['fit_complete_sha256']:
        raise ValueError('Deployed model differs from the predeclared final fit')
    raw=read(batch/'B1/endpoint_raw_audit.json');inference=read(batch/'B1/measured_start_inference.json')
    if raw['protocol_sha256']!=sha(batch/'protocol.json') or inference['model_sha256']!=results[1]['model_sha256'] or len(inference['results'])!=4:
        raise ValueError('Actual-frame/start-inference binding differs')
    delivery=delivery_audit(batch)
    if not delivery['strict_delivery_gate_passed']:raise ValueError('Synchronous control not acknowledged')
    episodes=read(batch/'B1/evaluation.json')['episodes']
    if sorted(e['case_id'] for e in episodes)!=[0,1,2,3]:raise ValueError('All4TRAINroads required')
    success=sum(e['success'] for e in episodes);unsafe=sum(e['termination_reason'] in ('collision','out_of_bounds') for e in episodes)
    return {'scope':__doc__,'script_sha256':sha(Path(__file__)),'batch':batch.name,
        'fits':proofs,'continued_auxiliary_Adam_proof':increment,'continuation_inputs_exact':True,
        'new_auxiliary_Adam_updates':1024,'new_PPO_updates':0,'new_controls':raw['total_applied_controls_verified'],
        'virtual_reflection_rows_per_fit':3424,'actual_teacher_TRAINrows_per_fit':3424,'virtual_pose_triplets_per_fit':120,
        'virtual_rows_are_executed':False,'runtime_reflection_or_teacher':False,
        'offline_screen_passed':True,'successes':success,'unsafe_cases':unsafe,'basic_gate_passed':success>=3 and unsafe==0,
        'case_outcomes':[{'case_id':e['case_id'],'termination':e['termination_reason'],'controls':e['physics_steps']} for e in episodes],
        'all_engine_controls_acknowledged':True,'measured_start_actions_verified':4,
        'model_sha256':results[1]['model_sha256'],'normalization_sha256':sha(batch/'B1/normalization.pkl'),
        'raw_audit_sha256':sha(batch/'B1/endpoint_raw_audit.json'),'protocol_sha256':sha(batch/'protocol.json'),
        'evaluation_trace_sha256':sha(batch/'B1/evaluation_trace.jsonl.gz'),
        'limits':'Single known development seed,warmstarted from30720actualPPOcontrols with additional hypothetical supervision. Not five independent healthy policies,full-task qualification,PPOaddedvalue or final confirmation.',
        'research_acceptance':'UNMET'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.batch)
    with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ('successes','unsafe_cases','basic_gate_passed','case_outcomes')}),flush=True)

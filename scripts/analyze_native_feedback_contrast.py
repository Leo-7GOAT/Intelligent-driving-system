"""Audit the pose-response auxiliary fit and its complete native deployment probe."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_native_control_delivery import analyze as delivery_audit
from scripts.evaluate_native_rule import require_finished_parent
from scripts.fit_flat_feedback_contrast import exact


def analyze(batch):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    batch=Path(batch).resolve();require_finished_parent(batch)
    old=ROOT/'runs/20261101T084713744547Z_native_sync_delivery_transfer'
    p=read(batch/'protocol.json');op=read(old/'protocol.json')
    if (p['native_case_manifest']!=op['native_case_manifest'] or p['routes']!=op['routes']
            or read(batch/'B1/config.json')['env']!=read(old/'B1/config.json')['env']
            or sha(batch/'calibration/calibration.json')!=sha(old/'calibration/calibration.json')):
        raise ValueError('Task,reward,goals,observations or lowlevel changed beyond the declared fit')
    for name,digest in op['source_hashes'].items():
        if p['source_hashes'].get(name)!=digest:raise ValueError('Declared sync runtime changed:'+name)
    fits=[ROOT/'runs/B1_seed22_feedback_contrast_fit_20261101',ROOT/'runs/B1_seed22_feedback_contrast_extension_20261101']
    results=[read(fits[0]/'rejected.json'),read(fits[1]/'complete.json')]
    origin=ROOT/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay'
    reference=load_ppo(origin/'model.zip',device='cpu')
    state=reference.policy.state_dict();ppo=reference.policy.optimizer.state_dict()
    proofs=[]
    for folder,result in zip(fits,results):
        if sha(folder/'fit_inputs.npz')!=result['fit_inputs_sha256'] or sha(folder/'executed_fit.py')!=result['script_sha256']:
            raise ValueError('Executed fit evidence differs')
        if sha(folder/'checkpoint/model.zip')!=result['model_sha256']:raise ValueError('Fit model differs')
        model=load_ppo(folder/'checkpoint/model.zip',device='cpu');current=model.policy.state_dict()
        if current.keys()!=state.keys():raise ValueError('Direct actor architecture changed')
        frozen=[k for k in state if 'value_net' in k or 'policy_net_road' in k or k=='log_std']
        if (any(not torch.equal(state[k],current[k]) for k in frozen) or not exact(ppo,model.policy.optimizer.state_dict())
                or (model.num_timesteps,model._n_updates)!=(reference.num_timesteps,reference._n_updates)
                or sha(folder/'checkpoint/normalization.pkl')!=sha(origin/'normalization.pkl')):
            raise ValueError('Frozen critic,roadencoder,logstd,PPOAdam,counters or normalization changed')
        changed=[k for k in state if not torch.equal(state[k],current[k])]
        if not changed or any('policy_net' not in k and 'action_net' not in k for k in changed):raise ValueError('Expected actual actor-only parameter displacement')
        proofs.append({'model_sha256':result['model_sha256'],'auxiliary_steps':result['new_auxiliary_Adam_updates'],
            'changed_actor_tensors':len(changed),'frozen_tensors_exact':len(frozen),
            'PPOAdam_counters_normalizer_exact':True,'new_PPO_updates':0})
    auxiliary=torch.load(fits[1]/'auxiliary_Adam.pt',map_location='cpu',weights_only=True)
    steps=[int(v['step']) for v in auxiliary['state'].values()]
    if not steps or set(steps)!={1024}:raise ValueError('Actual saved auxiliary Adam steps differ')
    if sha(fits[1]/'auxiliary_Adam.pt')!=results[1]['auxiliary_Adam_sha256']:raise ValueError('Auxiliary optimizer changed')
    if sha(batch/'B1/model.zip')!=results[1]['model_sha256']:raise ValueError('Evaluated model differs from fixed final fit')
    if sha(batch/'fit_evidence/complete.json')!=p['native_feedback_contrast']['fit_complete_sha256']:
        raise ValueError('Probe protocol does not bind the fit')
    a=read(batch/'B1/endpoint_raw_audit.json');inference=read(batch/'B1/measured_start_inference.json')
    if (a['protocol_sha256']!=sha(batch/'protocol.json') or inference['model_sha256']!=results[1]['model_sha256']
            or inference['batch_protocol_sha256']!=sha(batch/'protocol.json')):
        raise ValueError('Actual-frame and inference evidence does not bind the probe')
    acknowledgement=delivery_audit(batch)
    if not acknowledgement['strict_delivery_gate_passed']:raise ValueError('New native delivery contract failed')
    episodes=read(batch/'B1/evaluation.json')['episodes']
    if sorted(e['case_id'] for e in episodes)!=list(range(4)):raise ValueError('Not all original TRAINroads evaluated')
    success=sum(e['success'] for e in episodes);unsafe=sum(e['termination_reason'] in ('collision','out_of_bounds') for e in episodes)
    return {'scope':__doc__,'script_sha256':sha(Path(__file__)),'batch':batch.name,
        'diagnosis':'Small pointwise imitation error did not preserve local feedback. Held-reference pose queries are hypothetical,not executed teacher demonstrations.',
        'auxiliary_fits':proofs,'new_auxiliary_Adam_updates':sum(r['new_auxiliary_Adam_updates'] for r in results),
        'saved_continuation_Adam_states':len(steps),'saved_continuation_Adam_steps_each':1024,
        'first_auxiliary_optimizer_not_saved':True,'continuation_optimizer_restart_explicit':True,
        'discarded_ten_parameter_ridge_solves':1,'discarded_ridge_native_controls':0,
        'offline_screen':{k:results[1][k] for k in ('teacher_retention','pose_response','acceleration_anchor_RMS','offline_screen_passed')},
        'probe':{'successes':success,'cases':len(episodes),'unsafe_cases':unsafe,
            'basic_gate_passed':success>=3 and unsafe==0,'actual_controls':a['total_applied_controls_verified'],
            'new_PPO_updates':0,'case_outcomes':[{'case_id':e['case_id'],'termination':e['termination_reason'],'controls':e['physics_steps']} for e in episodes],
            'all_engine_controls_acknowledged':True,'start_actions_verified':len(inference['results']),
            'raw_audit_sha256':sha(batch/'B1/endpoint_raw_audit.json'),'protocol_sha256':sha(batch/'protocol.json'),
            'evaluation_trace_sha256':sha(batch/'B1/evaluation_trace.jsonl.gz')},
        'fixed_conditions':'Same all4original TRAINroads,task,goal gates,vehicle,reward,normalizer,direct NNarchitecture,and validated batch_sync_v1 lowlevel. Only learned actor mean weights differ.',
        'limits':'One known development seed with extra counterfactual pose supervision. Not five independent trained policies,new PPOvalue,final confirmation or full-task baseline qualification.',
        'research_acceptance':'UNMET'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.batch)
    with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result['probe']),flush=True)

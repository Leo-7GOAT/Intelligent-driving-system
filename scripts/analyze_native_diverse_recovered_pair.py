"""Reconstruct retained learning and lost work separately for the diverse pair."""
import argparse
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from navstack.research import atomic_json


def analyze(parent,recovery):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from scripts.analyze_native_feedback_ppo import adam_step_increments
    parent,recovery=Path(parent).resolve(),Path(recovery).resolve()
    require_finished_parent(recovery)
    old,new=read(parent/'protocol.json'),read(recovery/'protocol.json')
    if (old['methods']!=['H','H0'] or new['methods']!=['H0']
            or new['parent_batch']!=parent.name
            or old['native_validation']!=new['native_validation']
            or read(parent/'H/config.json')['env']!=read(recovery/'H0/config.json')['env']):
        raise ValueError('Require the same canonical pair,task inventory and fixed physical config')
    interrupted=read(parent/'interruption_audit.json')['methods']['H0']
    results={}
    for method,batch in [('H',parent),('H0',recovery)]:
        folder=batch/method;summary=read(folder/'summary.json');raw=read(folder/'endpoint_raw_audit.json')
        evaluation=read(folder/'evaluation.json')
        if (raw['protocol_sha256']!=sha(batch/'protocol.json')
                or raw['training_control_frames_verified']!=summary['physics_steps']
                or raw['evaluation_sha256']!=sha(folder/'evaluation.json')
                or summary['stopped_reason'] is not None):
            raise ValueError('Endpoint not bound to completed actual-frame audit')
        start=load_ppo(parent/'resume'/method/'model.zip',device='cpu')
        final=load_ppo(folder/'model.zip',device='cpu')
        controls=summary['physics_steps'];updates=summary['optimizer_steps'];consumed=controls
        if method=='H0':
            metadata=read(recovery/'resume/H0/metadata.json')
            controls+=metadata['physics_steps'];updates+=metadata['optimizer_steps']
            consumed+=interrupted['consumed_controls_lower_bound']
        if not 10240<=controls<10560:
            raise ValueError('Retained course exceeded registered one-rollout overshoot')
        proof=adam_step_increments(start.policy.optimizer.state_dict()['state'],
                                  final.policy.optimizer.state_dict()['state'],updates)
        changes={'actor':0,'critic':0}
        for name,value in start.policy.state_dict().items():
            if not torch.equal(value,final.policy.state_dict()[name]):
                changes['critic' if 'value_net' in name else 'actor']+=1
        if not all(changes.values()):raise ValueError('Require actual actor and critic learning')
        norms=[]
        for path in (parent/'resume'/method/'normalization.pkl',folder/'normalization.pkl'):
            with path.open('rb') as stream:norms.append(pickle.load(stream).obs_rms)
        np.testing.assert_array_equal(norms[0].mean,norms[1].mean)
        np.testing.assert_array_equal(norms[0].var,norms[1].var)
        if norms[0].count!=norms[1].count:raise ValueError('Frozen RMS count changed')
        episodes=evaluation['episodes']
        if sorted(x['case_id'] for x in episodes)!=list(range(24)):raise ValueError('Missing known development condition')
        results[method]=dict(batch=batch.name,model_sha256=sha(folder/'model.zip'),
            retained_new_course_controls=controls,consumed_new_course_controls_lower_bound=consumed,
            unknown_unflushed_tail=method=='H0',retained_new_course_Adam_steps=updates,
            actual_saved_Adam_proof=proof,changed_network_tensors=changes,frozen_RMS_exact=True,
            successes=sum(x['success'] for x in episodes),cases=24,
            unsafe_cases=sum(x['termination_reason'] in ('collision','out_of_bounds') for x in episodes),
            evaluation_controls=raw['evaluation_control_frames_verified'],
            raw_audit_sha256=sha(folder/'endpoint_raw_audit.json'))
    return dict(scope='One canonical seed actual diverse-TRAIN adaptation and known24development endpoints',
        analysis_source_sha256=sha(Path(__file__)),results=results,
        known_unsaved_H0_controls=interrupted['consumed_controls_lower_bound']-read(recovery/'resume/H0/metadata.json')['physics_steps'],
        original_H0_consumed_Adam_steps=interrupted['last_optimization']['optimizer_steps'],
        original_H0_retained_Adam_steps=read(recovery/'resume/H0/metadata.json')['optimizer_steps'],
        original_interruption_audit_sha256=sha(parent/'interruption_audit.json'),
        limits='Warm-started canonical22only. All costs and unknown lost tails retained. Not five independent seeds,120new conditions,isolated added-PPO superiority or a whole-budget matched H/B1comparison.',
        new_final_conditions=0,research_acceptance='UNMET')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True);parser.add_argument('--recovery',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous reconstruction')
    result=analyze(args.parent,args.recovery);atomic_json(output,result)
    print({m:(r['retained_new_course_controls'],r['successes']) for m,r in result['results'].items()})

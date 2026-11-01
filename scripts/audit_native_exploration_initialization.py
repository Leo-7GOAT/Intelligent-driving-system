"""Rebuild the explicit std intervention from its retained original checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def audit(batch):
    batch=Path(batch).resolve();runtime=batch/'source'
    if any(name.startswith('navstack.') for name in sys.modules):raise RuntimeError('Use a fresh frozen-runtime interpreter')
    sys.path.insert(0,str(runtime))
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.prepare_native_exploration_ppo import set_exploration,equal
    torch.set_num_threads(1)
    p=read(batch/'protocol.json');spec=p['native_exploration_PPO']
    original=batch/'B1/initial_reference_B1';initial=batch/'resume/B1'
    for name,digest in p['source_hashes'].items():
        if sha(runtime/name)!=digest:raise ValueError('Frozen initialization implementation changed')
    if (sha(original/'model.zip')!=spec['source_model_sha256']
            or sha(initial/'model.zip')!=spec['intervened_model_sha256']
            or sha(original/'normalization.pkl')!=spec['normalization_sha256']):
        raise ValueError('Original or intervened checkpoint differs from registration')
    for name in ('normalization.pkl','resume_state.pkl','config.json'):
        if (original/name).read_bytes()!=(initial/name).read_bytes():raise ValueError('Intervention changed '+name)
    left=load_ppo(original/'model.zip',device='cpu');right=load_ppo(initial/'model.zip',device='cpu')
    proof=set_exploration(left)
    if (proof!=spec['intervention'] or not equal(left.policy.state_dict(),right.policy.state_dict())
            or not equal(left.policy.optimizer.state_dict(),right.policy.optimizer.state_dict())
            or (left.num_timesteps,left._n_updates)!=(right.num_timesteps,right._n_updates)):
        raise ValueError('Initial checkpoint is not exactly the declared std-only intervention')
    return {'scope':__doc__,'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json'),
        'source_model_sha256':spec['source_model_sha256'],'intervened_model_sha256':spec['intervened_model_sha256'],
        'all_intervened_policy_tensors_exact':True,'PPO_optimizer_and_counters_exact':True,
        'normalization_RNG_and_config_bytes_exact':True,'new_controls':0,'new_optimizer_updates':0,
        'in_memory_manual_parameter_change_only':'Reapply the registered std.10intervention to the retained original checkpoint,then compare;no file/model overwrite.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.batch)
    with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)

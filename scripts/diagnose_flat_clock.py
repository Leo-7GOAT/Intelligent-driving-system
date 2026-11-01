"""Clock-only fixed-state actor sensitivity on unobstructed recorded TRAINroads."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys


def shifted_clock(observations,shift):
    import numpy as np
    x=np.asarray(observations,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=4705 or not np.isfinite(x).all():raise ValueError('Require finite4705-dimensional TRAINobservations')
    z=x.copy();z[:,21]-=shift;z[:,22]+=shift
    valid=(z[:,22]>=0)&(z[:,21]>=20)&(x[:,21]>=20)
    return z,valid


def analyze(root):
    root=Path(root);batch=root/'runs/20261101T074146168864Z_native_original_TRAIN_recovery_collection'
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Fresh frozen-runtime interpreter required')
    sys.path.insert(0,str(batch/'source'))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from scripts.audit_native_method_endpoint import read,sha
    p=read(batch/'protocol.json');recorded=read(batch/'B1/collection.json')
    if any(c['hazard'] is not None for c in p['native_case_manifest']['cases']):raise ValueError('Clock invariance probe is restricted to unobstructed static TRAINroads')
    path=batch/'B1/collection.npz';trace=batch/'B1/collection_trace.jsonl.gz'
    if sha(path)!=recorded['dataset_sha256'] or sha(trace)!=recorded['trace_sha256']:raise ValueError('Measured TRAINinputs changed')
    with np.load(path,allow_pickle=False) as d:x=d['observations'].copy();groups=d['groups'].copy()
    with gzip.open(trace,'rt',encoding='utf-8') as stream:rows=[json.loads(s) for s in stream]
    np.testing.assert_allclose(x[:,22],[r['state_before']['time'] for r in rows],atol=2e-6,rtol=1e-7)
    np.testing.assert_allclose(x[:,21]+x[:,22],45.,atol=2e-6)
    origin=root/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay'
    if sha(origin/'model.zip')!='4f20570844ff17da318c601893be8523ddf504193d21775f9553a3d5c8530ca0':raise ValueError('Fixed reference changed')
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu');model.policy.set_training_mode(False)
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False
    def predict(z):
        a=norm.normalize_obs(z)
        with torch.no_grad():return np.clip(np.concatenate([model.policy.get_distribution(torch.as_tensor(a[i:i+256])).distribution.mean.numpy() for i in range(0,len(a),256)]),-1,1)
    reference=predict(x);results=[]
    for shift in (-5.,5.):
        shifted,keep=shifted_clock(x,shift);prediction=predict(shifted);delta=prediction-reference
        case_results=[]
        for group in sorted(set(groups.tolist())):
            mask=keep&(groups==group)
            case_results.append({'case_id':int(group-2048),'rows':int(mask.sum()),
                'mean_action_difference':delta[mask].mean(0).tolist(),
                'RMS_action_difference':np.sqrt(np.square(delta[mask]).mean(0)).tolist(),
                'max_abs_action_difference':np.abs(delta[mask]).max(0).tolist()})
        results.append({'clock_shift_seconds':shift,'valid_nonurgent_rows':int(keep.sum()),
            'RMS_action_difference':np.sqrt(np.square(delta[keep]).mean(0)).tolist(),
            'max_abs_action_difference':np.abs(delta[keep]).max(0).tolist(),'cases':case_results})
    return {'scope':__doc__,'batch':batch.name,'dataset_sha256':sha(path),'trace_sha256':sha(trace),
        'model_sha256':sha(origin/'model.zip'),'normalization_sha256':sha(origin/'normalization.pkl'),
        'script_sha256':sha(Path(__file__)),'clock_indices':{'remaining':21,'elapsed':22},
        'unmodified_observation_indices':'All except21,22;deadline sum45preserved;both clocks have>=20remaining seconds.',
        'results':results,'new_controls':0,'new_optimizer_updates':0,
        'limits':'Offline counterfactual sensitivity only,not driving benefit. Stored original TRAINstates only;no validation/final data or fitting.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.root)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['results']),flush=True)

"""Offline left/right reflection sensitivity; no policy or simulator updates."""
import argparse
import json
from pathlib import Path
import pickle
import sys


def reflect_raw(observations):
    import numpy as np
    x=np.asarray(observations,dtype=np.float32)
    if x.ndim!=2 or x.shape[1] not in (4701,4705) or not np.isfinite(x).all():raise ValueError('Require finite physical road observations')
    y=x.copy();y[:,[1,4,7,9,11,13,15,17,18,160]]*=-1
    y[:,23:144]=x[:,23:144].reshape(-1,11,11)[:,::-1,:].reshape(-1,121)
    neighbors=y[:,144:159].reshape(-1,3,5);neighbors[:,:,[1,3]]*=-1
    boxes=x[:,163:343].reshape(-1,20,9);target=y[:,163:343].reshape(-1,20,9)
    points=boxes[:,:,:8].reshape(-1,20,4,2)[:,:,[3,2,1,0],:].copy();points[:,:,:,1]*=-1
    target[:,:,:8]=points.reshape(-1,20,8)
    corners=x[:,343:351].reshape(-1,4,2)[:,[3,2,1,0],:].copy();corners[:,:,1]*=-1
    y[:,343:351]=corners.reshape(-1,8)
    permutation=[0]
    for block in range(3):
        for offset in range(4,-1,-1):
            permutation.extend([1+block*10+offset*2,2+block*10+offset*2])
    candidates=x[:,351:599].reshape(-1,31,8)[:,permutation,:].copy();candidates[:,:,[1,2]]*=-1
    y[:,351:599]=candidates.reshape(-1,248)
    y[:,605:4701]=x[:,605:4701].reshape(-1,64,64)[:,::-1,:].reshape(-1,4096)
    return y


def analyze(root):
    root=Path(root);parent=root/'runs/20261101T011350364179Z_native_B1_basic_lr_extension'
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Fresh frozen runtime required')
    sys.path.insert(0,str(parent/'source'))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from scripts.audit_native_method_endpoint import read,sha
    with np.load(parent/'measured_demonstrations.npz',allow_pickle=False) as d:
        keep=d['train_mask']&(d['groups']>=0)&(d['groups']<1024)
        x=d['observations'][keep].copy();labels=d['actions'][keep].copy();groups=d['groups'][keep].copy()
    mirrored=reflect_raw(x);np.testing.assert_array_equal(reflect_raw(mirrored),x)
    origin=root/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay'
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu');model.policy.set_training_mode(False)
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False
    def predict(z):
        z=norm.normalize_obs(z)
        with torch.no_grad():return np.concatenate([model.policy.get_distribution(torch.as_tensor(z[i:i+256])).distribution.mean.numpy() for i in range(0,len(z),256)])
    actual=predict(x);reflected=predict(mirrored);reflected[:,1]*=-1
    records=[]
    for group in range(12):
        m=groups==group
        records.append({'case_id':group,'TRAINrows':int(m.sum()),
            'original_action_RMS_error':np.sqrt(np.square(actual[m]-labels[m]).mean(0)).tolist(),
            'reflected_input_unmirrored_output_RMS_error':np.sqrt(np.square(reflected[m]-labels[m]).mean(0)).tolist(),
            'reflection_inconsistency_RMS':np.sqrt(np.square(reflected[m]-actual[m]).mean(0)).tolist()})
    return {'scope':__doc__,'script_sha256':sha(Path(__file__)),'dataset_sha256':sha(parent/'measured_demonstrations.npz'),
        'model_sha256':sha(origin/'model.zip'),'normalization_sha256':sha(origin/'normalization.pkl'),
        'records':records,'new_controls':0,'new_optimizer_updates':0,
        'limits':'Reflection is a hypothetical representation diagnostic under nominal left/right vehicle symmetry,not a measured CARLA replay or driving gain. No averaging,canonicalization or augmentation adopted.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.root)
    with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result['records'][:4]),flush=True)

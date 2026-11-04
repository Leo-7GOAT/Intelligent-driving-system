"""Fixed-policy sensitivity to naturally changing witness bits on recorded TRAINstates."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys


def analyze(root):
    root=Path(root);batch=root/'runs/20261101T074146168864Z_native_original_TRAIN_recovery_collection'
    if any(name.startswith('navstack.') for name in sys.modules):raise RuntimeError('Use a fresh frozen-runtime interpreter')
    sys.path.insert(0,str(batch/'source'))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from scripts.audit_native_method_endpoint import read,sha
    from navstack.config import EnvConfig
    torch.set_num_threads(1)
    recorded=read(batch/'B1/collection.json');audit=read(batch/'B1/collection_raw_audit.json')
    dataset=batch/'B1/collection.npz';trace=batch/'B1/collection_trace.jsonl.gz'
    if (sha(dataset)!=recorded['dataset_sha256'] or sha(trace)!=recorded['trace_sha256']
            or sha(batch/'B1/collection.json')!=audit['collection_result_sha256']
            or not audit['all_measured_observations_bit_exact']):raise ValueError('TRAINobservations differ from their measured audit')
    with np.load(dataset,allow_pickle=False) as d:x=d['observations'].copy();groups=d['groups'].copy()
    with gzip.open(trace,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    if len(rows)!=len(x) or x.shape[1]!=4705:raise ValueError('Incomplete TRAINcontrol inventory')
    origin=root/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay'
    if sha(origin/'model.zip')!='4f20570844ff17da318c601893be8523ddf504193d21775f9553a3d5c8530ca0':raise ValueError('Fixed replay reference changed')
    model=load_ppo(origin/'model.zip',device='cpu');model.policy.set_training_mode(False)
    before={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False;rms=norm.__dict__['obs_rms'];stats=(rms.mean.copy(),rms.var.copy(),float(rms.count))
    normalized=norm.normalize_obs(x);held=normalized.copy()
    old_bits=normalized[:,351:599].reshape(-1,31,8)[:,:,5:8]
    held_bits=held[:,351:599].reshape(-1,31,8)[:,:,5:8]
    same=np.r_[False,groups[1:]==groups[:-1]]
    for i in np.flatnonzero(same):held_bits[i]=old_bits[i-1]
    changes=same&np.any(old_bits!=held_bits,axis=(1,2))
    def predict(z):
        with torch.no_grad():return np.clip(np.concatenate([model.policy.get_distribution(torch.as_tensor(z[i:i+256])).distribution.mean.numpy() for i in range(0,len(z),256)]),-1,1)
    current,past_bits=predict(normalized),predict(held)
    delta=current-past_bits;actual_delta=np.zeros_like(current);actual_delta[1:]=current[1:]-current[:-1]
    cfg=EnvConfig.from_dict(read(batch/'B1/config.json')['env']);vehicle=cfg.vehicle
    def projected(action,index):
        previous=rows[index]['previous_command']['steering'];limit=vehicle.steer_rate*cfg.control_dt
        return np.array([float(action[0])*(vehicle.max_accel if action[0]>=0 else vehicle.max_decel),
            np.clip(float(action[1])*vehicle.max_steer,max(-vehicle.max_steer,previous-limit),min(vehicle.max_steer,previous+limit))])
    projected_delta=np.array([projected(current[i],i)-projected(past_bits[i],i) for i in range(len(x))])
    def summary(mask):
        if not mask.any():return {'rows':0}
        return {'rows':int(mask.sum()),'isolated_witness_action_RMS':np.sqrt(np.square(delta[mask]).mean(0)).tolist(),
            'isolated_witness_action_max_abs':np.abs(delta[mask]).max(0).tolist(),
            'full_adjacent_action_RMS':np.sqrt(np.square(actual_delta[mask]).mean(0)).tolist(),
            'projected_command_RMS_acceleration_steering':np.sqrt(np.square(projected_delta[mask]).mean(0)).tolist(),
            'projected_command_max_abs_acceleration_steering':np.abs(projected_delta[mask]).max(0).tolist()}
    examples=[]
    for i in sorted(np.flatnonzero(changes),key=lambda j:float(np.linalg.norm(delta[j])),reverse=True)[:8]:
        examples.append({'case_id':rows[i]['case_id'],'control_index':rows[i]['control_index'],'frame_before':rows[i]['frame_before'],
            'changed_witness_bits':int(np.count_nonzero(old_bits[i]!=held_bits[i])),
            'current_action':current[i].tolist(),'same_state_previous_bits_action':past_bits[i].tolist(),
            'projected_command_difference':projected_delta[i].tolist()})
    if any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items()):raise ValueError('Sensitivity changed the policy')
    np.testing.assert_array_equal(stats[0],rms.mean);np.testing.assert_array_equal(stats[1],rms.var);assert stats[2]==rms.count
    return {'scope':__doc__,'batch':batch.name,'dataset_sha256':sha(dataset),'trace_sha256':sha(trace),
        'model_sha256':sha(origin/'model.zip'),'normalization_sha256':sha(origin/'normalization.pkl'),
        'auditor_source_sha256':sha(Path(__file__)),'measured_TRAINstates':len(x),
        'naturally_changing_witness_transitions':summary(changes),'unchanged_witness_transitions':summary(same&~changes),
        'examples':examples,'new_controls':0,'new_optimizer_updates':0,
        'limits':'One fixed replay policy at recorded TRAINstates from another learner/teacher mixture. Holding only previous witness bits is an offline counterfactual,not a replay,causal driving benefit or proposal to alter evaluation inputs.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.root)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('measured_TRAINstates','naturally_changing_witness_transitions','unchanged_witness_transitions')}),flush=True)

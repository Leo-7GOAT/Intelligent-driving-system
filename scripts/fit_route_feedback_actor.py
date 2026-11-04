"""Fit only a zero-initialized learned route-error residual on audited TRAIN data."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(output):
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.experiments import signature
    from navstack.independent_clip_ppo import load_ppo
    from navstack.research import atomic_json,resources_ok
    from navstack.route_feedback_policy import promote
    from scripts.native_route_feedback_data import load_bundle
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.fit_broader_measured_actor import exact_state
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every fixed residual fit')
    recipe_path=ROOT/'protocols/native_B1_route_feedback_actor_20261102.json';p=read(recipe_path)
    if p['version']!='route_feedback_actor_v5' or p['steps']!=4096:raise ValueError('Fixed residual recipe differs')
    for name,digest in p['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Registered residual source changed:'+name)
    source=ROOT/p['source_checkpoint'];metadata=verified_checkpoint_metadata(source)
    if metadata['model_sha256']!=p['source_model_sha256']:raise ValueError('Fixed learned parent changed')
    reference=ROOT/'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO/B1/config.json'
    data=load_bundle(ROOT,reference)
    torch.set_num_threads(1);model=load_ppo(source/'model.zip',device='cpu')
    with (source/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    original={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    old_adam=deepcopy(model.policy.optimizer.state_dict());rms=pickle.dumps(norm.obs_rms)
    counters=(model.num_timesteps,model._n_updates)
    report,params=promote(model,norm);norm.training=False
    cfg=EnvConfig.from_dict(read(source/'config.json')['env'])
    cfg=EnvConfig.from_dict({**cfg.to_dict(),'flat_route_observation':True})
    output.mkdir();source_copies={}
    for name in p['source_hashes']:
        target=output/'executed_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        source_copies[name]=sha(target)
    started=time.monotonic();flags=[v.requires_grad for v in model.policy.parameters()];mode=model.policy.training
    selected_ids=set(map(id,params))
    for v in model.policy.parameters():v.requires_grad_(id(v) in selected_ids)
    model.policy.set_training_mode(False)
    normalized=norm.normalize_obs(data['observations']);latents=[];base=[]
    with torch.no_grad():
        for start in range(0,len(normalized),256):
            value=torch.as_tensor(normalized[start:start+256],dtype=torch.float32)
            latent=model.policy.mlp_extractor.forward_actor(model.policy.extract_features(value))
            latents.append(latent);base.append(model.policy.action_net.base_action(latent))
    latent=torch.cat(latents);base=torch.cat(base);target=torch.as_tensor(data['actions'])
    del normalized,latents
    indices=np.flatnonzero(data['train']);groups=data['groups'];masses=data['masses'];weight=np.empty(len(indices))
    for group in np.unique(groups[indices]):
        keep=groups[indices]==group;weight[keep]=masses[int(group)]/(sum(masses.values())*keep.sum())
    rng=np.random.default_rng(22);adam=torch.optim.Adam(params,lr=.001);history=[];scales=torch.tensor([.1,.02])
    atomic_json(output/'registration.json',dict(version=p['version'],recipe_sha256=sha(recipe_path),
        source_model_sha256=metadata['model_sha256'],source_normalization_sha256=metadata['normalization_sha256'],
        source_checkpoint=p['source_checkpoint'],steps=4096,new_learned_parameters=report['new_learned_parameters'],
        train_rows=int(data['train'].sum()),holdout_rows=int(data['holdout'].sum()),data_sources=data['provenance'],
        recovery_audit_sha256=data['recovery_audit_sha256'],source_copies=source_copies,
        route_features_sha256=hashlib.sha256(data['observations'][:,599:603].copy().tobytes()).hexdigest(),
        sampling='Equal physicalgeometry/family/episode,uniform rows;TRAINonly,correlatedholdout never fitted',
        new_PPO_updates=0,new_controls=0))
    try:
        for step in range(1,4097):
            choice=rng.choice(indices,256,replace=True,p=weight)
            prediction=base[choice]+model.policy.action_net.residual(latent[choice],base[choice])
            loss=((prediction-target[choice])/scales).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite residual loss')
            adam.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,.5,error_if_nonfinite=True);adam.step()
            if step==1 or step%256==0:
                okay,reason=resources_ok(ROOT)
                if not okay or time.monotonic()-started>600:raise RuntimeError('Residual fit guard:'+str(reason))
                row=dict(auxiliary_steps=step,scaled_MSE=float(loss.detach()));history.append(row);atomic_json(output/'heartbeat.json',row)
        if (any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in original.items())
                or not exact_state(old_adam['state'],model.policy.optimizer.state_dict()['state'])
                or old_adam['param_groups']!=model.policy.optimizer.state_dict()['param_groups'][:-1]
                or counters!=(model.num_timesteps,model._n_updates) or pickle.dumps(norm.obs_rms)!=rms):
            raise ValueError('Residual fit changed original actor/critic/PPO/RMS state')
        actual=sorted({int(v['step']) for v in adam.state.values()})
        if actual!=[4096] or len(adam.state)!=5:raise ValueError('Residual Adam budget differs')
        with torch.no_grad():predicted=(base+model.policy.action_net.residual(latent,base)).numpy()
        metrics=[]
        for group in np.unique(groups[data['train']]):
            row=dict(group=int(group))
            for name,mask in [('train',data['train']),('correlated_holdout',data['holdout'])]:
                keep=mask&(groups==group);row[name+'_RMS']=np.sqrt(np.square(predicted[keep]-data['actions'][keep]).mean(0)).tolist()
            metrics.append(row)
    finally:
        for value,flag in zip(model.policy.parameters(),flags):value.requires_grad_(flag)
        model.policy.set_training_mode(mode)
    destination=output/'checkpoint';shutil.copytree(source,destination)
    model.save(destination/'model.zip');norm.save(destination/'normalization.pkl')
    atomic_json(destination/'config.json',dict(env=cfg.to_dict(),signature=signature(cfg),seed=22,backend='CARLA_0.9.15',
        scope='Versioned direct learned residual on existing known map route errors;extra TRAINsupervision,no newPPOorcontroller.'))
    metadata.update(model_sha256=sha(destination/'model.zip'),normalization_sha256=sha(destination/'normalization.pkl'),
        route_feedback_version=p['version'],additional_route_residual_steps=4096,new_learned_parameters=report['new_learned_parameters'],
        cumulative_measured_actor_steps=21504,old_actor_critic_PPO_RMS_retained=True)
    atomic_json(destination/'metadata.json',metadata);verified_checkpoint_metadata(destination)
    torch.save(adam.state_dict(),output/'auxiliary_Adam.pt');atomic_json(output/'sampler_state.json',rng.bit_generator.state)
    atomic_json(output/'complete.json',dict(version=p['version'],model_sha256=metadata['model_sha256'],
        normalization_sha256=metadata['normalization_sha256'],actual_Adam_step_each=actual,auxiliary_parameter_states=len(adam.state),
        new_auxiliary_Adam_steps=4096,all_original_actor_critic_tensors_exact=True,PPO_old_groups_and_RMS_retained=True,
        new_controls=0,new_PPO_updates=0,history=history,case_metrics=metrics,wall_seconds=time.monotonic()-started,
        observation_schema=signature(cfg),research_acceptance='UNMET'))
    print(json.dumps(dict(model_sha256=metadata['model_sha256'],new_auxiliary_Adam_steps=4096)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)

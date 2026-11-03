"""Prepare explicitly incompatible reference-context checkpoints;zero new learning."""
import argparse
from pathlib import Path
import pickle
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,read_prefix,sha
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare(output,gain_scale=1.0):
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo
    from navstack.reference_context_policy import migrate_learner,VERSION
    from scripts.audit_native_policy_sequence import observation_env
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every explicit model migration')
    output.mkdir(parents=True);torch.set_num_threads(1);results={}
    sources={'H':ROOT/'runs/20261102T143204721793Z_native_diverse_H_H0_PPO10240',
             'H0':ROOT/'runs/20261103T043242Z_native_recovery'}
    atomic_json(output/'registration.json',dict(version=VERSION,methods=list(sources),
        initial_effect='Two additional zero-initialized parameter tensors;all old learned tensors and Adam states retained',
        gain_scale=gain_scale,new_controls=0,new_PPO_updates=0,new_auxiliary_updates=0,source_sha256=sha(Path(__file__))))
    for method,parent in sources.items():
        folder=parent/method;source=folder/read(folder/'latest_checkpoint.json')['folder']
        metadata=verified_checkpoint_metadata(source);raw=read(folder/'endpoint_raw_audit.json')
        if raw['evaluation_sha256']!=sha(folder/'evaluation.json'):raise ValueError('Audited endpoint changed')
        model=load_ppo(source/'model.zip',device='cpu')
        with (source/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
        norm.training=False
        migrated,newnorm=migrate_learner(model,norm,ablated=method=='H0',gain_scale=gain_scale)
        bridge,error=read_prefix(parent/'infrastructure'/raw['bridge_response_stream'])
        if error:raise ValueError('Bound measured stream is incomplete')
        begins={w['response']['episode_id']:w['response'] for w in bridge if 'scene' in w.get('response',{})}
        rows,error=read_prefix(folder/'evaluation_trace.jsonl.gz')
        if error:raise ValueError(error)
        inputs=[];seen=set()
        cfg=EnvConfig.from_dict(read(source/'config.json')['env'])
        for row in rows:
            if row['case_id'] in seen:continue
            seen.add(row['case_id']);episode=row['info']['trace'][0]['backend_evidence']['episode_id'];begin=begins[episode]
            env=observation_env(cfg,begin['scene'],row['info']['initial_scenario'],begin=begin)
            env.world.road_collision_backend=cfg.road_collision_backend
            obs=env._observation()[None];env.close()
            extended=np.concatenate((obs,np.zeros((1,31),dtype=np.float32)),axis=1)
            old=norm.normalize_obs(obs);new=newnorm.normalize_obs(extended)
            np.testing.assert_array_equal(old,new[:,:-31])
            with torch.no_grad():
                a,b=torch.as_tensor(old),torch.as_tensor(new)
                if (not torch.equal(model.policy.get_distribution(a).distribution.probs,migrated.policy.get_distribution(b).distribution.probs)
                        or not torch.equal(model.policy.predict_values(a),migrated.policy.predict_values(b))):
                    raise ValueError('Zero-initialized policy/value output differs at a measured start')
            inputs.append(dict(case_id=row['case_id'],action=int(migrated.predict(new,deterministic=True)[0][0])))
        if seen!=set(range(24)):raise ValueError('All24measured starts required')
        destination=output/method;destination.mkdir()
        migrated.save(destination/'model.zip');newnorm.save(destination/'normalization.pkl')
        shutil.copy2(source/'resume_state.pkl',destination/'resume_state.pkl')
        config=read(source/'config.json');config['observation_encoding']={'version':VERSION}
        config['observation_shape']=list(migrated.observation_space.shape)
        config['reference_gain_scale']=gain_scale
        atomic_json(destination/'config.json',config)
        metadata.update(model_sha256=sha(destination/'model.zip'),normalization_sha256=sha(destination/'normalization.pkl'),
            resume_state_sha256=sha(destination/'resume_state.pkl'),observation_encoding=VERSION,
            reference_gain_scale=gain_scale,
            transformation_new_controls=0,transformation_new_updates=0)
        atomic_json(destination/'metadata.json',metadata);verified_checkpoint_metadata(destination)
        reloaded=load_ppo(destination/'model.zip',device='cpu')
        if any(not torch.equal(v,reloaded.policy.state_dict()[name]) for name,v in migrated.policy.state_dict().items()):raise ValueError('Saved transformed model changed')
        results[method]=dict(parent=parent.name,parent_checkpoint=source.name,
            old_model_sha256=sha(source/'model.zip'),new_model_sha256=metadata['model_sha256'],
            normalization_sha256=metadata['normalization_sha256'],RNG_bytes_exact=sha(source/'resume_state.pkl')==metadata['resume_state_sha256'],
            original_Adam_states_retained=len(model.policy.optimizer.state),new_empty_parameter_states=2,
            all24measured_start_probabilities_values_exact=True,measured_starts=inputs,
            observation_shape=list(migrated.observation_space.shape),new_controls=0,new_updates=0)
        atomic_json(output/'status.json',dict(completed=results))
    atomic_json(output/'complete.json',dict(version=VERSION,gain_scale=gain_scale,results=results,
        status='INITIALIZERS_PREPARED_NOT_NATIVE_TRAINED',new_controls=0,new_updates=0,research_acceptance='UNMET'))
    print({m:r['observation_shape'] for m,r in results.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--gain-scale',type=float,default=1.0)
    a=p.parse_args();prepare(a.output,a.gain_scale)

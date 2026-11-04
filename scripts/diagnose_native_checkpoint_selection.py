"""Compare fresh initialization and a live job's immutable saved checkpoint.

Recorded initial scenes only; no CARLA connection, plant step, optimizer update
or complete-run frame-audit claim. Production reference queries do not execute.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import actor_vector,signature,write_json
from navstack.independent_clip_ppo import load_ppo
from navstack.road_candidate_policy import RoadCandidateActorCriticPolicy,RoadCandidateVecNormalize
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_native_policy_sequence import observation_env


def analyze(batch,method,checkpoint_name,output):
    batch,output=Path(batch).resolve(),Path(output)
    if output.exists():raise FileExistsError('Preserve checkpoint diagnostic')
    if Path(checkpoint_name).name!=checkpoint_name or not checkpoint_name.startswith('checkpoint_'):
        raise ValueError('Checkpoint must be a direct method child directory')
    protocol=read(batch/'protocol.json')
    if method not in ('H','H0') or method not in protocol['methods'] or protocol.get('recovery'):
        raise ValueError('Only fresh categorical native H/H0 initialization is supported')
    if protocol.get('jobs') or not protocol.get('road_candidate_policy'):
        raise ValueError('Require the fresh structured policy, not inherited experiment arms')
    for name,digest in protocol['source_hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Use the exact frozen observation/policy/planner source')
    folder=batch/method;checkpoint=folder/checkpoint_name
    metadata=read(checkpoint/'metadata.json');saved=read(checkpoint/'config.json')
    config=EnvConfig.from_dict(saved['env'])
    if saved!=read(folder/'config.json'):raise ValueError('Checkpoint and method configurations differ')
    if config.planner.max_seconds is not None:raise ValueError('Require reproducible node-budget planning')
    if saved['signature']!=signature(config):raise ValueError('Checkpoint semantics changed')
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),('resume_state.pkl','resume_state_sha256')]:
        if sha(checkpoint/name)!=metadata[key]:raise ValueError('Checkpoint artifact changed')
    records=[r for r in map(json.loads,(folder/'optimization.jsonl').read_text(encoding='utf-8').splitlines())
             if r['physics_steps']==metadata['physics_steps']]
    if len(records)!=1:raise ValueError('No unique checkpoint optimization record')
    torch.set_num_threads(1)
    learned=load_ppo(checkpoint/'model.zip',device='cpu')
    if learned.num_timesteps!=metadata['actions']:raise ValueError('Saved action counter differs from checkpoint metadata')
    # Construction needs spaces and seeds only. This environment cannot connect
    # to CARLA; assert that SB3 never resets or advances even the local plant.
    local=NavigationEnv(config);raw=DummyVecEnv([lambda:local])
    fresh_norm=RoadCandidateVecNormalize(raw,norm_reward=False,gamma=config.gamma)
    try:
        fresh=PPO(RoadCandidateActorCriticPolicy,fresh_norm,n_steps=protocol['rollout_actions'],
            batch_size=32,n_epochs=protocol['ppo_epochs'],gamma=config.gamma,gae_lambda=.95**.5,
            policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]}},seed=protocol['seed'],device='cpu')
        assert local.reset_count==0 and local.total_physics_steps==0
        actor_change=float(torch.linalg.vector_norm(actor_vector(learned)-actor_vector(fresh)))
        np.testing.assert_allclose(actor_change,records[0]['actor_l2_change'],rtol=1e-6,atol=1e-8)
        with (checkpoint/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
        norm.training=False
        before=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),float(norm.obs_rms.count))
        cases=[];started=time.monotonic();queries=0
        for case_id in range(len(protocol['routes'])):
            path=folder/('scene_%04d.json'%(case_id+1));scene=read(path)
            if scene['scene']['native_layout']['geometry_id']!=protocol['routes'][case_id]['geometry_id']:
                raise ValueError('Initial scene is not the registered route')
            env=observation_env(config,scene['scene'],scene['initial'])
            env.episode={'plan_times':[],'plan_failures':0};env._rule_plan_cache=None
            observation=norm.normalize_obs(env._observation()[None]);selections={};cache={}
            try:
                initial=env.state
                for label,model in (('initial_weights',fresh),('checkpoint_weights',learned)):
                    action=int(model.predict(observation,deterministic=True)[0][0])
                    with torch.no_grad():dist=model.policy.get_distribution(torch.as_tensor(observation)).distribution
                    if action not in cache:
                        if time.monotonic()-started>600:raise RuntimeError('Offline diagnostic wall budget exceeded')
                        env.reference=None
                        mode,detail=env._prepare_reference(action);queries+=1
                        cache[action]=dict(mode=mode,plan=None if detail is None else
                            {k:v for k,v in detail.items() if k!='planning_time'},
                            reference_samples=0 if env.reference is None else len(env.reference.x))
                    selections[label]=dict(action=action,maximum_probability=float(dist.probs.max()),
                        entropy=float(dist.entropy()[0]),selected_reference=cache[action])
                assert env.state==initial and not hasattr(env.plant,'step')
                cases.append(dict(case_id=case_id,scene_sha256=sha(path),
                    same_deterministic_action=selections['initial_weights']['action']==selections['checkpoint_weights']['action'],
                    policies=selections))
                print(json.dumps(cases[-1]),flush=True)
            finally:env.close()
        np.testing.assert_array_equal(before[0],norm.obs_rms.mean)
        np.testing.assert_array_equal(before[1],norm.obs_rms.var)
        assert before[2]==norm.obs_rms.count
        result=dict(scope='Intermediate checkpoint versus seed-reconstructed initial actor on the same frozen-normalized recorded initial scenes; offline selected-reference availability, not executed outcomes, endpoint score or complete frame audit',
            script_sha256=sha(__file__),protocol_sha256=sha(batch/'protocol.json'),method=method,
            checkpoint=checkpoint_name,metadata_sha256=sha(checkpoint/'metadata.json'),
            config_sha256=sha(checkpoint/'config.json'),checkpoint_config_matches_method_config=True,
            checkpoint_controls=metadata['physics_steps'],actor_l2_change=actor_change,
            initialization_actor_change_matches_training_log=True,normalization_unchanged=True,
            same_normalization_for_both_weight_sets=True,reference_queries=queries,
            new_controls=0,optimizer_updates=0,cases=cases)
        write_json(output,result);return result
    finally:fresh_norm.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True)
    parser.add_argument('--method',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    analyze(args.batch,args.method,args.checkpoint,args.output)

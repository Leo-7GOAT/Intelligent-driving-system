"""Reward-selected replay of a PPO policy's own successful training episodes.

No external teacher, planner action replacement, simulator step or PPO update
occurs here. This is an explicitly counted auxiliary actor-fitting phase.
"""
import argparse
from collections import deque,Counter
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2),encoding='utf-8')


def terminal_quality(state,goal):
    return {'distance':float(np.hypot(state['x']-goal[0],state['y']-goal[1])),
            'speed':abs(float(state['v'])),
            'yaw_error':abs(float((state['yaw']-goal[2]+np.pi)%(2*np.pi)-np.pi))}


def has_terminal_margin(quality):
    return quality['distance']<=.5 and quality['speed']<=.15 and quality['yaw_error']<=.2


def actor_replay_loss(distribution,target,speed_label_smoothing=0.):
    if not np.isfinite(speed_label_smoothing) or not 0<=speed_label_smoothing<=.5:
        raise ValueError('Invalid speed-label smoothing')
    loss=-distribution.log_prob(target).mean()
    if speed_label_smoothing:
        factors=distribution.distribution
        if not isinstance(factors,list) or [d.probs.shape[-1] for d in factors]!=[3,7]:
            raise ValueError('Speed smoothing requires the registered3x7direct catalog')
        speed=factors[0]
        loss+=speed_label_smoothing*(speed.log_prob(target[:,0]).mean()-speed.logits.mean())
    return loss


def fit_actor(model,observations,actions,groups,steps,seed=20261026,speed_label_smoothing=0.):
    from navstack.independent_clip_ppo import actor_critic_parameters
    from scripts.continue_native_learning import _equal
    actor,critic=actor_critic_parameters(model.policy)
    frozen=[p.detach().clone() for p in critic]
    critic_adam=[deepcopy(model.policy.optimizer.state.get(p,{})) for p in critic]
    updates=model._n_updates;generator=np.random.default_rng(seed)
    labels=np.unique(groups);indices={k:np.flatnonzero(groups==k) for k in labels}
    losses=[];model.policy.set_training_mode(True)
    for step in range(steps):
        selected=[int(generator.choice(indices[k])) for k in generator.choice(labels,size=64)]
        obs=torch.as_tensor(observations[selected],device=model.device)
        target=torch.as_tensor(actions[selected],device=model.device)
        loss=actor_replay_loss(model.policy.get_distribution(obs),target,speed_label_smoothing)
        if not torch.isfinite(loss):raise ValueError('Nonfinite self-success actor objective')
        model.policy.optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(actor,.5,error_if_nonfinite=True)
        model.policy.optimizer.step();losses.append(float(loss.detach()))
    if model._n_updates!=updates:raise ValueError('Auxiliary actor fit changed PPO counters')
    for p,value,state in zip(critic,frozen,critic_adam):
        if not torch.equal(p,value) or not _equal(model.policy.optimizer.state.get(p,{}),state):
            raise ValueError('Actor replay changed critic weights or optimizer state')
    return {'actor_optimizer_steps':steps,'ppo_updates_added':0,'critic_unchanged':True,
            'critic_adam_unchanged':True,'first_loss':losses[0],'last_loss':losses[-1],
            'speed_label_smoothing':speed_label_smoothing}


def run(model_batch,method,sources,steps=1000,terminal_margin=False):
    if not 1<=steps<=2000:raise ValueError('Use a bounded auxiliary fit')
    model_batch=Path(model_batch).resolve();runtime=model_batch/'source'
    if any(k.startswith('navstack.') for k in sys.modules):
        raise RuntimeError('Run in a fresh interpreter to use the exact frozen observation code')
    sys.path.insert(0,str(runtime))
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv,flat_action_values
    from navstack.world import World
    from navstack.road_grid import RoadGrid
    from navstack.vehicle import VehicleState,Command,wrap
    from navstack.independent_clip_ppo import load_ppo
    from navstack.research import resources_ok
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.audit_native_method_endpoint import audit
    from scripts.audit_native_policy_sequence import advance_observation
    require_finished_parent(model_batch)
    healthy,reason=resources_ok(ROOT)
    if not healthy:raise RuntimeError(str(reason))
    torch.set_num_threads(1)
    folder=model_batch/method;config_record=read(folder/'config.json');config=EnvConfig.from_dict(config_record['env'])
    if config.method!='B1' or getattr(config,'flat_action_catalog','continuous')!='speed_steer_3x7_v1':
        raise ValueError('This first replay experiment requires the direct categorical B1')
    records=[];raw=[];targets=[];groups=[];selection=[];excluded=0;unselected=[]
    for source_method in sources:
        location,name=source_method.rsplit(':',1);batch=Path(location).resolve();p=batch/name
        require_finished_parent(batch)
        if read(p/'config.json')['env']!=config_record['env']:raise ValueError('Replay task/action semantics differ')
        protocol=read(batch/'protocol.json')
        for key,digest in protocol['source_hashes'].items():
            if sha(batch/'source'/key)!=digest:raise ValueError('Frozen replay source changed')
        for key,digest in read(model_batch/'protocol.json')['source_hashes'].items():
            if key.startswith('navstack/') and protocol['source_hashes'].get(key)!=digest:
                raise ValueError('Do not pool experience from different execution/observation versions')
        rebuilt=audit(batch,name)
        if rebuilt!=read(p/'endpoint_raw_audit.json'):raise ValueError('Replay input is not raw-audited')
        episodes=[json.loads(l) for l in (p/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        success={e['episode_id']:e for e in episodes if e['success']}
        transitions={i:[] for i in success}
        for path in sorted(p.glob('transitions_*.jsonl.gz')):
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                for row in map(json.loads,stream):
                    if row['episode_id'] in success:transitions[row['episode_id']].append(row)
        selected_before=len(selection)
        for i,rows in transitions.items():
            if not rows or rows[-1]['termination_reason']!='success':raise ValueError('Incomplete successful episode')
            scene=read(p/f'scene_{i+1:04d}.json');initial=scene['initial'];layout=scene['scene']
            quality=terminal_quality(rows[-1]['trace'][-1]['state'],initial['goal'])
            if terminal_margin and not has_terminal_margin(quality):
                unselected.append({'source_batch':batch.name,'episode_id':i,'condition':layout['condition'],**quality})
                continue
            env=NavigationEnv(config);env.world=World(layout['width'],layout['height'],layout['static'],road=RoadGrid(layout['road_grid']))
            env.world.road_collision_backend=config.road_collision_backend
            env.goal=initial['goal'];env.navigation_route=layout['native_layout']['path']
            env.last_behavior=np.zeros(4);env.execution_outcome=np.zeros(6,dtype=np.float32)
            env.plant=SimpleNamespace(state=VehicleState(**initial['initial']),previous_command=Command(),delay_queue=deque(),evidence=initial['actuator_evidence'])
            group=layout['condition'];kept=0
            for row in rows:
                action=np.asarray(row['policy_action'],dtype=np.int64);trace=row['trace'];speed=env.state.v
                if not any(t['intervention'] or t['controller_status']=='failed' for t in trace):
                    raw.append(env._observation());targets.append(action);groups.append(group);kept+=1
                else:excluded+=1
                env.last_behavior=np.array([*flat_action_values(action,config),0.,0.])
                env.plant.state=VehicleState(**trace[-1]['state']);env.plant.previous_command=Command(**trace[-1]['command']);env.plant.evidence=trace[-1]['backend_evidence']
                env.execution_outcome=np.array([0.,0.,0.,sum(t['controller_status']=='failed' for t in trace)/len(trace),sum(t['intervention'] for t in trace)/len(trace),(env.state.v-speed)/config.vehicle.max_speed],dtype=np.float32)
            state=env.state
            if (np.hypot(state.x-env.goal[0],state.y-env.goal[1])>config.goal_tolerance or abs(state.v)>config.goal_speed or
                    abs(wrap(state.yaw-env.goal[2]))>config.goal_yaw_tolerance or env.world.out_of_bounds((state.x,state.y,state.yaw),config.vehicle)):
                raise ValueError('Stored success fails the actual terminal pose/speed/bounds gates')
            assert not hasattr(env.plant,'step');env.close()
            selection.append({'source_batch':batch.name,'method':name,'episode_id':i,'condition':group,'actions':kept,**quality})
        records.append({'batch':batch.name,'method':name,'raw_audit_sha256':sha(p/'endpoint_raw_audit.json'),
                        'all_training_episodes':len(episodes),'available_success_episodes':len(success),
                        'success_episodes_used':len(selection)-selected_before,'failures_retained_at_source':len(episodes)-len(success)})
    if not raw:raise ValueError('No own successful experience available')
    output=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_own_success_actor_fit');output.mkdir()
    observations=np.asarray(raw,dtype=np.float32);actions=np.asarray(targets,dtype=np.int64);groups=np.asarray(groups)
    np.savez_compressed(output/'own_success_inputs.npz',observations=observations,actions=actions,groups=groups)
    model=load_ppo(folder/'model.zip',device='cpu')
    with (folder/'normalization.pkl').open('rb') as stream:normalizer=pickle.load(stream)
    normalizer.training=False;normalizer.norm_reward=False
    means=normalizer.obs_rms.mean.copy();variance=normalizer.obs_rms.var.copy();count=normalizer.obs_rms.count
    latest=folder/read(folder/'latest_checkpoint.json')['folder']
    with (latest/'resume_state.pkl').open('rb') as stream:state=pickle.load(stream)
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    result=fit_actor(model,normalizer.normalize_obs(observations),actions,groups,steps)
    np.testing.assert_array_equal(means,normalizer.obs_rms.mean);np.testing.assert_array_equal(variance,normalizer.obs_rms.var);assert count==normalizer.obs_rms.count
    model.save(output/'model.zip');shutil.copy2(folder/'normalization.pkl',output/'normalization.pkl')
    state.update(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    with (output/'resume_state.pkl').open('wb') as stream:pickle.dump(state,stream)
    metadata={**result,'status':'complete','scope':'Auxiliary reward-selected replay of own PPO successes; no external teacher, new driving or PPO updates; not a competence claim',
        'source_batch':model_batch.name,'source_method':method,'source_model_sha256':sha(folder/'model.zip'),
        'source_policy_ppo_n_updates':model._n_updates,'source_raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'new_physics_steps':0,'samples':len(raw),'source_records':records,'episodes_used':selection,
        'condition_counts':dict(Counter(int(v) for v in groups)),'excluded_intervened_decisions':excluded,
        'success_selection':({'version':'terminal_margin_v1','distance_max_m':.5,'speed_max_m_s':.15,'yaw_max_rad':.2} if terminal_margin else {'version':'all_success_v1'}),
        'unselected_successes_retained_at_source':unselected,
        'script_sha256':sha(__file__),'dataset_sha256':sha(output/'own_success_inputs.npz')}
    for filename,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),('resume_state.pkl','resume_state_sha256')]:metadata[key]=sha(output/filename)
    config_record.update(scope=metadata['scope'],auxiliary_actor_fit={'optimizer_steps':steps,'own_success_samples':len(raw)})
    write(output/'config.json',config_record);write(output/'metadata.json',metadata)
    shutil.copy2(__file__,output/'fit_source.py')
    write(ROOT/'runs/own_success_actor_fit.json',{'artifact':str(output),'metadata_sha256':sha(output/'metadata.json')})
    print(json.dumps({'artifact':output.name,**result,'samples':len(raw),'conditions':metadata['condition_counts']}),flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-batch',required=True);p.add_argument('--method',default='B1')
    p.add_argument('--source',action='append',required=True);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--terminal-margin',action='store_true')
    a=p.parse_args();run(a.model_batch,a.method,a.source,a.steps,a.terminal_margin)

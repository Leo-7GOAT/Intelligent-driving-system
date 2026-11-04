"""Reconstruct closed native categorical PPO updates from audited feedback.

No simulator or mutable plant exists. Alternative standard-PPO branches are
discarded, fixed-data optimizer diagnostics, never on-policy driving evidence.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import pickle
import random
import sys

ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from navstack.config import EnvConfig
from navstack.experiments import actor_vector,write_json
from navstack.independent_clip_ppo import load_ppo
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_native_policy_sequence import observation_env,advance_observation
from scripts.audit_native_rollout_update import gradient_diagnostic,exact_kl


def equal(left,right):
    if torch.is_tensor(left):return torch.is_tensor(right) and torch.equal(left,right)
    if isinstance(left,np.ndarray):return isinstance(right,np.ndarray) and np.array_equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):return type(left)==type(right) and len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left==right


def high_reward(row,config):
    # DummyVecEnv casts the physical-time-discounted task reward to float32.
    return np.array([sum(config.gamma_low**j*t['reward'] for j,t in enumerate(row['trace']))],dtype=np.float32)


def rms_state(rms):return dict(mean=rms.mean,var=rms.var,count=rms.count)


def actor_group(name):
    if name.startswith('action_net.'):return 'shared_score_head'
    for part in ('road','context','candidate'):
        if name.startswith('mlp_extractor.policy_net_'+part+'.'):return part+'_encoder'
    return None


def actor_gradient_groups(model,multiplier):
    data=next(model.rollout_buffer.get(model.n_steps))
    _,log_prob,_=model.policy.evaluate_actions(data.observations,data.actions.long().flatten())
    advantages=(data.advantages-data.advantages.mean())/(data.advantages.std()+1e-8)
    loss=-(advantages*torch.exp(log_prob-data.old_log_prob)).mean()
    parameters=[(n,p) for n,p in model.policy.named_parameters() if actor_group(n)]
    gradients=torch.autograd.grad(loss,[p for _,p in parameters],allow_unused=True)
    grouped={}
    for (name,parameter),gradient in zip(parameters,gradients):
        group=grouped.setdefault(actor_group(name),[])
        group.append((torch.zeros_like(parameter) if gradient is None else gradient).flatten())
    epsilon=model.policy.optimizer.param_groups[0]['eps']
    return {name:dict(parameters=sum(v.numel() for v in values),
        policy_gradient_l2=float(torch.linalg.vector_norm(vector:=torch.cat(values))),
        post_global_clip_gradient_l2=float(torch.linalg.vector_norm(vector*multiplier)),
        fraction_after_clip_below_adam_epsilon=float(((vector*multiplier).abs()<epsilon).float().mean()))
        for name,values in grouped.items()}


def raw_logits(policy,observations):
    with torch.no_grad():
        features=policy.extract_features(torch.as_tensor(np.concatenate(observations)))
        return policy.action_net(policy.mlp_extractor.forward_actor(features)).numpy()


def parameter_group_changes(before,policy):
    grouped={}
    for name,parameter in policy.named_parameters():
        group=actor_group(name)
        if group:grouped.setdefault(group,[]).append((parameter.detach()-before[name]).flatten())
    return {name:float(torch.linalg.vector_norm(torch.cat(values))) for name,values in grouped.items()}


def analyze(batch,method,checkpoint_name,output):
    batch,output=Path(batch).resolve(),Path(output)
    if output.exists():raise FileExistsError('Preserve reconstructed-update evidence')
    if Path(checkpoint_name).name!=checkpoint_name or not checkpoint_name.startswith('checkpoint_'):
        raise ValueError('Select a direct checkpoint directory')
    protocol=read(batch/'protocol.json');folder=batch/method
    if method not in ('H0','H') or method not in protocol['methods'] or protocol.get('recovery') or protocol.get('jobs'):
        raise ValueError('Require a completed fresh categorical native method')
    for name,digest in protocol['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Use the exact frozen runtime source')
    audit=read(folder/'endpoint_raw_audit.json')
    for name,key in [('config.json','config_sha256'),('summary.json','summary_sha256'),
                     ('optimization.jsonl','optimization_sha256')]:
        if sha(folder/name)!=audit[key]:raise ValueError('Input differs from completed raw audit')
    checkpoint=folder/checkpoint_name;metadata=read(checkpoint/'metadata.json')
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),('resume_state.pkl','resume_state_sha256')]:
        if sha(checkpoint/name)!=metadata[key]:raise ValueError('Checkpoint artifact changed')
    if read(checkpoint/'config.json')!=read(folder/'config.json'):raise ValueError('Checkpoint config differs')
    config=EnvConfig.from_dict(read(folder/'config.json')['env'])
    rows=[]
    for path in sorted(folder.glob('transitions_*.jsonl.gz')):
        if sha(path)!=audit['transition_hashes'].get(path.name):raise ValueError('Transition stream is not raw-audited')
        with gzip.open(path,'rt',encoding='utf-8') as stream:rows.extend(map(json.loads,stream))
    summary=read(folder/'summary.json');offset=metadata['actions']
    remaining=len(rows)-offset
    if remaining<=0 or remaining%protocol['rollout_actions'] or remaining>256:
        raise ValueError('Select an earlier checkpoint with 1..256 remaining decisions in whole rollouts')
    torch.set_num_threads(1)
    model=load_ppo(checkpoint/'model.zip',device='cpu')
    target=load_ppo(folder/'model.zip',device='cpu')
    evaluation=read(folder/'evaluation.json')
    if sha(folder/'model.zip')!=evaluation['model_sha256'] or sha(folder/'normalization.pkl')!=evaluation['normalization_sha256']:
        raise ValueError('Final evaluated artifacts changed')
    if model.num_timesteps!=offset or model.n_steps!=protocol['rollout_actions']:
        raise ValueError('Checkpoint action/rollout counters differ')
    model.set_logger(configure(None,[]))
    with (checkpoint/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    with (folder/'normalization.pkl').open('rb') as stream:target_norm=pickle.load(stream)
    with (checkpoint/'resume_state.pkl').open('rb') as stream:rng=pickle.load(stream)
    if norm.norm_reward or not norm.training:raise ValueError('Require unnormalized rewards and updating observation statistics')
    norm.training=True
    def make_env(scene_index):
        record=read(folder/('scene_%04d.json'%scene_index))
        return observation_env(config,record['scene'],record['initial'])
    env=make_env(1);transient_return=np.zeros(1)
    def advance(env,row):
        advance_observation(env,dict(policy_action=row['policy_action'],info=row))
        if row['termination_reason']:
            env.close();return make_env(row['episode_id']+2)
        return env
    # Statistics already contain the entire prefix. Only reconstruct history and
    # the transient discounted-return accumulator omitted by VecNormalize pickle.
    for row in rows[:offset]:
        transient_return=transient_return*model.gamma+high_reward(row,config)
        if row['termination_reason']:transient_return[:]=0
        env=advance(env,row)
    norm.returns=transient_return
    raw=env._observation()[None];obs=norm.normalize_obs(raw)
    np.testing.assert_array_equal(obs,model._last_obs,err_msg='Saved checkpoint observation differs from measured history')
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    expected={r['actions']:r for r in map(json.loads,(folder/'optimization.jsonl').read_text(encoding='utf-8').splitlines())}
    records=[];first_buffer=None;first_distributions=None;first_observations=None;first_numpy=None
    discarded_steps=[0]
    def count_step(*unused):discarded_steps[0]+=1
    handle=model.policy.optimizer.register_step_post_hook(count_step)
    starts=model._last_episode_starts.copy()
    for begin in range(offset,len(rows),model.n_steps):
        model.policy.set_training_mode(False);model.rollout_buffer.reset()
        observations=[];distributions=[];selected=rows[begin:begin+model.n_steps]
        for index,row in enumerate(selected):
            observations.append(obs.copy())
            with torch.no_grad():
                tensor=torch.as_tensor(obs);action,value,log_prob=model.policy(tensor)
                distributions.append(copy.deepcopy(model.policy.get_distribution(tensor).distribution))
            np.testing.assert_array_equal(action.numpy()[0],row['policy_action'],err_msg=f'Action/RNG mismatch at {begin+index}')
            reward=high_reward(row,config)
            model.rollout_buffer.add(obs,action.numpy().reshape(1,-1),reward,starts,value,log_prob)
            env=advance(env,row)
            raw=env._observation()[None]
            norm.obs_rms.update(raw);obs=norm.normalize_obs(raw)
            norm._update_reward(reward)
            done=bool(row['termination_reason'])
            if done:norm.returns[:]=0
            starts=np.array([done]);model.num_timesteps+=1
            model._last_obs=obs;model._last_original_obs=raw;model._last_episode_starts=starts
        with torch.no_grad():last_values=model.policy.predict_values(torch.as_tensor(obs))
        model.rollout_buffer.compute_returns_and_advantage(last_values,starts)
        numpy_before=np.random.get_state()
        gradients=gradient_diagnostic(model);np.random.set_state(numpy_before)
        gradients['actor_modules']=actor_gradient_groups(model,gradients['global_clip_multiplier'])
        np.random.set_state(numpy_before)
        if first_buffer is None:
            first_buffer=copy.deepcopy(model.rollout_buffer);first_distributions=distributions
            first_observations=observations;first_numpy=numpy_before
        model._current_progress_remaining=0.;model.train()
        actual=expected[model.num_timesteps]
        if metadata['optimizer_steps']+discarded_steps[0]!=actual['optimizer_steps']:
            raise ValueError('Reconstructed optimizer-step count differs')
        for key in ('train/approx_kl','train/value_loss','train/policy_gradient_loss','train/entropy_loss','train/explained_variance'):
            np.testing.assert_allclose(model.logger.name_to_value[key],actual['metrics'][key],rtol=3e-4,atol=2e-6)
        records.append(dict(start_action=begin,decisions=len(selected),physical_controls=sum(len(r['trace']) for r in selected),
            terminal_transitions=sum(bool(r['termination_reason']) for r in selected),
            actual_optimizer_steps=actual['optimizer_steps'],baseline_metrics_reproduced=True,gradients=gradients))
    if not equal(model.policy.state_dict(),target.policy.state_dict()):raise ValueError('Final policy/value tensors do not reproduce exactly')
    if not equal(model.policy.optimizer.state_dict(),target.policy.optimizer.state_dict()):raise ValueError('Final Adam state does not reproduce exactly')
    if not equal(rms_state(norm.obs_rms),rms_state(target_norm.obs_rms)):raise ValueError('Final observation statistics differ')
    if not equal(rms_state(norm.ret_rms),rms_state(target_norm.ret_rms)):raise ValueError('Final return statistics differ')
    handle.remove()
    if model._n_updates!=target._n_updates or model.num_timesteps!=summary['actions']:
        raise ValueError('Final learning counters differ')
    branches=[]
    for label,epsilon,value_coefficient,epochs in (
        ('baseline',None,None,None),('adam_epsilon_1e-8',1e-8,None,None),
        ('value_coefficient_0.01',None,.01,None),('epochs_10',None,None,10)):
        branch=load_ppo(checkpoint/'model.zip',device='cpu');branch.set_logger(configure(None,[]))
        branch.rollout_buffer=copy.deepcopy(first_buffer);branch._current_progress_remaining=0.
        if epsilon is not None:
            for group in branch.policy.optimizer.param_groups:group['eps']=epsilon
        if value_coefficient is not None:branch.vf_coef=value_coefficient
        if epochs is not None:branch.n_epochs=epochs
        branch_handle=branch.policy.optimizer.register_step_post_hook(count_step)
        before_steps=discarded_steps[0]
        np.random.set_state(first_numpy);before=actor_vector(branch).clone()
        before_parameters={n:p.detach().clone() for n,p in branch.policy.named_parameters() if actor_group(n)}
        logits_before=raw_logits(branch.policy,first_observations)
        branch.train()
        branch_handle.remove()
        delta=raw_logits(branch.policy,first_observations)-logits_before
        common=delta.mean(axis=1,keepdims=True);centered=delta-common
        with torch.no_grad():
            new=[branch.policy.get_distribution(torch.as_tensor(o)).distribution for o in first_observations]
        branches.append(dict(variant=label,adam_epsilon=branch.policy.optimizer.param_groups[0]['eps'],
            value_coefficient=branch.vf_coef,epochs=branch.n_epochs,
            discarded_optimizer_steps=discarded_steps[0]-before_steps,
            actor_change_from_checkpoint=float(torch.linalg.vector_norm(actor_vector(branch)-before)),
            actor_module_parameter_changes=parameter_group_changes(before_parameters,branch.policy),
            raw_logit_common_shift_rms=float(np.sqrt(np.mean(common**2))),
            centered_logit_change_rms=float(np.sqrt(np.mean(centered**2))),
            exact_distribution_kl_mean=float(np.mean([exact_kl(a,b) for a,b in zip(first_distributions,new)])),
            maximum_probability_mean=float(np.mean([float(d.probs.max()) for d in new]))))
    assert not hasattr(env.plant,'step');env.close()
    result=dict(scope='Raw-audited measured-state/RNG reconstruction of actual native PPO updates; final tensors, Adam state and observation/return statistics exact. Alternative first-rollout optimizer branches are discarded fixed-data diagnostics, not new on-policy training or driving performance.',
        batch=batch.name,method=method,checkpoint=checkpoint_name,endpoint_raw_audit_sha256=sha(folder/'endpoint_raw_audit.json'),
        checkpoint_metadata_sha256=sha(checkpoint/'metadata.json'),script_sha256=sha(__file__),
        checkpoint_observation_exact=True,sampled_actions_exact=True,final_policy_tensors_exact=True,
        final_adam_state_exact=True,final_observation_statistics_exact=True,final_return_statistics_exact=True,
        decisions_reconstructed=remaining,physical_controls_reconstructed=sum(r['physical_controls'] for r in records),
        actual_rollouts=records,discarded_first_rollout_optimizer_branches=branches,
        new_environment_controls=0,live_optimizer_updates=0,discarded_offline_optimizer_steps=discarded_steps[0])
    write_json(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True)
    parser.add_argument('--method',choices=['H0','H'],required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();analyze(args.batch,args.method,args.checkpoint,args.output)

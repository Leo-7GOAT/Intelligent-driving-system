"""Bounded actor-only correction from explicitly counterfactual expert labels."""
import math
from copy import deepcopy
import numpy as np
import torch

from .independent_clip_ppo import actor_critic_parameters


def _exact(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(_exact(v,right[k]) for k,v in left.items())
    if isinstance(left,(tuple,list)):
        return type(left) is type(right) and len(left)==len(right) and all(_exact(a,b) for a,b in zip(left,right))
    return left==right


def fit_trained_corrective_actor(model,normalizer,observations,actions,weights,*,steps=512,
                                learning_rate=1e-4,seed=22,callback=None):
    """Explicit trained-policy intervention; separate Adam, never a PPO update.

    The retained PPO optimizer is deliberately untouched. Any later PPO phase
    must collect a fresh rollout after this disclosed actor intervention.
    """
    if (type(steps) is not int or not 1<=steps<=2000 or not math.isfinite(learning_rate)
            or not 0<learning_rate<=.01 or model.num_timesteps<=0 or model._n_updates<=0
            or not model.policy.optimizer.state):
        raise ValueError('Require a trained PPO policy and bounded explicit auxiliary fit')
    observations=np.asarray(observations,dtype=np.float32)
    actions=np.asarray(actions,dtype=np.float32);weights=np.asarray(weights,dtype=np.float64)
    count=len(observations)
    if (not count or observations.shape!=(count,*model.observation_space.shape)
            or actions.shape!=(count,2) or weights.shape!=(count,)
            or not all(np.isfinite(x).all() for x in (observations,actions,weights))
            or np.any(np.abs(actions)>1) or np.any(weights<=0)):
        raise ValueError('Invalid physical direct-control correction dataset/weights')
    before=(normalizer.obs_rms.mean.copy(),normalizer.obs_rms.var.copy(),float(normalizer.obs_rms.count))
    actor,critic=actor_critic_parameters(model.policy)
    frozen=[p.detach().clone() for p in critic]
    ppo_state=deepcopy(model.policy.optimizer.state_dict())
    counters=(model.num_timesteps,model._n_updates)
    x=torch.as_tensor(normalizer.normalize_obs(observations),device=model.device)
    y=torch.as_tensor(actions,device=model.device)
    optimizer=torch.optim.Adam(actor,lr=learning_rate)
    rng=np.random.default_rng(seed);probabilities=weights/weights.sum();history=[]
    mode=model.policy.training;model.policy.set_training_mode(True)
    try:
        for step in range(1,steps+1):
            indices=rng.choice(count,size=min(256,count),replace=True,p=probabilities)
            prediction=model.policy.get_distribution(x[indices]).distribution.mean
            loss=(prediction-y[indices]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite corrective loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor,.5,error_if_nonfinite=True);optimizer.step()
            if step==1 or step%64==0 or step==steps:
                row={'additional_supervised_steps':step,'sampled_action_mse':float(loss.detach()),
                     'learning_rate':learning_rate,'PPO_updates':0}
                history.append(row)
                if callback:callback(row)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.policy.set_training_mode(mode)
    if (not all(torch.equal(p,q) for p,q in zip(critic,frozen))
            or not _exact(ppo_state,model.policy.optimizer.state_dict())
            or counters!=(model.num_timesteps,model._n_updates)):
        raise ValueError('Auxiliary correction changed critic, retained PPO optimizer or counters')
    np.testing.assert_array_equal(normalizer.obs_rms.mean,before[0])
    np.testing.assert_array_equal(normalizer.obs_rms.var,before[1])
    if float(normalizer.obs_rms.count)!=before[2]:raise ValueError('Auxiliary correction changed normalization')
    return {'version':'trained_actor_separate_Adam_v1','additional_supervised_steps':steps,
            'history':history,'critic_exact':True,'retained_PPO_optimizer_exact':True,
            'retained_PPO_counters_exact':True,'normalization_statistics_exact':True,'PPO_updates':0,
            'scope':'TRAIN-only offline actor intervention; no holdout accuracy or new-driving claim; fresh rollout required before PPO'}


def fit_corrective_actor(model,normalizer,observations,actions,weights,*,steps=544,
                         learning_rate=1e-4,seed=11,callback=None):
    if (type(steps) is not int or not 1<=steps<=2000 or not math.isfinite(learning_rate)
            or not 0<learning_rate<=.01 or model.num_timesteps or model._n_updates
            or model.policy.optimizer.state):
        raise ValueError('Require a zero-PPO initializer and bounded supervised updates')
    observations=np.asarray(observations,dtype=np.float32)
    actions=np.asarray(actions,dtype=np.float32);weights=np.asarray(weights,dtype=np.float64)
    count=len(observations)
    if (not count or observations.shape!=(count,*model.observation_space.shape)
            or actions.shape!=(count,2) or weights.shape!=(count,)
            or not all(np.isfinite(x).all() for x in (observations,actions,weights))
            or np.any(np.abs(actions)>1) or np.any(weights<=0)):
        raise ValueError('Invalid physical direct-control correction dataset/weights')
    before=(normalizer.obs_rms.mean.copy(),normalizer.obs_rms.var.copy(),float(normalizer.obs_rms.count))
    actor,critic=actor_critic_parameters(model.policy)
    frozen=[p.detach().clone() for p in critic]
    x=torch.as_tensor(normalizer.normalize_obs(observations),device=model.device)
    y=torch.as_tensor(actions,device=model.device)
    optimizer=torch.optim.Adam(actor,lr=learning_rate)
    rng=np.random.default_rng(seed);probabilities=weights/weights.sum();history=[]
    mode=model.policy.training;model.policy.set_training_mode(True)
    try:
        for step in range(1,steps+1):
            indices=rng.choice(count,size=min(256,count),replace=True,p=probabilities)
            prediction=model.policy.get_distribution(x[indices]).distribution.mean
            loss=(prediction-y[indices]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite corrective loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor,.5);optimizer.step()
            if step==1 or step%64==0 or step==steps:
                row={'additional_supervised_steps':step,'sampled_action_mse':float(loss.detach()),
                     'learning_rate':learning_rate,'PPO_updates':0}
                history.append(row)
                if callback:callback(row)
    finally:model.policy.set_training_mode(mode)
    assert all(torch.equal(p,q) for p,q in zip(critic,frozen))
    np.testing.assert_array_equal(normalizer.obs_rms.mean,before[0])
    np.testing.assert_array_equal(normalizer.obs_rms.var,before[1])
    assert float(normalizer.obs_rms.count)==before[2]
    assert not model.policy.optimizer.state and model.num_timesteps==model._n_updates==0
    return {'additional_supervised_steps':steps,'history':history,'critic_exact':True,
            'normalization_statistics_exact':True,'PPO_updates':0,
            'scope':'All supplied teacher/corrective rows are training. No holdout accuracy or new-driving claim.'}

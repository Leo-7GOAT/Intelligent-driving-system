"""Bounded actor-only correction from explicitly counterfactual expert labels."""
import math
import numpy as np
import torch

from .independent_clip_ppo import actor_critic_parameters


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

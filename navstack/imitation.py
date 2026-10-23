"""Explicit demonstration diagnostics, separate from on-policy PPO interaction.

Demonstrations come from executed planner/controller episodes. Replaying their
commands in a direct-control plant produces method-correct previous-action and
execution-outcome observations, rather than copying hierarchical observations.
"""
from dataclasses import asdict
import numpy as np
from .config import EnvConfig
from .env import NavigationEnv


def command_action(command, vehicle):
    acceleration=command['acceleration']
    return np.array([acceleration/(vehicle.max_accel if acceleration>=0 else vehicle.max_decel),
                     command['steering']/vehicle.max_steer],dtype=np.float32).clip(-1,1)


def replay_demonstration(config, seed, ticks, options=None):
    """Execute a separate real plant replay; never assign intermediate states."""
    cfg=EnvConfig.from_dict(config.to_dict());cfg.method='B1';cfg.flat_hold=1
    env=NavigationEnv(cfg);obs,_=env.reset(seed=seed,options=options)
    observations=[];actions=[];rewards=[];traces=[];maximum=0.
    initial=env.initial_scenario
    try:
        for source in ticks:
            action=command_action(source['requested_command'],cfg.vehicle)
            observations.append(obs.copy());actions.append(action)
            obs,reward,done,_,info=env.step(action)
            measured=info['trace'][0]['state']
            error=max(abs(measured[key]-source['state'][key]) for key in ('x','y','yaw','v','steering','acceleration','time'))
            maximum=max(maximum,error)
            rewards.append(reward);traces.extend(info['trace'])
            if maximum>1e-3:raise ValueError('Expert/direct physical replay diverged: '+str(maximum))
            if done:break
        # A completed source episode must also terminate in direct replay.
        if not env.done or len(traces)!=len(ticks):
            raise ValueError('Expert/direct episode termination does not align')
        returns=np.empty(len(rewards),dtype=np.float32);value=0.
        for i in reversed(range(len(rewards))):
            value=rewards[i]+cfg.gamma_low*value;returns[i]=value
        return {'observations':np.array(observations,dtype=np.float32),'actions':np.array(actions,dtype=np.float32),
            'returns':returns,'trace':traces,'initial':initial,'metrics':info['episode_metrics'],
            'max_state_difference':maximum,'terminal_state':asdict(env.state)}
    finally:env.close()


def fit_demonstrations(model, normalizer, observations, actions, returns, train_mask,
                       epochs=200, seed=11, callback=None):
    """Supervised actor/value initialization; never call it a PPO update.

    Normalization is fit only on training examples. Holdout examples report
    error, never provide gradients or normalization statistics. The separate
    Adam optimizer leaves PPO's initially empty optimizer state untouched.
    """
    import torch
    train_mask=np.asarray(train_mask,dtype=bool)
    if not train_mask.any() or train_mask.all():raise ValueError('Need separate train and holdout examples')
    if len(observations)!=len(actions) or len(actions)!=len(returns) or len(returns)!=len(train_mask):
        raise ValueError('Dataset lengths differ')
    if not all(np.isfinite(x).all() for x in (observations,actions,returns)):raise ValueError('Non-finite demonstration')
    normalizer.obs_rms.update(observations[train_mask])
    normalized=normalizer.normalize_obs(observations)
    x=torch.as_tensor(normalized,dtype=torch.float32);y=torch.as_tensor(actions,dtype=torch.float32)
    target=torch.as_tensor(returns[:,None],dtype=torch.float32)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed);indices=np.flatnonzero(train_mask);steps=0;history=[]
    model.policy.set_training_mode(True)
    def losses(index):
        prediction=model.policy.get_distribution(x[index]).distribution.mean
        actor=(prediction-y[index]).square().mean()
        value=(model.policy.predict_values(x[index])-target[index]).square().mean()
        return actor,value
    for epoch in range(epochs):
        for index in np.array_split(rng.permutation(indices),max(1,int(np.ceil(len(indices)/256)))):
            actor,value=losses(index)
            optimizer.zero_grad();(actor+.5*value).backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5)
            optimizer.step();steps+=1
        if (epoch+1)%10==0 or epoch==0 or epoch+1==epochs:
            with torch.no_grad():
                ta,tv=losses(np.flatnonzero(train_mask));ha,hv=losses(np.flatnonzero(~train_mask))
            row={'supervised_epoch':epoch+1,'supervised_optimizer_steps':steps,'train_action_mse':float(ta),
                'holdout_action_mse':float(ha),'train_value_mse':float(tv),'holdout_value_mse':float(hv)}
            history.append(row)
            if callback:callback(row)
    return history

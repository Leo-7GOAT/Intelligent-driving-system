"""Explicit demonstration diagnostics, separate from on-policy PPO interaction.

Demonstrations come from executed planner/controller episodes. Replaying their
commands in a direct-control plant produces method-correct previous-action and
execution-outcome observations, rather than copying hierarchical observations.
"""
from dataclasses import asdict
import numpy as np
from .config import EnvConfig
from .env import NavigationEnv


class RecoveryTeacher:
    """Training-data oracle on actual learner states, never an env-step hook.

    A private planning/controller workspace reads the live plant and world.
    Querying it neither advances the plant nor changes learner observations or
    episode accounting. A collector must explicitly choose the executed action
    and separately charge all teacher query costs. Learned evaluation never
    instantiates this helper.
    """
    def __init__(self, env):
        from copy import copy, deepcopy
        from .mpc import MPC
        if env.config.method != 'B1' or env.config.flat_hold != 1 or env.done:
            raise ValueError('Teacher queries require an active hold1 direct-control episode')
        self.learner = env
        self.workspace = copy(env)
        self.workspace.episode = deepcopy(env.episode)
        self.workspace.controller = MPC(env.config.vehicle, env.config.control_dt, env.config.mpc_horizon)
        self.workspace._rule_plan_cache = None
        self.workspace.reference = None
        self.controls_per_decision = round(env.config.decision_dt/env.config.control_dt)
        self.last_control = None
        self.mode, self.plan, self.action = None, None, None

    def query(self):
        import time
        from copy import deepcopy
        from .env import rule_action
        from .vehicle import brake
        env, oracle = self.learner, self.workspace
        if env.done or env.plant is not oracle.plant:
            raise ValueError('Create a new teacher workspace after an episode reset')
        control = env.episode['physics_steps']
        if self.last_control is not None and control != self.last_control+1:
            raise ValueError('Query exactly once before each actually executed control')
        state_before, metrics_before = deepcopy(env.state), deepcopy(env.episode)
        started = time.perf_counter()
        decision = self.last_control is None or control % self.controls_per_decision == 0
        if decision:
            self.action = rule_action(oracle)
            self.mode, self.plan = oracle._prepare_reference(self.action)
        if self.mode == 'reference':
            result = oracle.controller.step(env.state, oracle.reference, env.plant.previous_command)
            command, status = result.command, result.status
        else:
            command, status = brake(env.state, env.config.vehicle), self.mode
        label = command_action(asdict(command), env.config.vehicle)
        metadata = {'control': control, 'teacher_decision': decision,
                    'teacher_high_action': self.action, 'teacher_mode': self.mode,
                    'teacher_plan': self.plan if decision else None,
                    'teacher_rule_diagnostics': oracle.rule_diagnostics if decision else None,
                    'teacher_controller_status': status, 'query_seconds': time.perf_counter()-started,
                    'requested_command': asdict(command)}
        assert env.state == state_before and env.episode == metrics_before
        self.last_control = control
        return label, metadata


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
                       epochs=200, seed=11, callback=None, update_normalization=True,
                       value_weight=.5):
    """Supervised actor/value initialization; never call it a PPO update.

    Normalization is fit only on training examples. Holdout examples report
    error, never provide gradients or normalization statistics. The separate
    Adam optimizer leaves PPO's initially empty optimizer state untouched.
    """
    import torch
    if not np.isfinite(value_weight) or value_weight < 0:
        raise ValueError('Supervised value weight must be finite and nonnegative')
    train_mask=np.asarray(train_mask,dtype=bool)
    if not train_mask.any() or train_mask.all():raise ValueError('Need separate train and holdout examples')
    if len(observations)!=len(actions) or len(actions)!=len(returns) or len(returns)!=len(train_mask):
        raise ValueError('Dataset lengths differ')
    if not all(np.isfinite(x).all() for x in (observations,actions,returns)):raise ValueError('Non-finite demonstration')
    if update_normalization:
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
            optimizer.zero_grad()
            (actor+value_weight*value if value_weight else actor).backward()
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


def collect_recovery_episode(config, model, normalizer, seed, teacher_probability,
                             mixture_seed, options=None, callback=None,
                             teacher_mode='held_reference'):
    """Execute a disclosed teacher/learner mixture, label its actual states.

    Neither the teacher label nor mixture success is a learned-policy result.
    The selection probability is fixed by the collection protocol. Every
    visited state, including failed episodes, contributes an example; callers
    separately exclude declared heldout cases from fitting.
    """
    if not 0 <= teacher_probability <= 1:
        raise ValueError('Invalid teacher mixture probability')
    if teacher_mode not in ('held_reference','current_state'):
        raise ValueError('Unknown teacher labeling mode')
    env = NavigationEnv(config)
    obs, _ = env.reset(seed=seed, options=options)
    oracle = RecoveryTeacher(env)
    rng = np.random.default_rng(mixture_seed)
    observations, labels, rewards, sources = [], [], [], []
    rows = []
    initial = env.initial_scenario
    try:
        while not env.done:
            state_before = asdict(env.state)
            prediction, _ = model.predict(normalizer.normalize_obs(obs[None]), deterministic=True)
            learner = prediction[0]
            label, query = (RecoveryTeacher(env) if teacher_mode=='current_state' else oracle).query()
            # Mixture execution scheduling is independent of label query rate.
            # The current-state oracle replans every control, but does not get
            # extra opportunities to replace a learner action within a block.
            if len(observations) % oracle.controls_per_decision == 0:
                use_teacher = bool(rng.random() < teacher_probability)
            action = label if use_teacher else learner
            observations.append(obs.copy()); labels.append(label.copy()); sources.append(use_teacher)
            obs, reward, _, _, info = env.step(action)
            rewards.append(reward)
            row = {'state_before': state_before, 'learner_action': learner.copy(),
                   'teacher_action': label.copy(), 'executed_action': action.copy(),
                   'executed_by': 'teacher' if use_teacher else 'learner',
                   'teacher_query': query, 'info': info}
            rows.append(row)
            if callback:
                callback(row)
        returns = np.empty(len(rewards), dtype=np.float32)
        value = 0.
        for i in reversed(range(len(rewards))):
            value = rewards[i]+config.gamma_low*value
            returns[i] = value
        return {'observations': np.asarray(observations, dtype=np.float32),
                'actions': np.asarray(labels, dtype=np.float32), 'returns': returns,
                'teacher_executed': np.asarray(sources, dtype=bool), 'initial': initial,
                'metrics': info['episode_metrics'], 'teacher_queries': len(rows),
                'teacher_decisions': sum(r['teacher_query']['teacher_decision'] for r in rows),
                'teacher_query_seconds': sum(r['teacher_query']['query_seconds'] for r in rows),
                'teacher_control_failures': sum(r['teacher_query']['teacher_controller_status']=='failed' for r in rows)}
    finally:
        env.close()

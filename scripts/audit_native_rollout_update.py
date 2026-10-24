"""Rebuild the first native continuation rollout from measured states and saved RNG.

No simulator connection or plant integration is possible. Discarded in-memory
optimizer branches diagnose optimization, not driving performance or fresh PPO.
"""
import argparse
from collections import deque
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace

ROOT = Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from navstack.config import EnvConfig
from navstack.env import NavigationEnv, ACTIONS
from navstack.experiments import actor_vector
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehicleState, Command, wrap
from navstack.world import World

CLIP_MODULE_PATH = Path(__file__).resolve().parents[1]/'navstack/independent_clip_ppo.py'
_clip_spec = importlib.util.spec_from_file_location('audited_independent_clip', CLIP_MODULE_PATH)
_clip_module = importlib.util.module_from_spec(_clip_spec)
_clip_spec.loader.exec_module(_clip_module)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured_episode(config, scene_record):
    scene, initial = scene_record['scene'], scene_record['initial']
    if initial['static'] or initial['dynamic']:
        raise ValueError('Reconstruction supports empty measured native roads only')
    env = NavigationEnv(config)
    env.world = World(scene['width'], scene['height'], road=RoadGrid(scene['road_grid']))
    env.goal = initial['goal']
    env.last_behavior = np.zeros(4)
    env.execution_outcome = np.zeros(6, dtype=np.float32)
    env.plant = SimpleNamespace(state=VehicleState(**initial['initial']),
                               previous_command=Command(), delay_queue=deque())
    return env


def advance_measurements(env, row):
    action = row['policy_action']
    speed_before = env.state.v
    if env.config.method == 'B1':
        env.last_behavior = np.array([*np.clip(action, -1, 1), 0., 0.])
    elif env.config.subgoal_mode.startswith('goal_frame') and int(action):
        target, speed = env.decode_action(int(action))
        dx, dy = target[0]-env.state.x, target[1]-env.state.y
        co, si = np.cos(env.state.yaw), np.sin(env.state.yaw)
        env.last_behavior = np.array([co*dx+si*dy, -si*dx+co*dy,
                                      wrap(target[2]-env.state.yaw), speed])
    else:
        env.last_behavior = np.array(ACTIONS[int(action)])
    ticks = row['trace']
    env.plant.state = VehicleState(**ticks[-1]['state'])
    env.plant.previous_command = Command(**ticks[-1]['command'])
    if env.config.execution_feedback != 'off':
        env.execution_outcome = np.array([
            float(row['mode'] == 'reference'), float(row['mode'] == 'plan_failed'),
            float(row['mode'] == 'dynamic_brake'),
            sum(t['controller_status'] == 'failed' for t in ticks)/len(ticks),
            sum(t['intervention'] for t in ticks)/len(ticks),
            (env.state.v-speed_before)/env.config.vehicle.max_speed], dtype=np.float32)


def normalize_new_observation(norm, raw):
    norm.obs_rms.update(raw)
    return norm.normalize_obs(raw)


def gradient_diagnostic(model):
    data = next(model.rollout_buffer.get(model.n_steps))
    actions = data.actions.long().flatten() if model.action_space.shape == () else data.actions
    values, log_prob, entropy = model.policy.evaluate_actions(data.observations, actions)
    advantage = (data.advantages-data.advantages.mean())/(data.advantages.std()+1e-8)
    policy_loss = -(advantage*torch.exp(log_prob-data.old_log_prob)).mean()
    value_loss = model.vf_coef*torch.nn.functional.mse_loss(data.returns, values.flatten())
    params = list(model.policy.named_parameters())
    pg = torch.autograd.grad(policy_loss, [p for _, p in params], retain_graph=True, allow_unused=True)
    vg = torch.autograd.grad(value_loss, [p for _, p in params], allow_unused=True)
    groups = {}
    all_squared = 0.
    for label, predicate in (
        ('actor', lambda n: 'policy_net' in n or 'action_net' in n or n == 'log_std'),
        ('critic', lambda n: 'value_net' in n),
    ):
        p2 = v2 = combined2 = 0.
        sizes, tiny = 0, 0
        epsilon = float(model.policy.optimizer.param_groups[0].get('eps', 0.))
        for (name, parameter), a, b in zip(params, pg, vg):
            if not predicate(name):
                continue
            a = torch.zeros_like(parameter) if a is None else a
            b = torch.zeros_like(parameter) if b is None else b
            p2 += float(a.square().sum())
            v2 += float(b.square().sum())
            combined = a+b
            combined2 += float(combined.square().sum())
            sizes += combined.numel()
            tiny += int((combined.abs() < epsilon).sum())
        groups[label] = {'policy_gradient_l2': p2**.5, 'value_gradient_l2': v2**.5,
                         'combined_l2': combined2**.5,
                         'fraction_components_below_adam_epsilon': tiny/max(1, sizes)}
        all_squared += combined2
    multiplier = min(1., model.max_grad_norm/(all_squared**.5+1e-6))
    for label, predicate in (
        ('actor', lambda n: 'policy_net' in n or 'action_net' in n or n == 'log_std'),
        ('critic', lambda n: 'value_net' in n),
    ):
        components = []
        for (name, parameter), a, b in zip(params, pg, vg):
            if predicate(name):
                a = torch.zeros_like(parameter) if a is None else a
                b = torch.zeros_like(parameter) if b is None else b
                components.append(((a+b)*multiplier).flatten())
        groups[label]['fraction_after_clip_below_adam_epsilon'] = float(
            (torch.cat(components).abs() < epsilon).float().mean())
    return {'groups': groups, 'total_gradient_l2': all_squared**.5,
            'global_clip_multiplier': multiplier,
            'adam_epsilon': epsilon, 'advantage_std': float(data.advantages.std()),
            'policy_loss': float(policy_loss.detach()), 'weighted_value_loss': float(value_loss.detach())}


def exact_kl(old, new):
    # Float32 categorical sums can produce tiny negative KL near equality.
    # Re-normalize in float64 before measuring the full distribution divergence.
    if isinstance(old, torch.distributions.Categorical):
        p, q = old.probs.double(), new.probs.double()
        p, q = p/p.sum(-1, keepdim=True), q/q.sum(-1, keepdim=True)
        return float((p*(p.log()-q.log())).sum())
    if isinstance(old, torch.distributions.Normal):
        return float(torch.distributions.kl_divergence(
            torch.distributions.Normal(old.mean.double(), old.stddev.double()),
            torch.distributions.Normal(new.mean.double(), new.stddev.double())).sum())
    raise TypeError('Unsupported action distribution')


def audit(batch, method, output):
    batch, output = Path(batch).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous rollout evidence')
    protocol = read(batch/'protocol.json')
    for name, digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name) != digest:
            raise ValueError('Frozen observation and policy source required')
    folder = batch/method
    recovery = protocol['recovery'][method]
    checkpoint = batch/recovery['checkpoint']
    metadata = read(checkpoint/'metadata.json')
    for filename, key in [('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256'),
                          ('resume_state.pkl', 'resume_state_sha256')]:
        if sha(checkpoint/filename) != metadata[key]:
            raise ValueError('Continuation checkpoint changed')
    config = EnvConfig.from_dict(read(folder/'config.json')['env'])
    torch.set_num_threads(1)
    model = PPO.load(checkpoint/'model.zip', device='cpu')
    model.set_logger(configure(None, []))
    with (checkpoint/'normalization.pkl').open('rb') as stream:
        norm = pickle.load(stream)
    norm.training = True
    with (checkpoint/'resume_state.pkl').open('rb') as stream:
        rng = pickle.load(stream)
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
    env = measured_episode(config, read(folder/'scene_0001.json'))
    obs = normalize_new_observation(norm, env._observation()[None])
    model.rollout_buffer.reset()
    model.policy.set_training_mode(False)
    rows = []
    paths = sorted(folder.glob('transitions_*.jsonl.gz'))
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                rows.append(json.loads(line))
                if len(rows) == model.n_steps:
                    break
        if len(rows) == model.n_steps:
            break
    if len(rows) != model.n_steps:
        raise ValueError('First rollout is incomplete')
    episode_start = np.ones(1, dtype=bool)
    observations, old_distributions = [], []
    for index, row in enumerate(rows):
        observations.append(obs.copy())
        with torch.no_grad():
            tensor = torch.as_tensor(obs)
            action, value, log_prob = model.policy(tensor)
            distribution = model.policy.get_distribution(tensor).distribution
            old_distributions.append(copy.deepcopy(distribution))
        np.testing.assert_array_equal(action.numpy()[0], row['policy_action'],
                                      err_msg=f'RNG/action mismatch at decision {index}')
        reward = np.array([sum(config.gamma_low**i*t['reward'] for i, t in enumerate(row['trace']))], dtype=np.float32)
        model.rollout_buffer.add(obs, action.numpy().reshape(1, -1), reward, episode_start, value, log_prob)
        advance_measurements(env, row)
        done = bool(row['termination_reason'])
        if done:
            env = measured_episode(config, read(folder/f"scene_{row['episode_id']+2:04d}.json"))
        obs = normalize_new_observation(norm, env._observation()[None])
        episode_start = np.array([done])
    with torch.no_grad():
        last_values = model.policy.predict_values(torch.as_tensor(obs))
    model.rollout_buffer.compute_returns_and_advantage(last_values, episode_start)
    # Diagnostics must not consume the shuffle RNG used by the actual PPO update.
    numpy_before = np.random.get_state()
    gradients = gradient_diagnostic(model)
    np.random.set_state(numpy_before)
    branches = []
    variants = [('baseline', model.n_epochs, None, False), ('epochs_10', 10, None, False),
                ('epochs_32', 32, None, False), ('adam_epsilon_1e-8', model.n_epochs, 1e-8, False),
                ('independent_actor_critic_clip', model.n_epochs, None, True)]
    for label, epochs, epsilon, separate_clip in variants:
        cls = _clip_module.IndependentClipPPO if separate_clip else PPO
        branch = cls.load(checkpoint/'model.zip', device='cpu')
        branch.rollout_buffer = copy.deepcopy(model.rollout_buffer)
        branch.n_epochs = epochs
        branch._current_progress_remaining = 0.
        branch.set_logger(configure(None, []))
        if epsilon is not None:
            for group in branch.policy.optimizer.param_groups:
                group['eps'] = epsilon
        np.random.set_state(numpy_before)
        before = actor_vector(branch).clone()
        branch.train()
        kl = []
        maxima = []
        with torch.no_grad():
            for observation, old in zip(observations, old_distributions):
                new = branch.policy.get_distribution(torch.as_tensor(observation)).distribution
                kl.append(exact_kl(old, new))
                if hasattr(new, 'probs'):
                    maxima.append(float(new.probs.max()))
        branches.append({'variant': label, 'epochs': epochs, 'adam_epsilon': branch.policy.optimizer.param_groups[0]['eps'],
                         'independent_actor_critic_clip': separate_clip,
                         'actor_l2_change': float(torch.linalg.vector_norm(actor_vector(branch)-before)),
                         'exact_distribution_kl_mean': float(np.mean(kl)),
                         'maximum_action_probability_mean': float(np.mean(maxima)) if maxima else None,
                         'metrics': {k: float(v) for k, v in branch.logger.name_to_value.items()}})
    actual = json.loads((folder/'optimization.jsonl').open(encoding='utf-8').readline())
    np.testing.assert_allclose(branches[0]['actor_l2_change'], actual['actor_l2_change'], rtol=2e-4, atol=1e-7)
    for key in ('train/approx_kl', 'train/value_loss', 'train/policy_gradient_loss', 'train/entropy_loss'):
        np.testing.assert_allclose(branches[0]['metrics'][key], actual['metrics'][key], rtol=3e-4, atol=2e-6)
    result = {'scope': 'First actual native rollout reconstructed from measured states, checkpoint normalization and RNG; all sampled actions exact; baseline optimizer metrics reproduced. Alternative update branches are discarded offline diagnostics on the same rollout, not new on-policy driving or performance evidence.',
              'batch': batch.name, 'method': method, 'rollout_actions': len(rows),
              'physical_controls': sum(len(row['trace']) for row in rows),
              'new_environment_controls': 0, 'gradients': gradients, 'offline_update_branches': branches,
              'sampled_actions_exact': True, 'baseline_update_reproduced': True,
              'checkpoint_model_sha256': sha(checkpoint/'model.zip'),
              'checkpoint_normalization_sha256': sha(checkpoint/'normalization.pkl'),
              'checkpoint_rng_sha256': sha(checkpoint/'resume_state.pkl'),
              'first_rollout_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
              'script_sha256': sha(Path(__file__)), 'frozen_source_root': str(ROOT)}
    result['independent_clipping_implementation_sha256'] = sha(CLIP_MODULE_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--method', required=True, choices=['H0', 'H', 'B1'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.batch, args.method, args.output)

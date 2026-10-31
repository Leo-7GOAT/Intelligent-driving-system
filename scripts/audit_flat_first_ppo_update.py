"""Reconstruct the first completed flat native PPO rollout and discard LR probes.

No simulator is connected and no alternative policy is saved or deployed.
"""
import argparse
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT', REPO)).resolve()
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from navstack.config import EnvConfig
from navstack.independent_clip_ppo import load_ppo
from navstack.measured_demo import measured_flat_env
from scripts.audit_native_method_endpoint import read, sha

_helper_spec = importlib.util.spec_from_file_location(
    'flat_update_measured_history', REPO / 'scripts/audit_flat_terminal_policy_drift.py')
_helper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper)
advance_recorded_flat = _helper.advance_recorded_flat


def analyze(batch, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise FileExistsError('Preserve earlier diagnostic')
    started = time.monotonic()
    protocol = read(batch / 'protocol.json')
    folder = batch / 'B1'
    audit = read(folder / 'endpoint_raw_audit.json')
    if not protocol.get('freeze_observation_statistics') or protocol.get('own_success_replay'):
        raise ValueError('Only frozen-statistics PPO without auxiliary replay is supported')
    for name, digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT / name) != digest:
            raise ValueError('Use the archived runtime: ' + name)
    for name, key in [('config.json', 'config_sha256'), ('optimization.jsonl', 'optimization_sha256')]:
        if sha(folder / name) != audit[key]:
            raise ValueError('Raw-audited input differs')
    origin = batch / protocol['recovery']['B1']['checkpoint']
    metadata = read(origin / 'metadata.json')
    for name, key in [('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256'),
                      ('resume_state.pkl', 'resume_state_sha256')]:
        if sha(origin / name) != metadata[key]:
            raise ValueError('Initializer artifact differs')
    torch.set_num_threads(1)
    model = load_ppo(origin / 'model.zip', device='cpu')
    if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
        raise ValueError('Require a zero-PPO initializer')
    baseline_rate = float(model.lr_schedule(1.))
    if not 0 < baseline_rate <= 1e-3 or model.lr_schedule(0.) != baseline_rate:
        raise ValueError('Require the registered finite constant learning rate')
    with (origin / 'normalization.pkl').open('rb') as stream:
        norm = pickle.load(stream)
    norm.training = False
    if norm.norm_reward:
        raise ValueError('Require unnormalized physical rewards')
    with (origin / 'resume_state.pkl').open('rb') as stream:
        rng = pickle.load(stream)
    rows = []
    for name, digest in sorted(audit['transition_hashes'].items()):
        path = folder / name
        if sha(path) != digest:
            raise ValueError('Recorded transition stream differs')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                if len(rows) <= model.n_steps:
                    rows.append(json.loads(line))
    if len(rows) < model.n_steps:
        raise ValueError('First rollout is incomplete')
    responses = batch / 'infrastructure' / audit['bridge_response_stream']
    if sha(responses) != audit['bridge_response_sha256']:
        raise ValueError('Raw response stream differs')
    begins = {}
    with gzip.open(responses, 'rt', encoding='utf-8') as stream:
        for line in stream:
            response = json.loads(line).get('response', {})
            if 'scene' in response:
                begins[response['episode_id']] = response
    config = EnvConfig.from_dict(read(folder / 'config.json')['env'])

    def episode(row):
        record = read(folder / ('scene_%04d.json' % (row['episode_id'] + 1)))
        begin = begins[row['trace'][0]['backend_evidence']['episode_id']]
        return measured_flat_env(config, record['scene'], record['initial'], begin=begin)

    env = episode(rows[0])
    model.policy.set_training_mode(False)
    model.rollout_buffer.reset()
    obs = norm.normalize_obs(env._observation()[None])
    episode_start = np.ones(1, dtype=bool)
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    try:
        for index, row in enumerate(rows[:model.n_steps]):
            if time.monotonic() - started > 600:
                raise TimeoutError('Bounded reconstruction expired')
            with torch.no_grad():
                action, value, log_prob = model.policy(torch.as_tensor(obs))
            np.testing.assert_array_equal(action.numpy()[0], row['policy_action'],
                                          err_msg='Sampled action differs at %d' % index)
            reward = np.array([sum(config.gamma_low ** j * tick['reward']
                                   for j, tick in enumerate(row['trace']))], dtype=np.float32)
            # Registered fixed reward units still apply when adaptive return
            # normalization is disabled. Match VecNormalize.step_wait exactly.
            reward = norm.normalize_reward(reward)
            model.rollout_buffer.add(obs, action.numpy(), reward, episode_start, value, log_prob)
            if len(row['trace']) != 1:
                raise ValueError('Require one measured control per flat decision')
            advance_recorded_flat(env, row['trace'][0], row['policy_action'])
            done = bool(row['termination_reason'])
            if done:
                if index + 1 >= len(rows):
                    raise ValueError('Need next reset observation')
                env.close()
                env = episode(rows[index + 1])
            obs = norm.normalize_obs(env._observation()[None])
            episode_start = np.array([done])
        with torch.no_grad():
            last_values = model.policy.predict_values(torch.as_tensor(obs))
        model.rollout_buffer.compute_returns_and_advantage(last_values, episode_start)
    finally:
        env.close()
    shuffle = np.random.get_state()
    observations = torch.as_tensor(model.rollout_buffer.observations[:, 0].copy())
    with torch.no_grad():
        old = model.policy.get_distribution(observations).distribution
        old_mean, old_std = old.mean.double().clone(), old.stddev.double().clone()
    actual = json.loads((folder / 'optimization.jsonl').read_text().splitlines()[0])
    target_dir = folder / ('checkpoint_%09d' % model.n_steps)
    target_meta = read(target_dir / 'metadata.json')
    if sha(target_dir / 'model.zip') != target_meta['model_sha256']:
        raise ValueError('First completed checkpoint differs')
    target = load_ppo(target_dir / 'model.zip', device='cpu')
    branches = []
    for rate in dict.fromkeys((baseline_rate, 3e-5, 1e-5, 3e-6)):
        branch = load_ppo(origin / 'model.zip', device='cpu')
        branch.rollout_buffer = copy.deepcopy(model.rollout_buffer)
        branch.lr_schedule = lambda progress, rate=rate: rate
        branch._current_progress_remaining = 0.
        branch.set_logger(configure(None, []))
        steps = []
        hook = branch.policy.optimizer.register_step_post_hook(lambda *unused: steps.append(1))
        np.random.set_state(shuffle)
        try:
            branch.train()
        finally:
            hook.remove()
        exact = all(torch.equal(value, target.policy.state_dict()[name])
                    for name, value in branch.policy.state_dict().items())
        if rate == baseline_rate:
            if not exact or len(steps) != actual['optimizer_steps']:
                differences = {name: float((value - target.policy.state_dict()[name]).abs().max())
                               for name, value in branch.policy.state_dict().items()
                               if not torch.equal(value, target.policy.state_dict()[name])}
                raise ValueError(json.dumps({'error': 'Baseline PPO reproduction failed',
                    'steps': len(steps), 'expected_steps': actual['optimizer_steps'],
                    'maximum_parameter_differences': differences,
                    'metrics': {key: float(value) for key, value in branch.logger.name_to_value.items()}}))
        with torch.no_grad():
            new = branch.policy.get_distribution(observations).distribution
            kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(old_mean, old_std),
                torch.distributions.Normal(new.mean.double(), new.stddev.double())).sum(-1)
        branches.append({'learning_rate': rate, 'optimizer_steps': len(steps),
                         'baseline_parameters_exact': exact if rate == baseline_rate else None,
                         'exact_full_rollout_kl_mean': float(kl.mean()),
                         'exact_full_rollout_kl_max': float(kl.max()),
                         'metrics': {key: float(value) for key, value in branch.logger.name_to_value.items()}})
    report = {'scope': __doc__, 'batch': batch.name, 'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'),
              'script_sha256': sha(Path(__file__)), 'sampled_actions_exact': model.n_steps,
              'first_checkpoint_parameters_exact': True, 'initializer_model_sha256': metadata['model_sha256'],
              'learner_reward_units': protocol.get('PPO_reward_units', {'scale': 1.}),
              'initializer_rng_sha256': metadata['resume_state_sha256'], 'branches': branches,
              'limits': 'Fixed recorded rollout only. Alternative optimizer branches are discarded; no on-policy performance or native success claim.',
              'new_physical_controls': 0, 'new_deployed_updates': 0, 'wall_seconds': time.monotonic() - started}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.batch, args.output)

import copy
import json
import time

import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import scripts.analyze_native_rollout_pair as analyzer
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.independent_clip_ppo import load_ppo
from navstack.research import ResearchRecorder


@pytest.mark.parametrize('size', [32, 256])
def test_schedule_matches_real_ppo_collection_and_adam_steps(tmp_path, size):
    torch.set_num_threads(1)
    config = EnvConfig(method='B1', scenario='empty', flat_hold=10)
    raw = DummyVecEnv([lambda: NavigationEnv(config)])
    vec = VecNormalize(raw, norm_reward=False, gamma=config.gamma)
    model = PPO('MlpPolicy', vec, n_steps=32, batch_size=32, n_epochs=2, seed=11, device='cpu')
    model.learn(32)
    model.save(tmp_path / 'parent.zip')
    parent = analyzer.learner_data(tmp_path / 'parent.zip')
    model = load_ppo(tmp_path / 'parent.zip', env=vec, rollout_actions=size,
                     allow_rollout_intervention=True, device='cpu')
    (tmp_path / 'recorded').mkdir()
    callback = ResearchRecorder(tmp_path / 'recorded', time.time()+60)
    optimizer_steps = []
    hook = model.policy.optimizer.register_step_post_hook(lambda *args: optimizer_steps.append(1))
    try:
        model.learn(size, callback=callback, reset_num_timesteps=False)
        model.save(tmp_path / 'endpoint.zip')
        summary = {**callback.counts, 'optimization_phases': 1, 'optimizer_steps': len(optimizer_steps),
            'rollout_actions': size, 'batch_size': 32, 'ppo_epochs': 2}
        row = {**summary, 'metrics': {'train/n_updates': model._n_updates}}
        saved = analyzer.learner_data(tmp_path / 'endpoint.zip')
        result = analyzer.check_schedule(parent, saved, [row], summary, size)
        assert result['decisions'] == size
        assert result['optimizer_steps_per_phase'] == result['optimizer_steps'] == 2*size//32
        assert summary['physics_steps'] > 0
        wrong = copy.deepcopy(row); wrong['optimizer_steps'] -= 1
        with pytest.raises(ValueError, match='actual|Actual'):
            analyzer.check_schedule(parent, saved, [wrong], summary, size)
        changed = {**saved, 'learning_rate': .001}
        with pytest.raises(ValueError, match='other than rollout'):
            analyzer.check_schedule(parent, changed, [row], summary, size)
    finally:
        hook.remove(); callback.close(); vec.close()


@pytest.fixture
def pair(tmp_path, monkeypatch):
    """Pairing semantics fixture; real collection/Adam is tested above."""
    names = ['H0_rollout32', 'H0_rollout256']
    protocol = {'rollout_length_experiment': True, 'methods': names,
        'jobs': [{'name': name, 'method': 'H0', 'gradient_clip_mode': 'global', 'rollout_actions': size}
                 for name, size in zip(names, (32, 256))],
        'method_budgets': dict.fromkeys(names, 102400), 'recovery': {}}
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / 'config.json').write_text('{}')
        checkpoint = tmp_path / 'resume' / name
        checkpoint.mkdir(parents=True)
        (checkpoint / 'model.zip').write_bytes(b'identical fixture checkpoint; never deserialized')
        protocol['recovery'][name] = {'checkpoint': 'resume/'+name, 'lineage_controls': 120320,
            'consumed_controls_lower_bound': 129865, 'retained_optimizer_steps': 752,
            'unknown_unflushed_parent_tail': True}
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    def endpoint(batch, name, protocol, size):
        cases = [{'case_id': 0, 'success': False, 'termination_reason': 'out_of_bounds',
                  'progress': 3. if size == 32 else 1., 'intervention_rate': 0.}]
        return {'cases': cases, 'new_training_controls': 102400 if size == 32 else 102550,
                'evaluation_summary': {'successes': 0}}
    monkeypatch.setattr(analyzer, 'checked_endpoint', endpoint)
    monkeypatch.setattr(analyzer, 'initial_cases', lambda folder: {0: {'road_sha256': 'same-road',
        'goal': [8, 6, 0], 'static': [], 'dynamic': [], 'initial': {'x': 6, 'y': 6, 'yaw': 0, 'v': 0}}})
    return tmp_path


def test_failed_cases_and_actual_overshoot_are_retained(pair):
    result = analyzer.compare(pair, 'H0')
    assert result['success_difference_count'] == 0
    assert result['mean_progress_difference_m'] == -2
    assert result['actual_training_control_difference'] == 150
    assert result['paired_cases'][0]['rollout256_termination'] == 'out_of_bounds'


def test_different_parent_cannot_be_called_rollout_only_intervention(pair):
    (pair / 'resume/H0_rollout256/model.zip').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identical saved learning state'):
        analyzer.compare(pair, 'H0')


def test_different_roads_cannot_be_paired_by_case_number(pair, monkeypatch):
    original = analyzer.initial_cases
    def changed(folder):
        cases = original(folder)
        if folder.name.endswith('256'):
            cases[0]['road_sha256'] = 'other-road'
        return cases
    monkeypatch.setattr(analyzer, 'initial_cases', changed)
    with pytest.raises(ValueError, match='road/task identity'):
        analyzer.compare(pair, 'H0')


def test_combined_clipping_change_is_not_a_rollout_only_pair(pair):
    path = pair / 'protocol.json'
    protocol = analyzer.read(path)
    protocol['gradient_clipping_experiment'] = True
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='single-factor'):
        analyzer.compare(pair, 'H0')

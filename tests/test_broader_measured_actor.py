from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.fit_broader_measured_actor import fit_actor, validated_arrays, sha


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Linear(3, 4)
        self.action_net = torch.nn.Linear(4, 2)
        self.value_net = torch.nn.Linear(3, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(2))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def set_training_mode(self, mode):
        self.train(mode)

    def get_distribution(self, x):
        return SimpleNamespace(distribution=SimpleNamespace(mean=self.action_net(torch.tanh(self.policy_net(x)))))


@pytest.mark.parametrize('steering_scale',[.02,.01])
def test_actor_teaching_preserves_live_PPO_state_and_ignores_holdout_labels(steering_scale):
    torch.manual_seed(4)
    policy = Policy()
    # Populate actual retained Adam moments, including critic and exploration.
    sum(p.square().sum() for p in policy.parameters()).backward()
    policy.optimizer.step()
    policy.optimizer.zero_grad(set_to_none=True)
    other = deepcopy(policy)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    before_adam = deepcopy(policy.optimizer.state_dict())
    models = [SimpleNamespace(policy=p, device='cpu', num_timesteps=51200, _n_updates=240) for p in (policy, other)]
    x = np.random.default_rng(3).normal(size=(24, 3)).astype(np.float32)
    y = np.zeros((24, 2), np.float32)
    groups = np.repeat([0, 1, 2], 8)
    train = np.arange(24) % 5 != 0
    holdout = ~train
    altered = y.copy()
    altered[holdout] = .8
    results = [fit_actor(m, x, targets, groups, train, holdout, seed=11, steps=8, steering_scale=steering_scale)
               for m, targets in zip(models, (y, altered))]
    from scripts.fit_flat_feedback_contrast import exact
    assert exact(before_adam, policy.optimizer.state_dict())
    assert all(torch.equal(policy.state_dict()[k], other.state_dict()[k]) for k in before)
    assert any(not torch.equal(before[k], policy.state_dict()[k]) for k in before if 'policy_net' in k)
    assert all(torch.equal(before[k], policy.state_dict()[k]) for k in before if 'value_net' in k or k == 'log_std')
    assert results[0][0]['actual_Adam_step_each'] == [8]
    assert results[0][0]['case_metrics'] != results[1][0]['case_metrics']
    assert (models[0].num_timesteps, models[0]._n_updates) == (51200, 240)
    assert policy.training and all(p.requires_grad for p in policy.parameters())


def dataset(tmp_path):
    groups = np.repeat(np.arange(36), 6)
    path = tmp_path / 'data.npz'
    arrays = dict(observations=np.zeros((216, 4705), np.float32), actions=np.zeros((216, 2), np.float32),
                  groups=groups, train_mask=np.tile(np.arange(6) % 5 != 0, 36))
    np.savez_compressed(path, **arrays)
    manifest = dict(dataset_sha256=sha(path), teacher_unique_controls=216,
        cases=[dict(group=i, source_id=0, case_id=i, controls=6, termination_reason='success') for i in range(36)])
    families = ['curve', 'static_vehicle', 'crossing_walker'] * 12
    return path, arrays, manifest, families


def test_all_measured_cases_retained_failed_episodes_excluded_from_gradients(tmp_path):
    path, _, manifest, families = dataset(tmp_path)
    manifest['cases'][1]['termination_reason'] = 'collision'
    x, _, groups, train, holdout, success = validated_arrays(path, manifest, families)
    assert len(x) == 216 and 1 not in success
    assert not train[groups == 1].any() and not holdout[groups == 1].any()
    assert len(success) == 35 and not (train & holdout).any()


@pytest.mark.parametrize('fault', ['bytes', 'mask', 'groups', 'action', 'nonfinite', 'coverage'])
def test_invalid_or_incomplete_teacher_views_rejected(tmp_path, fault):
    path, arrays, manifest, families = dataset(tmp_path)
    if fault == 'bytes':
        path.write_bytes(b'changed')
    elif fault == 'coverage':
        for i, kind in enumerate(families):
            if kind == 'static_vehicle':
                manifest['cases'][i]['termination_reason'] = 'collision'
    else:
        if fault == 'mask': arrays['train_mask'][0] = True
        if fault == 'groups': arrays['groups'][0] = 35
        if fault == 'action': arrays['actions'][0, 0] = 1.1
        if fault == 'nonfinite': arrays['observations'][0, 0] = np.nan
        np.savez_compressed(path, **arrays)
        manifest['dataset_sha256'] = sha(path)
    with pytest.raises(ValueError):
        validated_arrays(path, manifest, families)


def test_actor_teaching_restores_flags_after_guard_failure():
    policy = Policy()
    model = SimpleNamespace(policy=policy, device='cpu', num_timesteps=51200, _n_updates=240)
    def stop(_):
        raise RuntimeError('finite resource guard')
    with pytest.raises(RuntimeError, match='resource guard'):
        fit_actor(model, np.zeros((6, 3), np.float32), np.zeros((6, 2), np.float32), np.zeros(6),
                  np.array([False, True, True, True, True, False]), np.array([True, False, False, False, False, True]),
                  seed=11, steps=8, callback=stop)
    assert policy.training and all(p.requires_grad for p in policy.parameters())


def test_teaching_observer_cannot_reprepare_existing_execution(tmp_path):
    from scripts.teach_broader_b1_cohort import prepare
    with pytest.raises(FileExistsError, match='existing'):
        prepare(tmp_path)


def test_teaching_observer_cannot_relaunch_started_execution(tmp_path):
    from scripts.teach_broader_b1_cohort import run
    (tmp_path / 'started.json').write_text('{}')
    with pytest.raises(FileExistsError, match='repeat'):
        run(tmp_path)


def test_actor_fit_does_not_import_a_different_experiment_script(monkeypatch):
    import builtins
    original_import = builtins.__import__
    def isolated_import(name, *args, **kwargs):
        if name == 'scripts.fit_flat_feedback_contrast':
            raise ModuleNotFoundError('Unavailable in the frozen policy source')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', isolated_import)
    model = SimpleNamespace(policy=Policy(), device='cpu', num_timesteps=51200, _n_updates=240)
    result, _, _ = fit_actor(model, np.zeros((6, 3), np.float32), np.zeros((6, 2), np.float32),
        np.zeros(6), np.array([False, True, True, True, True, False]),
        np.array([True, False, False, False, False, True]), seed=11, steps=2)
    assert result['actual_Adam_step_each'] == [2]


def test_completed_teacher_adoption_rejects_unfinished_data_without_process_adoption(tmp_path, monkeypatch):
    import json
    import scripts.teach_broader_b1_cohort as module
    teacher = tmp_path / 'teacher'
    teacher.mkdir()
    (teacher / 'status.json').write_text(json.dumps(dict(phase='BUILDING_EXECUTED_TRAIN_DIRECT_VIEWS')))
    monkeypatch.setattr(module, 'TEACHER_EXECUTION', teacher)
    monkeypatch.setattr(module, 'OwnedProcess', lambda *args, **kwargs: pytest.fail('Must not adopt a PID in completed-data mode'))
    output = tmp_path / 'unstarted'
    with pytest.raises(ValueError, match='completed audited'):
        module.prepare(output, completed_teacher=True)
    assert not output.exists()

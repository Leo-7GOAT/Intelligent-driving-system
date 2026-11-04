import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from navstack.config import EnvConfig, PlannerConfig
from navstack.research import cumulative_train


def test_frozen_stats_ppo_updates_and_exact_resume(tmp_path):
    config = EnvConfig(method='B1', scenario='empty', deadline=2,
                       planner=PlannerConfig(max_seconds=None))
    options = {'use_sde': True, 'log_std_init': -4, 'sde_sample_freq': 1}
    seed = cumulative_train(tmp_path/'seed', config, 11, 64, time.time()+90,
                            rollout_physics=64, epochs=2, ppo_options=options)
    initial = Path(seed['checkpoint'])
    frozen = {**options, 'freeze_observation_stats': True}
    for name in ('whole', 'split', 'control'):
        run = tmp_path/name
        shutil.copytree(initial, run/'initial')
        (run/'latest.json').write_text(json.dumps({**seed, 'checkpoint': str(run/'initial')}))
        (run/'ppo_options.json').write_text(json.dumps(options if name=='control' else frozen))
    def train(name, target):
        return cumulative_train(tmp_path/name, config, 11, target, time.time()+90,
                                rollout_physics=64, epochs=2,
                                ppo_options=options if name=='control' else frozen)
    whole = train('whole', 256)
    middle = train('split', 128)
    split = train('split', 256)
    control = train('control', 256)
    def normalization(checkpoint):
        with (Path(checkpoint)/'normalization.pkl').open('rb') as stream:
            return pickle.load(stream)
    before = normalization(initial)
    for result in (whole, middle, split):
        after = normalization(result['checkpoint'])
        assert not after.training and not after.norm_reward
        for field in ('mean', 'var', 'count'):
            assert np.array_equal(getattr(before.obs_rms, field), getattr(after.obs_rms, field))
        assert result['optimizer_steps'] > seed['optimizer_steps']
        assert result['reload_exact']
    assert normalization(control['checkpoint']).obs_rms.count > before.obs_rms.count
    models = [PPO.load(Path(r['checkpoint'])/'model.zip', device='cpu')
              for r in (seed, whole, split)]
    assert whole['physics_steps'] == split['physics_steps'] == 256
    assert whole['optimizer_steps'] == split['optimizer_steps']
    assert all(torch.equal(v, models[2].policy.state_dict()[k])
               for k,v in models[1].policy.state_dict().items())
    assert not torch.equal(models[0].policy.action_net.weight, models[1].policy.action_net.weight)


def test_freeze_rejects_unfitted_fresh_run(tmp_path):
    with pytest.raises(ValueError, match='inherited checkpoint'):
        cumulative_train(tmp_path, EnvConfig(method='B1'), 11, 64, time.time()+60,
                         ppo_options={'freeze_observation_stats': True})

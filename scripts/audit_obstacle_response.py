"""Frozen-policy obstacle response probes; counterfactual commands never execute."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checkpoint, source, output = (Path(p).resolve() for p in (args.checkpoint, args.source, args.output))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.experiments import load_run, jsonable
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    cfg = EnvConfig.from_dict(json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))['env'])
    if cfg.method != 'B1':
        raise ValueError('Continuous direct-control diagnostic only')
    cfg.split = 'validation'  # Never activate an inherited training curriculum.
    model, norm = load_run(checkpoint)
    stats = copy.deepcopy(norm.obs_rms)
    results = []
    with gzip.open(output/'trace.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for scenario in ('static', 'narrow', 'unseen'):
            cfg.scenario = scenario
            for seed in (7301, 7302):
                env = NavigationEnv(cfg)
                obs, _ = env.reset(seed=seed)
                initial = copy.deepcopy(env.initial_scenario)
                probes = []
                try:
                    while not env.done:
                        state_before = copy.deepcopy(env.state)
                        # Remove static obstacles ONLY to construct a sensitivity
                        # input at the same state. Restore before the actual step.
                        obstacles, timing = env.world.static, env.feedback_seconds
                        try:
                            env.world.static = []
                            empty_obs = env._observation()
                        finally:
                            env.world.static = obstacles
                            env.feedback_seconds = timing
                        assert env.state == state_before
                        with torch.no_grad():
                            real = model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(obs[None]))).distribution
                            mean, std = real.mean[0].numpy().copy(), real.stddev[0].numpy().copy()
                            absent = model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(empty_obs[None]))).distribution.mean[0].numpy().copy()
                        action, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
                        obs, _, _, _, info = env.step(action[0])
                        probe = {'time':state_before.time, 'raw_mean':mean, 'std':std,
                                 'without_obstacles_mean':absent, 'executed_action':action[0]}
                        probes.append(probe)
                        stream.write(json.dumps({'scenario':scenario,'seed':seed,**probe,'trace':info['trace']},default=jsonable)+'\n')
                    means = np.array([p['raw_mean'] for p in probes])
                    absent = np.array([p['without_obstacles_mean'] for p in probes])
                    results.append({'scenario':scenario,'seed':seed,'initial':initial,
                        'terminal_state':info['trace'][-1]['state'],**info['episode_metrics'],
                        'raw_mean_min':means.min(axis=0),'raw_mean_max':means.max(axis=0),
                        'action_std_mean':np.mean([p['std'] for p in probes],axis=0),
                        'clipped_mean_fraction':(np.abs(means)>=1).mean(axis=0),
                        'obstacle_removal_raw_delta_mean':np.abs(means-absent).mean(axis=0),
                        'obstacle_removal_clipped_delta_mean':np.abs(np.clip(means,-1,1)-np.clip(absent,-1,1)).mean(axis=0)})
                finally:
                    env.close()
    np.testing.assert_array_equal(stats.mean,norm.obs_rms.mean)
    np.testing.assert_array_equal(stats.var,norm.obs_rms.var)
    assert stats.count == norm.obs_rms.count
    norm.close()
    report = {'scope':'Six existing development cases; obstacle removal is unexecuted sensitivity, not an ablation or safety test',
        'model_sha256':hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest(),
        'source_hashes':{p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'navstack').glob('*.py')},
        'normalization_frozen':True,'episodes':results}
    (output/'results.json').write_text(json.dumps(report,indent=2,default=jsonable),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('scenario','seed','termination_reason','clipped_mean_fraction','action_std_mean','obstacle_removal_clipped_delta_mean')} for r in results],default=jsonable))


if __name__ == '__main__':
    main()

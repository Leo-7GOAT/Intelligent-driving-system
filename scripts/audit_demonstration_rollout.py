"""Measure closed-loop drift on declared demonstration train/holdout cases.

This executes the learned policy, never the teacher. No test cases or fitting
are introduced. Teacher observations are used only for an offline range audit.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    batch, output = Path(args.batch).resolve(), Path(args.output).resolve()
    source = batch/'source'
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.experiments import load_run, jsonable, write_json
    torch.set_num_threads(1)
    protocol = json.loads((batch/'protocol.json').read_text(encoding='utf-8'))
    for relative, digest in protocol['source_hashes'].items():
        assert hashlib.sha256((source/relative).read_bytes()).hexdigest() == digest
    data_path = batch/'data/demonstrations.npz'
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == protocol['data_sha256']
    cases = json.loads((batch/'data/protocol.json').read_text(encoding='utf-8'))['cases']
    data = np.load(data_path)
    mask = np.array([cases[int(i)]['data_split'] == 'train' for i in data['case_id']])
    lo, hi = data['observations'][mask].min(0), data['observations'][mask].max(0)
    checkpoint = batch/'B1/supervised_initial'
    model, norm = load_run(checkpoint)
    stats = copy.deepcopy(norm.obs_rms)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        with gzip.open(output/'trace.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for case_id, case in enumerate(cases):
                if case['seed'] not in (8101, 8107):
                    continue
                cfg = EnvConfig.from_dict(protocol['env'])
                cfg.scenario, cfg.split = case['scenario'], 'train'
                assert not cfg.route_stages and not cfg.curriculum
                env = NavigationEnv(cfg)
                obs, _ = env.reset(seed=case['seed'])
                out_of_range = np.zeros(len(obs), dtype=int)
                clipped = np.zeros(len(obs), dtype=int)
                count = 0
                first_departure = None
                try:
                    while not env.done:
                        normalized = norm.normalize_obs(obs[None])
                        out_of_range += ((obs < lo-1e-5) | (obs > hi+1e-5))
                        clipped += (np.abs(normalized[0]) >= 9.999)
                        if first_departure is None and abs(obs[18]) > .02:
                            first_departure = {'control': count, 'sin_heading_error': float(obs[18]),
                                               'lateral_goal': float(obs[17])}
                        action, _ = model.predict(normalized, deterministic=True)
                        obs, _, _, _, info = env.step(action[0])
                        stream.write(json.dumps({'case_id': case_id, 'action': action[0],
                                                 'trace': info['trace']}, default=jsonable)+'\n')
                        count += 1
                    rows.append({'case_id': case_id, **case, **info['episode_metrics'],
                        'first_heading_departure': first_departure,
                        'outside_training_range_fraction': (out_of_range/count).tolist(),
                        'normalized_clip_fraction': (clipped/count).tolist()})
                    write_json(output/'progress.json', rows)
                    print(json.dumps({k: rows[-1][k] for k in
                        ('scenario', 'seed', 'data_split', 'success', 'termination_reason')}), flush=True)
                finally:
                    env.close()
        np.testing.assert_array_equal(stats.mean, norm.obs_rms.mean)
        np.testing.assert_array_equal(stats.var, norm.obs_rms.var)
        assert stats.count == norm.obs_rms.count
        write_json(output/'results.json', {'scope': 'Eight demonstration-layout diagnostic replays; no fitting, no independent confirmation',
            'model_sha256': hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest(),
            'normalization_frozen': True, 'episodes': rows})
    finally:
        norm.close()


if __name__ == '__main__':
    main()

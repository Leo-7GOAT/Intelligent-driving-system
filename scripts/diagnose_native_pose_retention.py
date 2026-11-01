"""Compare fixed TRAIN teacher/pose-response errors before and after one actual PPO course."""
import argparse
import gzip
import json
import math
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha


def analyze(batch):
    if any(k.startswith('navstack.') for k in sys.modules):
        raise RuntimeError('Use a fresh interpreter with the exact frozen policy runtime')
    batch = Path(batch).resolve()
    sys.path.insert(0, str(batch / 'source'))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from navstack.vehicle import wrap
    from scripts.evaluate_native_rule import require_finished_parent
    require_finished_parent(batch)
    p = read(batch / 'protocol.json')
    before, after = batch / 'resume/B1', batch / 'B1'
    evaluation = read(after / 'evaluation.json')
    for name, digest in p['source_hashes'].items():
        if sha(batch / 'source' / name) != digest:
            raise ValueError('Frozen runtime changed')
    if sha(before / 'model.zip') != p['shared_parking_PPO']['source_model_sha256'] or sha(after / 'model.zip') != evaluation['model_sha256']:
        raise ValueError('Require the exact registered initial and final models')
    dataset = batch / 'fit_evidence/fit_inputs.npz'
    fit = read(batch / 'fit_evidence/complete.json')
    if sha(dataset) != fit['dataset_sha256']:
        raise ValueError('Fixed TRAIN teacher/query data changed')
    with np.load(dataset, allow_pickle=False) as arrays:
        obs, labels, groups = (arrays[k].copy() for k in ('observations', 'actions', 'groups'))
        q, steering, eps = (arrays[k].copy() for k in ('pose_observations', 'MPC_steerings', 'epsilons'))
    limit = read(after / 'config.json')['env']['vehicle']['max_steer']
    target = (steering[:, 1:] - steering[:, :1]) / limit
    torch.set_num_threads(1)
    arms = []
    statistics = None
    for label, folder in (('before_PPO', before), ('after_PPO', after)):
        model = load_ppo(folder / 'model.zip', device='cpu')
        model.policy.set_training_mode(False)
        with (folder / 'normalization.pkl').open('rb') as stream:
            norm = pickle.load(stream)
        current = (norm.obs_rms.mean, norm.obs_rms.var, norm.obs_rms.count)
        if statistics is None:
            statistics = tuple(np.copy(v) for v in current)
        for a, b in zip(current, statistics):
            np.testing.assert_array_equal(a, b)
        with torch.no_grad():
            x = torch.as_tensor(norm.normalize_obs(obs))
            prediction = torch.cat([model.policy.get_distribution(x[i:i + 256]).distribution.mean for i in range(0, len(x), 256)]).numpy()
            queries = torch.as_tensor(norm.normalize_obs(q.reshape(-1, obs.shape[1])))
            response = model.policy.get_distribution(queries).distribution.mean.numpy().reshape(-1, 3, 2)
        metrics = []
        for group in range(4):
            error = np.sqrt(np.square(prediction[groups == group] - labels[groups == group]).mean(0))
            metrics.append({'case_id': group, 'teacher_acceleration_RMS': float(error[0]), 'teacher_steering_RMS_rad': float(error[1] * limit)})
        responses = []
        for epsilon in np.unique(eps):
            error = (response[:, 1:, 1] - response[:, :1, 1] - target)[eps == epsilon]
            responses.append({'axis': 'lateral_metres' if epsilon > .05 else 'yaw_radians',
                              'normalized_response_RMS_error': float(np.sqrt(np.square(error).mean()))})
        arms.append({'phase': label, 'model_sha256': sha(folder / 'model.zip'), 'teacher_metrics': metrics, 'pose_responses': responses})
    traces = {}
    with gzip.open(after / 'evaluation_trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            traces.setdefault(row['case_id'], []).append(row)
    failed = []
    for case, rows in traces.items():
        metric = next(e for e in evaluation['episodes'] if e['case_id'] == case)
        if metric['success']:
            continue
        goal = rows[0]['info']['initial_scenario']['goal']
        ticks = [t for r in rows for t in r['info']['trace']]
        closest = min(ticks, key=lambda t: math.hypot(t['state']['x'] - goal[0], t['state']['y'] - goal[1]))
        state = closest['state']
        failed.append({'case_id': case, 'termination': metric['termination_reason'], 'controls': len(ticks),
            'closest_goal_distance_m': math.hypot(state['x'] - goal[0], state['y'] - goal[1]),
            'heading_error_at_closest_rad': abs(wrap(state['yaw'] - goal[2])), 'speed_at_closest': state['v'],
            'goal_parking_overrides': metric['goal_parking_controls']})
    return {'scope': __doc__, 'script_sha256': sha(Path(__file__)), 'batch': batch.name, 'seed': p['seed'],
            'dataset_sha256': sha(dataset), 'model_and_observation_statistics_verified': True,
            'arms': arms, 'failed_native_cases': failed, 'new_controls': 0, 'new_optimizer_updates': 0,
            'limits': 'Fixed TRAIN-only measured teacher rows and virtual pose perturbations. Error changes diagnose retention;they do not alone establish the cause of physical failures or new driving capability.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.batch)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)

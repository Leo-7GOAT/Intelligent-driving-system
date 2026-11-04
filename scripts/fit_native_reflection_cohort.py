"""Apply one prospectively fixed TRAIN-only actor-fit recipe to all five independent B1 parents."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha
from scripts.fit_flat_feedback_contrast import exact
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def fit_one(plan_path, seed, output):
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo
    from navstack.research import atomic_json, resources_ok
    from scripts.evaluate_native_rule import require_finished_parent
    plan = read(plan_path)
    if seed not in (11, 22, 33, 44, 55) or [r['seed'] for r in plan['parents']] != [11, 22, 33, 44, 55]:
        raise ValueError('Require every registered independent parent')
    if plan['auxiliary_steps_each'] != 1024 or plan['auxiliary_learning_rate'] != 1e-4:
        raise ValueError('The common fit budget and rate are fixed before any fit')
    if sha(Path(__file__)) != plan['fit_script_sha256']:
        raise ValueError('Registered fit implementation changed')
    row = next(r for r in plan['parents'] if r['seed'] == seed)
    parent = ROOT / row['batch']
    require_finished_parent(parent)
    origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    summary = read(parent / 'B1/summary.json')
    if (metadata['model_sha256'] != row['model_sha256'] or summary['lineage_control_steps'] != 20480
            or sha(parent / 'B1/endpoint_raw_audit.json') != row['raw_audit_sha256']):
        raise ValueError('Independent actual-trained parent differs')
    dataset = ROOT / plan['dataset']
    if sha(dataset) != plan['dataset_sha256']:
        raise ValueError('Common TRAIN-only teacher/reflection/response arrays changed')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Never overwrite or automatically repeat a cohort fit')
    okay, reason = resources_ok(ROOT)
    if not okay:
        raise RuntimeError(str(reason))
    cfg = EnvConfig.from_dict(read(origin / 'config.json')['env'])
    torch.set_num_threads(1)
    model = load_ppo(origin / 'model.zip', device='cpu')
    model.policy.set_training_mode(False)
    with (origin / 'normalization.pkl').open('rb') as stream:
        norm = pickle.load(stream)
    stats = (norm.obs_rms.mean.copy(), norm.obs_rms.var.copy(), float(norm.obs_rms.count))
    with np.load(dataset, allow_pickle=False) as arrays:
        inputs, targets, groups = (arrays[k].copy() for k in ('observations', 'actions', 'groups'))
        queries, steering, eps = (arrays[k].copy() for k in ('pose_observations', 'MPC_steerings', 'epsilons'))
    if inputs.shape != (6848, 4705) or queries.shape != (120, 3, 4705):
        raise ValueError('Require the exact registered acknowledged/virtual TRAIN inventory')
    x, y = torch.as_tensor(norm.normalize_obs(inputs)), torch.as_tensor(targets)
    q = torch.as_tensor(norm.normalize_obs(queries.reshape(-1, 4705))).reshape(-1, 3, 4705)
    target = torch.as_tensor((steering[:, 1:] - steering[:, :1]) / cfg.vehicle.max_steer)
    scales = torch.empty(len(eps), 1)
    for epsilon in np.unique(eps):
        scales[eps == epsilon] = float(np.sqrt(np.square(target.numpy()[eps == epsilon]).mean()))
    parameters = [v for k, v in model.policy.named_parameters() if 'policy_net' in k or 'action_net' in k]
    fixed = {k: v.detach().clone() for k, v in model.policy.state_dict().items() if 'value_net' in k or k == 'log_std'}
    old_adam = deepcopy(model.policy.optimizer.state_dict())
    counters = (model.num_timesteps, model._n_updates)
    actor_before = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in parameters)).hexdigest()
    for value in model.policy.parameters():
        value.requires_grad_(any(value is fitted for fitted in parameters))
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / 'executed_fit.py')
    atomic_json(output / 'registration.json', {'plan_sha256': sha(Path(plan_path)), 'seed': seed,
        'source_model_sha256': metadata['model_sha256'], 'dataset_sha256': plan['dataset_sha256'],
        'actor_before_sha256': actor_before, 'new_controls': 0, 'new_PPO_updates': 0,
        'auxiliary_steps': 1024, 'learning_rate': 1e-4})
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    rng = np.random.default_rng(seed)
    weights = np.zeros(len(groups))
    for group in np.unique(groups):
        weights[groups == group] = 1 / (len(np.unique(groups)) * np.count_nonzero(groups == group))
    history = []
    started = time.monotonic()
    predict = lambda z: model.policy.get_distribution(z).distribution.mean
    for step in range(1, 1025):
        if time.monotonic() - started > 600:
            raise TimeoutError('Finite single-seed fit CPU budget exhausted; no automatic retry')
        indices = rng.choice(len(x), 256, replace=True, p=weights)
        means = predict(x[indices])
        response = predict(q.flatten(0, 1)).reshape(-1, 3, 2)
        point = ((means - y[indices]) / torch.tensor([.1, .02])).square().mean()
        derivative = ((response[:, 1:, 1] - response[:, :1, 1] - target) / scales).square().mean()
        optimizer.zero_grad(set_to_none=True)
        (point + derivative).backward()
        torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 128 == 0:
            okay, reason = resources_ok(ROOT)
            if not okay:
                raise RuntimeError(str(reason))
            record = {'seed': seed, 'auxiliary_steps': step, 'teacher_loss': float(point.detach()), 'response_loss': float(derivative.detach())}
            history.append(record)
            atomic_json(output / 'heartbeat.json', record)
            print(json.dumps(record), flush=True)
    for value in model.policy.parameters():
        value.requires_grad_(True)
    if (not exact(old_adam, model.policy.optimizer.state_dict()) or counters != (model.num_timesteps, model._n_updates)
            or any(not torch.equal(v, model.policy.state_dict()[k]) for k, v in fixed.items())):
        raise ValueError('Critic/log_std/retained PPO state changed during auxiliary fitting')
    np.testing.assert_array_equal(stats[0], norm.obs_rms.mean)
    np.testing.assert_array_equal(stats[1], norm.obs_rms.var)
    if stats[2] != norm.obs_rms.count:
        raise ValueError('Observation count changed')
    steps = sorted({int(v['step']) for v in optimizer.state_dict()['state'].values()})
    if steps != [1024]:
        raise ValueError('Actual auxiliary Adam state differs from the registered budget')
    with torch.no_grad():
        prediction = torch.cat([predict(x[i:i + 256]) for i in range(0, len(x), 256)]).numpy()
        response = predict(q.flatten(0, 1)).reshape(-1, 3, 2)
    metrics = []
    for group in sorted(np.unique(groups)):
        error = np.sqrt(np.square(prediction[groups == group] - targets[groups == group]).mean(0))
        metrics.append({'case_id': int(group % 12), 'reflected': bool(group >= 12),
                        'acceleration_RMS': float(error[0]), 'steering_RMS_rad': float(error[1] * cfg.vehicle.max_steer)})
    responses = []
    for epsilon in np.unique(eps):
        axis = 'lateral_metres' if epsilon > .05 else 'yaw_radians'
        error = (response[:, 1:, 1] - response[:, :1, 1] - target).numpy()[eps == epsilon]
        responses.append({'axis': axis, 'normalized_steering_response_RMS': float(np.sqrt(np.square(error).mean())),
                          'ceiling': plan['response_ceilings'][axis]})
    basic = [m for m in metrics if m['case_id'] < 4]
    passed = all(m['steering_RMS_rad'] <= (.02 if m['reflected'] else .01) and (m['reflected'] or m['acceleration_RMS'] <= .1) for m in basic)
    passed = passed and all(m['normalized_steering_response_RMS'] <= m['ceiling'] for m in responses)
    checkpoint = output / 'checkpoint'
    shutil.copytree(origin, checkpoint)
    model.save(checkpoint / 'model.zip')
    torch.save(optimizer.state_dict(), output / 'auxiliary_Adam.pt')
    atomic_json(output / 'sampler_state.json', rng.bit_generator.state)
    metadata.update(model_sha256=sha(checkpoint / 'model.zip'), cohort_reflection_actor_fit_steps=1024,
        cohort_reflection_plan_sha256=sha(Path(plan_path)),
        note='Independent seed parent with20480actual PPOcontrols plus1024separate TRAIN-only actor Adam steps. Original critic,log_std,PPOAdam,counters,RMS and RNG file bytes retained. Not new PPOlearning or native qualification.')
    atomic_json(checkpoint / 'metadata.json', metadata)
    verified_checkpoint_metadata(checkpoint)
    actor_after = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in parameters)).hexdigest()
    result = {'scope': __doc__, 'seed': seed, 'parent_batch': parent.name, 'plan_sha256': sha(Path(plan_path)),
        'model_sha256': metadata['model_sha256'], 'source_model_sha256': row['model_sha256'],
        'normalization_sha256': metadata['normalization_sha256'], 'dataset_sha256': plan['dataset_sha256'],
        'new_auxiliary_Adam_steps': 1024, 'auxiliary_Adam_parameter_states': len(optimizer.state),
        'actual_auxiliary_Adam_step_each': steps, 'auxiliary_Adam_sha256': sha(output / 'auxiliary_Adam.pt'),
        'PPO_critic_logstd_optimizer_counters_RMS_retained': True,
        'actor_before_sha256': actor_before, 'actor_after_sha256': actor_after,
        'teacher_metrics': metrics, 'pose_response': responses, 'offline_screen_passed': bool(passed),
        'history': history, 'elapsed_seconds': time.monotonic() - started,
        'new_controls': 0, 'new_PPO_updates': 0, 'research_acceptance': 'UNMET'}
    atomic_json(output / 'complete.json', result)
    print(json.dumps({'seed': seed, 'offline_screen_passed': bool(passed), 'model_sha256': metadata['model_sha256']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fit_one(args.plan, args.seed, args.output)

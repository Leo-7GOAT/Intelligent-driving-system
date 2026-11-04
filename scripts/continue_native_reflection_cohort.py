"""One fixed512-step auxiliary continuation for every failed1024-step TRAIN-only fit;no driving."""
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


def run(plan_path, output):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from navstack.research import atomic_json, resources_ok
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = read(plan_path)
    if output.exists() or plan['new_auxiliary_steps_each'] != 512 or sha(Path(__file__)) != plan['script_sha256']:
        raise ValueError('Require a fresh output and the exact fixed continuation implementation')
    base_plan = read(ROOT / plan['base_plan'])
    if sha(ROOT / plan['base_plan']) != plan['base_plan_sha256'] or [r['seed'] for r in base_plan['parents']] != [11, 22, 33, 44, 55]:
        raise ValueError('All five original registered parents are required')
    dataset = ROOT / base_plan['dataset']
    if sha(dataset) != base_plan['dataset_sha256']:
        raise ValueError('Common TRAIN-only arrays changed')
    output.mkdir(parents=True)
    shutil.copy2(plan_path, output / 'registered_continuation.json')
    torch.set_num_threads(1)
    with np.load(dataset, allow_pickle=False) as data:
        obs, labels, groups = (data[k].copy() for k in ('observations', 'actions', 'groups'))
        queries, steering, eps = (data[k].copy() for k in ('pose_observations', 'MPC_steerings', 'epsilons'))
    completed = []
    for row in base_plan['parents']:
        seed = row['seed']
        origin = ROOT / f'runs/B1_shared_parking_cohort_fit_seed{seed}_20261101'
        original = read(origin / 'complete.json')
        if original['offline_screen_passed'] or original['new_auxiliary_Adam_steps'] != 1024 or original['actual_auxiliary_Adam_step_each'] != [1024]:
            raise ValueError('This common extension requires the preserved failed1024-step endpoints')
        metadata = verified_checkpoint_metadata(origin / 'checkpoint')
        model = load_ppo(origin / 'checkpoint/model.zip', device='cpu')
        model.policy.set_training_mode(False)
        with (origin / 'checkpoint/normalization.pkl').open('rb') as stream:
            norm = pickle.load(stream)
        moments = (norm.obs_rms.mean.copy(), norm.obs_rms.var.copy(), float(norm.obs_rms.count))
        x, y = torch.as_tensor(norm.normalize_obs(obs)), torch.as_tensor(labels)
        q = torch.as_tensor(norm.normalize_obs(queries.reshape(-1, 4705))).reshape(-1, 3, 4705)
        limit = read(origin / 'checkpoint/config.json')['env']['vehicle']['max_steer']
        target = torch.as_tensor((steering[:, 1:] - steering[:, :1]) / limit)
        scale = torch.empty(len(eps), 1)
        for epsilon in np.unique(eps):
            scale[eps == epsilon] = float(np.sqrt(np.square(target.numpy()[eps == epsilon]).mean()))
        fitted = [v for k, v in model.policy.named_parameters() if 'policy_net' in k or 'action_net' in k]
        fixed = {k: v.clone() for k, v in model.policy.state_dict().items() if 'value_net' in k or k == 'log_std'}
        old_adam = deepcopy(model.policy.optimizer.state_dict())
        counters = (model.num_timesteps, model._n_updates)
        for value in model.policy.parameters():
            value.requires_grad_(any(value is p for p in fitted))
        optimizer = torch.optim.Adam(fitted, lr=1e-4)
        if sha(origin / 'auxiliary_Adam.pt') != original['auxiliary_Adam_sha256']:
            raise ValueError('Original auxiliary optimizer changed')
        optimizer.load_state_dict(torch.load(origin / 'auxiliary_Adam.pt', map_location='cpu', weights_only=True))
        rng = np.random.default_rng()
        rng.bit_generator.state = read(origin / 'sampler_state.json')
        weights = np.zeros(len(groups))
        for group in np.unique(groups):
            weights[groups == group] = 1 / (len(np.unique(groups)) * np.count_nonzero(groups == group))
        folder = output / ('seed' + str(seed))
        folder.mkdir()
        shutil.copy2(__file__, folder / 'executed_continuation.py')
        history = []
        started = time.monotonic()
        predict = lambda z: model.policy.get_distribution(z).distribution.mean
        for step in range(1, 513):
            if time.monotonic() - started > 600:
                raise TimeoutError('Finite continuation CPU budget exhausted;no automatic retry')
            indices = rng.choice(len(x), 256, replace=True, p=weights)
            prediction = predict(x[indices])
            response = predict(q.flatten(0, 1)).reshape(-1, 3, 2)
            point = ((prediction - y[indices]) / torch.tensor([.1, .02])).square().mean()
            derivative = ((response[:, 1:, 1] - response[:, :1, 1] - target) / scale).square().mean()
            optimizer.zero_grad(set_to_none=True)
            (point + derivative).backward()
            torch.nn.utils.clip_grad_norm_(fitted, .5, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 128 == 0:
                okay, reason = resources_ok(ROOT)
                if not okay:
                    raise RuntimeError(str(reason))
                record = {'seed': seed, 'additional_auxiliary_steps': step, 'teacher_loss': float(point.detach()), 'response_loss': float(derivative.detach())}
                history.append(record)
                atomic_json(folder / 'heartbeat.json', record)
                print(json.dumps(record), flush=True)
        for value in model.policy.parameters():
            value.requires_grad_(True)
        if (not exact(old_adam, model.policy.optimizer.state_dict()) or counters != (model.num_timesteps, model._n_updates)
                or any(not torch.equal(v, model.policy.state_dict()[k]) for k, v in fixed.items())):
            raise ValueError('Frozen PPO learning state changed')
        np.testing.assert_array_equal(moments[0], norm.obs_rms.mean)
        np.testing.assert_array_equal(moments[1], norm.obs_rms.var)
        if moments[2] != norm.obs_rms.count or {int(s['step']) for s in optimizer.state_dict()['state'].values()} != {1536}:
            raise ValueError('RMS or exact continued Adam step counts differ')
        with torch.no_grad():
            prediction = torch.cat([predict(x[i:i + 256]) for i in range(0, len(x), 256)]).numpy()
            response = predict(q.flatten(0, 1)).reshape(-1, 3, 2).numpy()
        metrics = []
        for group in sorted(np.unique(groups)):
            error = np.sqrt(np.square(prediction[groups == group] - labels[groups == group]).mean(0))
            metrics.append({'case_id': int(group % 12), 'reflected': bool(group >= 12), 'acceleration_RMS': float(error[0]), 'steering_RMS_rad': float(error[1] * limit)})
        responses = []
        for epsilon in np.unique(eps):
            axis = 'lateral_metres' if epsilon > .05 else 'yaw_radians'
            error = (response[:, 1:, 1] - response[:, :1, 1] - target.numpy())[eps == epsilon]
            responses.append({'axis': axis, 'normalized_steering_response_RMS': float(np.sqrt(np.square(error).mean())), 'ceiling': base_plan['response_ceilings'][axis]})
        basic = [m for m in metrics if m['case_id'] < 4]
        passed = all(m['steering_RMS_rad'] <= (.02 if m['reflected'] else .01) and (m['reflected'] or m['acceleration_RMS'] <= .1) for m in basic) and all(m['normalized_steering_response_RMS'] <= m['ceiling'] for m in responses)
        checkpoint = folder / 'checkpoint'
        shutil.copytree(origin / 'checkpoint', checkpoint)
        model.save(checkpoint / 'model.zip')
        torch.save(optimizer.state_dict(), folder / 'auxiliary_Adam.pt')
        atomic_json(folder / 'sampler_state.json', rng.bit_generator.state)
        metadata.update(model_sha256=sha(checkpoint / 'model.zip'), cohort_reflection_actor_fit_steps=1536,
                        continuation_plan_sha256=sha(plan_path))
        atomic_json(checkpoint / 'metadata.json', metadata)
        verified_checkpoint_metadata(checkpoint)
        result = {'seed': seed, 'source_model_sha256': original['model_sha256'], 'model_sha256': metadata['model_sha256'],
            'new_auxiliary_Adam_steps': 512, 'actual_auxiliary_Adam_step_each': [1536],
            'auxiliary_optimizer_and_sampler_resumed': True, 'PPO_critic_logstd_optimizer_counters_RMS_retained': True,
            'teacher_metrics': metrics, 'pose_response': responses, 'offline_screen_passed': bool(passed),
            'history': history, 'elapsed_seconds': time.monotonic() - started, 'new_controls': 0, 'new_PPO_updates': 0}
        atomic_json(folder / 'complete.json', result)
        completed.append(result)
        atomic_json(output / 'status.json', {'phase': 'CPU_CONTINUATION_RUNNING', 'completed': completed})
    atomic_json(output / 'status.json', {'phase': 'ALL_FIVE_CPU_CONTINUATIONS_COMPLETE', 'completed': completed,
        'new_auxiliary_Adam_steps': 2560, 'offline_screens_passed': sum(r['offline_screen_passed'] for r in completed),
        'new_controls': 0, 'new_PPO_updates': 0, 'limits': 'Isolated CPU candidates only;the actual native queue and all its inputs remain unchanged. No native qualification or additional continuation is automatic.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.plan, args.output)

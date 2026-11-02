"""Bounded actor-only teaching from audited, measured broader TRAIN views.

This changes actor weights, not PPO counters, critic, exploration, observation
statistics or physical interaction. A subsequent native rollout is required.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_state(left, right):
    """Compare retained Adam state without importing another experiment script."""
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_state(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact_state(a, b) for a, b in zip(left, right))
    return left == right


def validated_arrays(dataset, manifest, families):
    """Keep every recorded case; use successful TRAIN episodes as labels."""
    import numpy as np
    if (len(families) != 36 or set(families) != {'curve', 'static_vehicle', 'crossing_walker'}
            or any(families.count(kind) != 12 for kind in set(families))):
        raise ValueError('Require the registered twelve cases per TRAIN family')
    if sha(dataset) != manifest['dataset_sha256']:
        raise ValueError('Measured dataset bytes changed')
    with np.load(dataset, allow_pickle=False) as loaded:
        arrays = {k: loaded[k].copy() for k in ('observations', 'actions', 'groups', 'train_mask')}
    x, y, groups, mask = (arrays[k] for k in ('observations', 'actions', 'groups', 'train_mask'))
    count = manifest['teacher_unique_controls']
    if (x.shape != (count, 4705) or y.shape != (count, 2) or groups.shape != (count,)
            or mask.shape != (count,) or mask.dtype != np.bool_
            or not np.issubdtype(groups.dtype, np.integer)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or (np.abs(y) > 1).any()
            or len(manifest['cases']) != 36):
        raise ValueError('Require all36 measured TRAIN cases and finite bounded actions')
    eligible = np.zeros(count, dtype=bool)
    offset = 0
    successful = []
    for group, case in enumerate(manifest['cases']):
        end = offset + case['controls']
        if (case['group'] != group or case['source_id'] != 0 or case['case_id'] != group
                or case['controls'] < 5 or end > count
                or not np.all(groups[offset:end] == group)
                or not np.array_equal(mask[offset:end], np.arange(case['controls']) % 5 != 0)):
            raise ValueError('Measured case/split accounting differs')
        if case['termination_reason'] == 'success':
            eligible[offset:end] = True
            successful.append(group)
        offset = end
    if offset != count:
        raise ValueError('Missing measured controls')
    if set(families[g] for g in successful) != set(families):
        raise ValueError('Successful teacher coverage missing a core TRAIN family')
    return x, y, groups, mask & eligible, ~mask & eligible, successful


def fit_actor(model, inputs, labels, groups, train_mask, holdout_mask, *, seed,
              steps=1024, callback=None):
    """Case-balanced actor gradients; correlated holdout never enters gradients."""
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import actor_critic_parameters
    if type(steps) is not int or not 1 <= steps <= 1024:
        raise ValueError('Actor teaching requires a bounded optimizer budget')
    inputs, labels = np.asarray(inputs), np.asarray(labels)
    groups, train_mask, holdout_mask = map(np.asarray, (groups, train_mask, holdout_mask))
    n = len(inputs)
    if (inputs.ndim != 2 or labels.shape != (n, 2) or groups.shape != (n,)
            or train_mask.shape != (n,) or holdout_mask.shape != (n,)
            or train_mask.dtype != np.bool_ or holdout_mask.dtype != np.bool_
            or not train_mask.any() or not holdout_mask.any() or (train_mask & holdout_mask).any()
            or not np.isfinite(inputs).all() or not np.isfinite(labels).all()):
        raise ValueError('Require finite inputs and disjoint nonempty TRAIN/holdout masks')
    actor, _ = actor_critic_parameters(model.policy)
    fitted = [p for name, p in model.policy.named_parameters()
              if 'policy_net' in name or 'action_net' in name]
    if not fitted or set(map(id, fitted)) - set(map(id, actor)):
        raise ValueError('Require explicitly disjoint actor parameters')
    fixed = {k: v.detach().clone() for k, v in model.policy.state_dict().items()
             if 'policy_net' not in k and 'action_net' not in k}
    ppo_adam = deepcopy(model.policy.optimizer.state_dict())
    counters = model.num_timesteps, model._n_updates
    flags = [p.requires_grad for p in model.policy.parameters()]
    training = model.policy.training
    fitted_ids = set(map(id, fitted))
    for p in model.policy.parameters():
        p.requires_grad_(id(p) in fitted_ids)
    model.policy.set_training_mode(False)
    x = torch.as_tensor(inputs, dtype=torch.float32, device=model.device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=model.device)
    scales = x.new_tensor([.1, .02])
    indices = np.flatnonzero(train_mask)
    unique = np.unique(groups[indices])
    weights = np.empty(len(indices))
    for group in unique:
        keep = groups[indices] == group
        weights[keep] = 1 / (len(unique) * keep.sum())
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(fitted, lr=1e-4)
    history = []
    try:
        for step in range(1, steps + 1):
            selected = rng.choice(indices, 256, replace=True, p=weights)
            mean = model.policy.get_distribution(x[selected]).distribution.mean
            loss = ((mean - y[selected]) / scales).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite teacher loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fitted, .5, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 128 == 0 or step == steps:
                row = dict(auxiliary_steps=step, teacher_scaled_MSE=float(loss.detach()))
                history.append(row)
                if callback:
                    callback(row)
        if (not exact_state(ppo_adam, model.policy.optimizer.state_dict())
                or counters != (model.num_timesteps, model._n_updates)
                or any(not torch.equal(value, model.policy.state_dict()[key]) for key, value in fixed.items())):
            raise ValueError('Actor teaching changed retained PPO/critic/exploration state')
        actual = sorted({int(value['step']) for value in optimizer.state.values()})
        if actual != [steps]:
            raise ValueError('Auxiliary Adam budget differs')
        with torch.no_grad():
            predicted = torch.cat([model.policy.get_distribution(x[i:i + 256]).distribution.mean
                                   for i in range(0, n, 256)]).cpu().numpy()
        metrics = []
        for group in unique:
            row = dict(group=int(group))
            for name, mask in (('train', train_mask), ('correlated_holdout', holdout_mask)):
                keep = mask & (groups == group)
                row[name + '_controls'] = int(keep.sum())
                row[name + '_action_RMS'] = np.sqrt(np.square(predicted[keep] - labels[keep]).mean(0)).tolist()
            metrics.append(row)
        return dict(history=history, case_metrics=metrics, actual_Adam_step_each=actual,
                    auxiliary_parameter_states=len(optimizer.state)), optimizer, rng.bit_generator.state
    finally:
        for p, flag in zip(model.policy.parameters(), flags):
            p.requires_grad_(flag)
        model.policy.set_training_mode(training)


def run(plan_path, seed, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Never overwrite or automatically repeat a measured actor fit')
    plan = read(plan_path)
    if (plan['version'] != 'broader_measured_actor_v1' or plan['steps'] != 1024
            or plan['learning_rate'] != 1e-4 or sha(__file__) != plan['script_sha256']
            or [p['seed'] for p in plan['parents']] != [11, 22, 33, 44, 55]):
        raise ValueError('Require the complete registered fixed-budget five-parent plan')
    row = next(p for p in plan['parents'] if p['seed'] == seed)
    parent, teacher, views = map(Path, (row['batch'], plan['teacher'], plan['views']))
    source = parent / 'source'
    if any(k.startswith('navstack.') for k in sys.modules):
        raise RuntimeError('Each parent requires a fresh frozen-runtime interpreter')
    for name, digest in read(parent / 'protocol.json')['source_hashes'].items():
        if sha(source / name) != digest:
            raise ValueError('Frozen policy source changed')
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from navstack.research import atomic_json, resources_ok
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    from scripts.evaluate_native_rule import require_finished_parent
    from navstack.native_cases import registered_native_cases
    require_finished_parent(parent)
    require_finished_parent(teacher)
    checkpoint = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(checkpoint)
    if metadata['model_sha256'] != row['model_sha256'] or metadata['lineage_control_steps'] != 51200:
        raise ValueError('Require every fixed final broader PPO endpoint')
    protocol = read(teacher / 'protocol.json')
    manifest = read(views / 'manifest.json')
    teacher_registration = manifest['sources']
    reference = Path(plan['reference_config'])
    strict = read(Path(plan['teacher_execution']) / 'strict_delivery.json')
    if (protocol.get('native_validation') is not None or protocol.get('data_role') != 'TRAIN_EXTRA_SUPERVISION'
            or registered_native_cases(protocol) != registered_native_cases(read(parent / 'protocol.json'))
            or len(teacher_registration) != 1 or teacher_registration[0]['batch'] != teacher.name
            or sha(teacher / 'protocol.json') != teacher_registration[0]['protocol_sha256']
            or sha(teacher / 'raw_audit.json') != teacher_registration[0]['raw_audit_sha256']
            or strict['all_engine_controls_acknowledged'] is not True
            or strict['verified_controls'] != manifest['teacher_unique_controls']
            or sha(views / 'manifest.json') != plan['manifest_sha256']
            or sha(Path(plan['teacher_execution']) / 'strict_delivery.json') != plan['strict_delivery_sha256']
            or sha(reference) != manifest['reference_config_sha256']
            or read(checkpoint / 'config.json')['env'] != read(reference)['env']
            or read(checkpoint / 'config.json')['signature'] != read(reference)['signature']):
        raise ValueError('TRAIN/physical/source/delivery provenance differs')
    for name, digest in protocol['source_hashes'].items():
        if sha(teacher / 'source' / name) != digest:
            raise ValueError('Teacher frozen source changed')
    raw = read(teacher / 'raw_audit.json')
    streams = list((teacher / 'infrastructure').glob('*_responses.jsonl.gz'))
    if (len(streams) != 1 or sha(streams[0]) != raw['bridge_response_sha256']
            or sha(teacher / 'evaluation_trace.jsonl.gz') != teacher_registration[0]['trace_sha256']):
        raise ValueError('Measured raw teacher streams changed')
    cases = registered_native_cases(protocol)
    kinds = [c.get('native_hazard', {}).get('kind', 'curve') for c in cases]
    x, y, groups, mask, holdout, success = validated_arrays(views / 'measured_demonstrations.npz', manifest, kinds)
    torch.set_num_threads(1)
    model = load_ppo(checkpoint / 'model.zip', device='cpu')
    with (checkpoint / 'normalization.pkl').open('rb') as stream:
        norm = pickle.load(stream)
    before_rms = pickle.dumps(norm.obs_rms)
    normalized = norm.normalize_obs(x)
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / 'executed_fit.py')
    atomic_json(output / 'registration.json', dict(plan_sha256=sha(plan_path), seed=seed,
        source_model_sha256=metadata['model_sha256'], dataset_sha256=manifest['dataset_sha256'],
        successful_TRAIN_cases=success, recorded_cases=36, excluded_failed_cases=sorted(set(range(36)) - set(success)),
        teacher_controls=manifest['teacher_unique_controls'], new_controls=0, new_PPO_updates=0))
    started = time.monotonic()
    def health(record):
        okay, reason = resources_ok(plan['workspace'])
        if not okay or time.monotonic() - started > 600:
            raise RuntimeError('Bounded measured fit guard: ' + str(reason))
        atomic_json(output / 'heartbeat.json', record)
        print(json.dumps(record), flush=True)
    health(dict(auxiliary_steps=0))
    result, optimizer, sampler = fit_actor(model, normalized, y, groups, mask, holdout,
                                          seed=seed, steps=1024, callback=health)
    if pickle.dumps(norm.obs_rms) != before_rms:
        raise ValueError('Observation RMS changed')
    destination = output / 'checkpoint'
    shutil.copytree(checkpoint, destination)
    model.save(destination / 'model.zip')
    metadata.update(model_sha256=sha(destination / 'model.zip'), measured_actor_steps=1024,
        measured_actor_plan_sha256=sha(plan_path),
        note='1024 separate TRAIN-only actor Adam steps. Critic/log_std/PPO Adam/counters/RMS/RNG retained. Fresh native rollout required; not new PPO learning.')
    atomic_json(destination / 'metadata.json', metadata)
    verified_checkpoint_metadata(destination)
    torch.save(optimizer.state_dict(), output / 'auxiliary_Adam.pt')
    atomic_json(output / 'sampler_state.json', sampler)
    result.update(seed=seed, model_sha256=metadata['model_sha256'], new_controls=0, new_PPO_updates=0,
        auxiliary_Adam_sha256=sha(output / 'auxiliary_Adam.pt'), elapsed_seconds=time.monotonic() - started,
        PPO_critic_logstd_optimizer_counters_RMS_retained=True,
        holdout_scope='Correlated per-episode diagnostic only; no gradients, independent validation or model selection.',
        research_acceptance='UNMET')
    atomic_json(output / 'complete.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--seed', required=True, type=int, choices=[11, 22, 33, 44, 55])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.plan, args.seed, args.output)

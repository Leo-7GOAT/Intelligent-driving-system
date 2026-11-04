"""Compare measured initial inputs with TRAIN support; no learning or driving."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys


def run(batch, views, output):
    batch, views, output = map(lambda p: Path(p).resolve(), (batch, views, output))
    if output.exists():
        raise FileExistsError('Preserve the previous diagnostic')
    if any(k.startswith('navstack.') for k in sys.modules):
        raise RuntimeError('Use a fresh frozen-runtime interpreter')
    runtime = batch / 'source'
    os.environ['NAVSTACK_AUDIT_SOURCE_ROOT'] = str(runtime)
    sys.path.insert(0, str(runtime))
    import numpy as np
    from scripts import audit_native_policy_inference as helper
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = read(views / 'manifest.json')
    dataset = views / 'measured_demonstrations.npz'
    if sha(dataset) != manifest['dataset_sha256']:
        raise ValueError('TRAIN measured views changed')
    captured = []
    original = helper.NavigationEnv._observation
    def capture(env):
        observation = original(env)
        captured.append(observation.copy())
        return observation
    helper.NavigationEnv._observation = capture
    initial_audit = output.with_name(output.stem + '_initial_inference.json')
    try:
        helper.analyze(batch, 'B1', initial_audit)
    finally:
        helper.NavigationEnv._observation = original
    inference = read(initial_audit)
    if len(captured) != len(inference['results']):
        raise ValueError('Initial observation/inference inventory differs')
    observed = np.stack(captured)
    with (batch / 'B1/normalization.pkl').open('rb') as stream:
        norm = pickle.load(stream)
    before = pickle.dumps(norm.obs_rms)
    with np.load(dataset, allow_pickle=False) as arrays:
        successful = [c['group'] for c in manifest['cases'] if c['termination_reason'] == 'success']
        selected = arrays['train_mask'] & np.isin(arrays['groups'], successful)
        training = arrays['observations'][selected]
    dimensions = np.r_[np.arange(351), np.arange(599, 605), np.arange(4701, 4705)]
    if observed.shape[1] != 4705 or training.shape[1] != 4705:
        raise ValueError('Require the fixed direct-policy observation schema')
    scale = np.sqrt(norm.obs_rms.var[dimensions] + norm.epsilon)
    train_z = (training[:, dimensions] - norm.obs_rms.mean[dimensions]) / scale
    measured_z = (observed[:, dimensions] - norm.obs_rms.mean[dimensions]) / scale
    np.testing.assert_allclose(np.clip(measured_z, -norm.clip_obs, norm.clip_obs).astype(np.float32),
        norm.normalize_obs(observed)[:, dimensions], atol=1e-6, rtol=0)
    clipped = np.abs(measured_z) > norm.clip_obs
    outside = ((observed[:, dimensions] < training[:, dimensions].min(0) - 1e-6)
               | (observed[:, dimensions] > training[:, dimensions].max(0) + 1e-6))
    episodes = {e['case_id']: e for e in read(batch / 'B1/evaluation.json')['episodes']}
    cases = [dict(case_id=row['case_id'], termination=episodes[row['case_id']]['termination_reason'],
        clipped_normalized_dimensions=int(clipped[i].sum()), outside_TRAIN_range_dimensions=int(outside[i].sum()),
        original_action=row['action']) for i, row in enumerate(inference['results'])]
    order = np.argsort(-clipped.sum(0), kind='stable')
    top = [dict(dimension=int(dimensions[i]), clipped_initial_cases=int(clipped[:, i].sum()),
                outside_TRAIN_cases=int(outside[:, i].sum()), TRAIN_clip_fraction=float((np.abs(train_z[:, i]) > norm.clip_obs).mean()),
                stored_RMS_std=float(np.sqrt(norm.obs_rms.var[dimensions[i]])),
                TRAIN_raw_range=[float(training[:, dimensions[i]].min()), float(training[:, dimensions[i]].max())],
                measured_initial_range=[float(observed[:, dimensions[i]].min()), float(observed[:, dimensions[i]].max())])
           for i in order[:30] if clipped[:, i].any() or outside[:, i].any()]
    if pickle.dumps(norm.obs_rms) != before:
        raise ValueError('Diagnostic changed observation statistics')
    report = dict(scope=__doc__, batch=batch.name, model_sha256=inference['model_sha256'],
        dataset_sha256=manifest['dataset_sha256'], initial_inference_sha256=sha(initial_audit),
        TRAIN_fitting_rows=len(training), normalized_dimensions=len(dimensions),
        TRAIN_clip_fraction=float((np.abs(train_z) > norm.clip_obs).mean()),
        measured_initial_clip_fraction=float(clipped.mean()),
        cases=cases, top_dimensions=top, new_controls=0, new_optimizer_updates=0,
        limits='Initial measured states only;TRAIN support is descriptive,not a causal diagnosis or proposed runtime clamp. No VAL labels/statistics enter fitting or RMS updates.')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(batch=batch.name, TRAIN_clip_fraction=report['TRAIN_clip_fraction'],
        initial_clip_fraction=report['measured_initial_clip_fraction'], top_dimensions=top[:5])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--views', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.batch, args.views, args.output)

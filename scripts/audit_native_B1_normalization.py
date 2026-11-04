"""Cross saved B1 weights/statistics on fixed teacher-training observations only."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def audit(batch, checkpoint, output):
    batch, checkpoint, output = map(lambda p: Path(p).resolve(), (batch, checkpoint, output))
    if output.exists():
        raise FileExistsError('Preserve previous counterfactual evidence')
    protocol = read(batch/'protocol.json')
    for name, digest in protocol['source_hashes'].items():
        if sha(batch/'source'/name) != digest:
            raise ValueError('Frozen source changed')
    sys.path.insert(0, str(batch/'source'))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    torch.set_num_threads(1)
    dataset = batch/'measured_demonstrations.npz'
    if sha(dataset) != protocol['extra_supervision']['dataset_sha256']:
        raise ValueError('Teacher dataset changed')
    with np.load(dataset, allow_pickle=False) as arrays:
        eligible = np.flatnonzero(arrays['train_mask'])
        selected = eligible[::max(1, int(np.ceil(len(eligible)/512)))]
        observations, labels = arrays['observations'][selected], arrays['actions'][selected]
    folders = [batch/'resume/B1', checkpoint]
    models, norms, inputs = [], [], []
    for folder in folders:
        metadata = read(folder/'metadata.json')
        for name, key in [('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256')]:
            if sha(folder/name) != metadata[key]:
                raise ValueError('Checkpoint file differs from registered metadata')
        models.append(load_ppo(folder/'model.zip', device='cpu'))
        with (folder/'normalization.pkl').open('rb') as stream:
            norm = pickle.load(stream)
        norm.training = False
        norms.append(norm)
        inputs.append({'physics_steps': metadata['physics_steps'], 'model_sha256': sha(folder/'model.zip'),
                       'normalization_sha256': sha(folder/'normalization.pkl'), 'RMS_count': float(norm.obs_rms.count)})
    actions, combinations = {}, []
    for weights, model in enumerate(models):
        for statistics, norm in enumerate(norms):
            before = pickle.dumps(norm.obs_rms)
            predicted = model.predict(norm.normalize_obs(observations), deterministic=True)[0]
            if pickle.dumps(norm.obs_rms) != before:
                raise ValueError('Inference changed normalization statistics')
            actions[weights, statistics] = predicted
            combinations.append({'weights': weights, 'statistics': statistics,
                                 'per_component_teacher_MSE': np.mean((predicted-labels)**2, axis=0).tolist()})
    contrasts = []
    for label, a, b in [('statistics_only', (0, 0), (0, 1)),
                         ('weights_with_initial_statistics', (0, 0), (1, 0)),
                         ('weights_with_updated_statistics', (0, 1), (1, 1))]:
        difference = np.abs(actions[a]-actions[b])
        contrasts.append({'contrast': label, 'median_abs_action_change': np.median(difference, axis=0).tolist(),
                          'p95_abs_action_change': np.quantile(difference, .95, axis=0).tolist()})
    result = {'scope': 'Same fixed teacher-training observations; counterfactual inference, not closed-loop outcomes or causal training effects.',
              'batch': batch.name, 'dataset_sha256': sha(dataset), 'selected_training_views': len(selected),
              'selected_index_sha256': hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
              'action_components': ['normalized_acceleration', 'normalized_steering'],
              'inputs': inputs, 'combinations': combinations, 'contrasts': contrasts,
              'new_physical_controls': 0, 'optimizer_updates': 0, 'teacher_queries': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('batch', 'checkpoint', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    audit(args.batch, args.checkpoint, args.output)

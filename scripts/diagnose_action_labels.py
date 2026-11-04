"""Offline component errors on identical visits; no closed-loop success claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from navstack.experiments import load_run
from scripts.verify_relabel_teacher import assert_action_only_change


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(checkpoint, relabel, output):
    checkpoint, relabel, output = Path(checkpoint), Path(relabel), Path(output)
    if output.exists():
        raise ValueError('Preserve existing diagnostic; choose a new output path')
    torch.set_num_threads(1)
    old, new = np.load(relabel/'input/original.npz'), np.load(relabel/'original.npz')
    assert_action_only_change(old, new)
    cases = json.loads((relabel/'input/original_protocol.json').read_text(encoding='utf-8'))['cases']
    model, norm = load_run(checkpoint)
    rows = []
    try:
        assert not norm.training
        before_count = norm.obs_rms.count
        predictions = []
        for start in range(0, len(new['actions']), 256):
            normalized = norm.normalize_obs(new['observations'][start:start+256])
            with torch.no_grad():
                mean = model.policy.get_distribution(torch.as_tensor(normalized, dtype=torch.float32, device=model.device)).distribution.mean
            predictions.append(mean.cpu().numpy())
        means = np.concatenate(predictions)
        predictions = means.clip(-1, 1)
        splits = np.array([cases[int(i)]['data_split'] for i in new['case_id']])
        scenes = np.array([cases[int(i)]['scenario'] for i in new['case_id']])
        for split in ('train', 'holdout'):
            for scene in sorted(set(scenes)):
                base = (splits == split) & (scenes == scene)
                for target_name, targets in [('held_reference', old['actions']), ('current_state', new['actions'])]:
                    for phase, selected in [('all', base), ('turning', base & (np.abs(targets[:, 1]) > .02)),
                                            ('braking', base & (targets[:, 0] < -.05))]:
                        count = int(selected.sum())
                        error = predictions[selected]-targets[selected]
                        rows.append({'split': split, 'scenario': scene, 'target': target_name, 'phase': phase, 'controls': count,
                                     'mse_longitudinal_steering': (error**2).mean(0).tolist() if count else None,
                                     'absolute_error_p95': np.quantile(np.abs(error), .95, axis=0).tolist() if count else None,
                                     'mean_outside_bounds_fraction': (np.abs(means[selected]) > 1).mean(0).tolist() if count else None})
        assert norm.obs_rms.count == before_count
        result = {'scope': __doc__, 'model_sha256': sha(checkpoint/'model.zip'),
                  'normalization_sha256': sha(checkpoint/'normalization.pkl'),
                  'input_sha256': sha(relabel/'input/original.npz'), 'relabel_sha256': sha(relabel/'original.npz'),
                  'additional_interactions': 0, 'additional_teacher_queries': 0, 'gradient_updates': 0,
                  'phase_definition': 'Turning: absolute normalized target steering >0.02; braking: normalized target longitudinal <-0.05. Thresholds are diagnostic slices, not selected performance endpoints.',
                  'rows': rows}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps([r for r in rows if r['split'] == 'holdout' and r['scenario'] == 'static']))
        return result
    finally:
        norm.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--relabel', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    diagnose(args.checkpoint, args.relabel, args.output)

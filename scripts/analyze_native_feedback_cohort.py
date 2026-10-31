"""Exploratory paired-seed statistics on audited DEVELOPMENT conditions."""
import argparse
from itertools import product
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_native_feedback_pair import analyze
from scripts.audit_native_method_endpoint import read, sha
from scripts.audit_native_ppo_configuration import profile


def paired_seed_statistics(differences):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 5 or not np.isfinite(values).all() or np.any(np.abs(values) > 1):
        raise ValueError('Require at least five bounded finite independent-seed differences')
    mean = float(values.mean())
    error = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = float(t.ppf(.975, len(values)-1) * error)
    if len(values) > 16:
        raise ValueError('Exact exploratory sign enumeration is bounded to16seeds')
    null = [abs(float(np.mean(values * signs))) for signs in product((-1, 1), repeat=len(values))]
    return {'independent_training_seeds': len(values), 'paired_mean_difference': mean,
            'paired_95pct_t_interval': [mean-margin, mean+margin],
            'exploratory_exact_sign_flip_p_two_sided': sum(x >= abs(mean)-1e-12 for x in null) / len(null),
            'assumptions': 'tinterval assumes approximately normal seed differences;sign-flip null assumes symmetric exchangeable seed differences. Small N,known development conditions and retrospective analysis do not establish confirmatory significance.'}


def run(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = read(manifest_path)
    pairs = manifest['pairs']
    if (len(pairs) < 5 or len({r['seed'] for r in pairs}) != len(pairs)
            or any(type(r['seed']) is not int for r in pairs)):
        raise ValueError('Require at least five unique independently trained seed identifiers')
    if output.exists():
        raise FileExistsError('Preserve previous statistical reconstruction')
    output.mkdir(parents=True)
    results, initial_rng = [], {'H': [], 'H0': []}
    recipes = {'H': [], 'H0': []}
    tasks, overshoot = [], []
    for row in pairs:
        h, h0 = [(ROOT / row[k]).resolve() for k in ('h', 'h0')]
        for batch, method in ((h, 'H'), (h0, 'H0')):
            protocol, summary = read(batch/'protocol.json'), read(batch/method/'summary.json')
            if (protocol['seed'] != row['seed'] or summary['physics_steps'] <= 0
                    or summary['optimizer_steps'] <= 0 or summary['stopped_reason'] is not None
                    or protocol.get('method_budgets', {}).get(method, protocol['controlled_budget_per_method']) != manifest['requested_controls_each']):
                raise ValueError('Incomplete or mismatched actual PPOtraining/budget')
            origin = batch / protocol['recovery'][method]['checkpoint']
            recipe = profile(origin)  # Enforces zero initial PPO/Adam state.
            recipes[method].append(recipe)
            configuration = read(batch/method/'config.json')['env']
            allowance = recipe['n_steps'] * round(configuration['decision_dt']/configuration['control_dt'])
            extra = summary['physics_steps'] - manifest['requested_controls_each']
            if not 0 <= extra <= allowance:
                raise ValueError('Training did not reach its budget or exceeded one declared rollout overshoot')
            overshoot.append({'seed': row['seed'], 'method': method, 'controls_over_requested_budget': extra})
            tasks.append({k: protocol.get(k) for k in ('routes', 'native_case_manifest', 'native_validation')})
            with (origin/'resume_state.pkl').open('rb') as stream:
                state = pickle.load(stream)
            initial_rng[method].append(hashlib.sha256(state['torch'].numpy().tobytes()).hexdigest())
        result = analyze(h, h0, output/('seed'+str(row['seed'])+'_pair.json'))
        result['walker_initialization_source_equal'] = sha(h/'source/scripts/carla_bridge37.py') == sha(h0/'source/scripts/carla_bridge37.py')
        results.append(result)
    for method in ('H', 'H0'):
        if len(set(initial_rng[method])) != len(pairs) or any(p != recipes[method][0] for p in recipes[method]):
            raise ValueError('Repeated initial Torch RNG or differing saved learner recipes')
    if any(task != tasks[0] for task in tasks):
        raise ValueError('Development condition inventories differ across seeds')
    left, right = [dict(recipes[m][0]) for m in ('H', 'H0')]
    left.pop('policy_class');right.pop('policy_class')
    if left != right:
        raise ValueError('H/H0learners differ beyond the declared feedback policy class')
    stats = paired_seed_statistics([r['paired_difference_percentage_points']/100 for r in results])
    result = {'scope': manifest['scope'], 'pairs_manifest_sha256': sha(manifest_path),
              'seeds': [r['seed'] for r in pairs], 'results': results, 'statistics': stats,
              'H_successes': sum(r['H_successes'] for r in results),
              'H0_successes': sum(r['H0_successes'] for r in results),
              'episodes_per_method': sum(r['cases'] for r in results),
              'unique_condition_count': results[0]['cases'],
              'initial_Torch_RNG_sha256': initial_rng,
              'identical_saved_learner_recipe_within_method': True,
              'matched_H_H0learner_except_declared_policy_class': True,
              'physical_rollout_overshoot_disclosed': overshoot,
              'walker_initialization_mismatch_seeds': [r['seed'] for r in results if not r['walker_initialization_source_equal']],
              'limits': 'All five seeds retained;seed22uses fresh H0budget replica while its older interrupted/recovered branch remains charged separately. Bridge source equality/mismatches are explicitly recorded per pair. Conditions are reused development layouts,not fresh final confirmation. No B1/RL-vs-rule superiority claim.',
              'new_controls': 0, 'optimizer_updates': 0, 'research_acceptance': 'UNMET'}
    (output/'cohort.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('results', 'initial_Torch_RNG_sha256')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.pairs_manifest, args.output)

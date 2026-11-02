"""Combine all five H/H0 measured-input feedback diagnostics without selecting seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


SEEDS = (11, 22, 33, 44, 55)
VARIANTS = ('candidate_witnesses', 'previous_execution', 'both')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(inputs, pairs_manifest):
    inputs, pairs_manifest = Path(inputs), Path(pairs_manifest)
    pairs = json.loads(pairs_manifest.read_text(encoding='utf-8'))['pairs']
    if [p['seed'] for p in pairs] != list(SEEDS):
        raise ValueError('Require every original independent seed in fixed order')
    expected_script = sha(Path(__file__).with_name('diagnose_native_feedback_use.py'))
    all_records = {'H': [], 'H0': []}
    history = {'H': 0, 'H0': 0}
    receipts, per_seed = [], []
    for pair in pairs:
        seed_results = {'seed': pair['seed']}
        for method in ('H', 'H0'):
            path = inputs / f"native_feedback_use_seed{pair['seed']}_{method}_20261101.json"
            d = json.loads(path.read_text(encoding='utf-8'))
            records = d['records']
            ids = [(r['case_id'], r['decision_index']) for r in records]
            if (d['script_sha256'] != expected_script or d['batch'] != Path(pair[method.lower()]).name
                    or d['method'] != method or d['cases'] != 24 or d['sampling_stride'] != 16
                    or d['sampled_actions_exact'] != len(records) or ids != sorted(set(ids))
                    or {r['case_id'] for r in records} != set(range(24))
                    or not d['original_physical_mask_retained'] or not d['parameters_and_RMS_unchanged']
                    or d['new_controls'] != 0 or d['new_optimizer_updates'] != 0):
                raise ValueError('Incomplete or incompatible measured-input diagnostic')
            for variant in VARIANTS:
                values = [r['counterfactuals'][variant] for r in records]
                if (any(not 0 <= v['total_variation'] <= 1 for v in values)
                        or d['totals'][variant]['argmax_changes'] != sum(v['argmax_changed'] for v in values)
                        or method == 'H0' and any(v['total_variation'] != 0 or v['argmax_changed'] for v in values)):
                    raise ValueError('Invalid probability change or non-invariant H0 negative control')
            receipts.append({'seed': pair['seed'], 'method': method, 'file': path.name, 'sha256': sha(path),
                'batch': d['batch'], 'protocol_sha256': d['protocol_sha256'], 'model_sha256': d['model_sha256'],
                'normalization_sha256': d['normalization_sha256'], 'trace_sha256': d['trace_sha256']})
            seed_results[method] = d['totals']
            all_records[method].extend(records)
            history[method] += d['measured_history_controls']
        per_seed.append(seed_results)
    totals = {}
    for method, records in all_records.items():
        available = [r['available_actions'] for r in records]
        totals[method] = {'sampled_measured_decisions': len(records), 'accounted_measured_history_controls': history[method],
            'available_actions': {'min': min(available), 'mean': statistics.fmean(available), 'max': max(available),
                'single_available_action_states': sum(a == 1 for a in available)},
            'hidden_inputs': {variant: {'argmax_changes': sum(r['counterfactuals'][variant]['argmax_changed'] for r in records),
                'argmax_change_fraction': statistics.fmean(r['counterfactuals'][variant]['argmax_changed'] for r in records),
                'mean_total_variation': statistics.fmean(r['counterfactuals'][variant]['total_variation'] for r in records),
                'max_total_variation': max(r['counterfactuals'][variant]['total_variation'] for r in records)} for variant in VARIANTS}}
    return {'scope': __doc__, 'script_sha256': sha(__file__), 'diagnostic_script_sha256': expected_script,
        'pairs_manifest_sha256': sha(pairs_manifest), 'independent_training_seeds': list(SEEDS),
        'unique_known_development_conditions': 24, 'new_final_conditions': 0,
        'sampling': 'Every sixteenth decision plus each case first/last; sampled original actions exactly reconstructed. All measured frame histories accounted for.',
        'physical_mask': 'Original endpoint availability mask retained for every counterfactual; learned normalized candidate witnesses and/or previous execution features zeroed.',
        'totals': totals, 'per_seed': per_seed, 'input_receipts': receipts,
        'H0_negative_control_exact': True, 'new_controls': 0, 'new_optimizer_updates': 0,
        'interpretation': 'H uses learned feedback, while hiding it changes few sampled greedy actions. H0 is exactly invariant as designed. The mask does not force a single available action in these samples. This local sensitivity does not establish useful decisions or an improvement in physical performance.',
        'limits': 'Retrospective development diagnostic, not an actual counterfactual rollout, all-decision action audit, causal success-rate estimate or proof of PPO necessity. Keep the zero-crossing original H/H0 performance interval.',
        'research_acceptance': 'UNMET'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--pairs-manifest', default='protocols/native_matched_broader_feedback_pairs_20261101.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Preserve previous summary')
    result = summarize(args.inputs, args.pairs_manifest)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['totals']), flush=True)

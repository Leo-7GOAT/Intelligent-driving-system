"""Locate planning failures in all five existing H/H0 development pairs, retaining every outcome."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(manifest):
    manifest = Path(manifest)
    pairs = read(manifest)['pairs']
    if [p['seed'] for p in pairs] != [11, 22, 33, 44, 55]:
        raise ValueError('Keep all five independent pairs in registered order')
    rows, inputs, concentrations = [], [], []
    totals = {m: {'successes': 0, 'cases': 0, 'all_plan_failures': 0,
        'mutually_successful_plan_failures': 0, 'other_plan_failures': 0} for m in ('H', 'H0')}
    for pair in pairs:
        evaluations = {}
        for method in ('H', 'H0'):
            batch = Path(pair[method.lower()])
            folder = batch / method
            path, raw = folder / 'evaluation.json', read(folder / 'endpoint_raw_audit.json')
            if sha(path) != raw['evaluation_sha256'] or sha(batch / 'protocol.json') != raw['protocol_sha256']:
                raise ValueError('Evaluation/protocol changed after its raw CARLA audit')
            episodes = read(path)['episodes']
            if sorted(e['case_id'] for e in episodes) != list(range(24)):
                raise ValueError('Every known validation condition is required')
            trace_path = folder / 'evaluation_trace.jsonl.gz'
            if sha(trace_path) != raw['evaluation_trace_sha256']:
                raise ValueError('Actual decision trace changed')
            failures = Counter()
            with gzip.open(trace_path, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    decision = json.loads(line)
                    failures[decision['case_id']] += decision['info']['mode'] == 'plan_failed'
            if any(failures[e['case_id']] != e['plan_failures'] for e in episodes):
                raise ValueError('Summary planning failures differ from actual recorded decision modes')
            evaluations[method] = episodes
            inputs.append({'seed': pair['seed'], 'method': method, 'batch': batch.name,
                'evaluation_sha256': sha(path), 'trace_sha256': sha(trace_path),
                'planning_failures_recounted_from_actual_trace': True,
                'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json')})
        mutual = set.intersection(*({e['case_id'] for e in episodes if e['success']} for episodes in evaluations.values()))
        row = {'seed': pair['seed'], 'mutually_successful_case_ids': sorted(mutual)}
        for method, episodes in evaluations.items():
            values = {'successes': sum(e['success'] for e in episodes), 'cases': 24,
                'all_plan_failures': sum(e['plan_failures'] for e in episodes),
                'mutually_successful_plan_failures': sum(e['plan_failures'] for e in episodes if e['case_id'] in mutual),
                'other_plan_failures': sum(e['plan_failures'] for e in episodes if e['case_id'] not in mutual)}
            row[method] = values
            for key, value in values.items():
                totals[method][key] += value
            for episode in episodes:
                if episode['plan_failures']:
                    concentrations.append({'seed': pair['seed'], 'method': method, 'case_id': episode['case_id'],
                        'termination': episode['termination_reason'], 'plan_failures': episode['plan_failures'],
                        'actions': episode['actions'], 'controls': episode['physics_steps'],
                        'interventions': episode['interventions'], 'mutually_successful': episode['case_id'] in mutual})
        rows.append(row)
    return {'scope': __doc__, 'script_sha256': sha(__file__), 'pairs_manifest_sha256': sha(manifest),
        'totals': totals, 'per_seed': rows, 'episodes_with_planning_failures': concentrations, 'input_hashes': inputs,
        'unique_known_development_conditions': 24, 'independent_training_pairs': 5,
        'interpretation': f"Planning failures are concentrated in {len(concentrations)} policy/seed/condition episodes. Inspect all-outcome and conditional per-seed counts together;aggregate counts alone do not establish stable planning efficiency or collaboration benefit.",
        'limits': 'Retrospective concentration analysis. Mutually successful cases are an explicitly conditional sensitivity analysis,not replacement completion statistics;the failed case remains in all-outcome counts. No new comparison or selected subgroup is promoted to a main result.',
        'new_controls': 0, 'new_optimizer_updates': 0, 'new_final_conditions': 0, 'research_acceptance': 'UNMET'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='protocols/native_matched_broader_feedback_pairs_20261101.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Preserve previous concentration report')
    result = analyze(args.manifest)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['totals']))

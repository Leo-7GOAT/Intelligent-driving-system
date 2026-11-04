"""Cost ledger for the complete audited five-pair,24-known-condition cohort."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_native_feedback_cohort import resolve_batch
from scripts.analyze_native_pristine_baselines import compare_cases
from scripts.audit_native_method_endpoint import read, sha


def analyze(cohort_path, inputs_root, output):
    cohort_path, output = Path(cohort_path), Path(output)
    if output.exists():
        raise FileExistsError('Preserve earlier cost reconstruction')
    cohort_digest = sha(cohort_path)
    cohort = read(cohort_path)
    if sha(cohort_path) != cohort_digest:
        raise ValueError('Cohort changed while being read')
    if (cohort.get('cohort_kind') != 'matched_broader_36train_24val_v1'
            or cohort.get('seeds') != [11, 22, 33, 44, 55]
            or cohort.get('unique_condition_count') != 24
            or cohort.get('identical_complete_execution_source_across_all_pairs') is not True
            or len(cohort.get('results', [])) != 5):
        raise ValueError('Require the complete previously audited matched cohort')
    rows, totals = [], {m: Counter() for m in ('H', 'H0')}
    for seed, pair in zip(cohort['seeds'], cohort['results']):
        if pair['seed'] != seed or len(pair['source_and_model_proof']) != 2:
            raise ValueError('Incomplete or reordered seed evidence')
        evaluations, bindings = {}, {}
        for method, proof in zip(('H', 'H0'), pair['source_and_model_proof']):
            folder = resolve_batch(inputs_root, 'runs/' + proof['batch']) / method
            raw_path = folder / 'endpoint_raw_audit.json'
            if sha(raw_path) != proof['raw_audit_sha256']:
                raise ValueError('Raw audit differs from the complete cohort')
            raw = read(raw_path)
            for name, key in (('evaluation.json', 'evaluation_sha256'), ('summary.json', 'summary_sha256')):
                if sha(folder / name) != raw[key]:
                    raise ValueError('Raw-bound endpoint changed: ' + name)
            evaluation, summary = read(folder / 'evaluation.json'), read(folder / 'summary.json')
            if (summary != proof['summary'] or evaluation['model_sha256'] != proof['model_sha256']
                    or len(evaluation['episodes']) != 24):
                raise ValueError('Different endpoint or incomplete condition inventory')
            evaluations[method] = evaluation['episodes']
            bindings[method] = {'raw_audit_sha256': sha(raw_path),
                'training_controls': summary['physics_steps'], 'training_decisions': summary['actions'],
                'actual_Adam_steps': summary['optimizer_steps'], 'training_wall_seconds': summary['wall_seconds'],
                'evaluation_controls': raw['evaluation_control_frames_verified']}
            totals[method].update({k: v for k, v in bindings[method].items() if k != 'raw_audit_sha256'})
            for episode in evaluation['episodes']:
                totals[method].update({k: episode[k] for k in
                    ('interventions', 'plan_failures', 'mpc_failures', 'simulation_seconds')})
        rows.append({'seed': seed, 'endpoints': bindings,
                     'paired_execution_costs': compare_cases(evaluations['H'], evaluations['H0'])})
    if sha(cohort_path) != cohort_digest:
        raise ValueError('Cohort changed during reconstruction')
    report = {'scope': __doc__, 'cohort_sha256': cohort_digest,
        'analysis_source_sha256': sha(Path(__file__)), 'results': rows,
        'totals': {m: dict(v) for m, v in totals.items()},
        'limits': 'All outcomes retained. Travel-time comparisons use only the listed mutually successful cases. '
                  'All-outcome duration is a cost,not a speed claim. These are new-cohort costs,not a whole-project '
                  'ledger;earlier training,teacher fitting,failed experiments and reconstruction costs remain separate. '
                  '24known conditions repeated across five trained policies are not120unique final conditions.',
        'new_controls_for_analysis': 0, 'optimizer_updates_for_analysis': 0, 'research_acceptance': 'UNMET'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--inputs-root', default=str(ROOT))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cohort, args.inputs_root, args.output)['totals']))

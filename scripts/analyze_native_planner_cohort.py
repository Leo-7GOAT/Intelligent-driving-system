"""Pair all five trained H/B2 policies on the complete known DEVELOPMENT inventory."""
import argparse
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_native_feedback_cohort import paired_seed_statistics, resolve_batch
from scripts.audit_native_method_endpoint import read, sha
from scripts.audit_native_ppo_configuration import verify as verify_learner


def verify_planner_change(h, b2):
    left, right = deepcopy(h), deepcopy(b2)
    if (left['method'], right['method']) != ('H', 'B2'):
        raise ValueError('Require the declared H/B2 methods')
    if (left['planner']['grid_postprocess'], right['planner']['grid_postprocess']) != ('legacy', 'body_spline_v1'):
        raise ValueError('Require body-feasible Grid A* postprocessing')
    right['method'] = left['method']
    right['planner']['grid_postprocess'] = left['planner']['grid_postprocess']
    if left != right:
        raise ValueError('Environment differs beyond the declared planner replacement')


def require_complete_cohort(state):
    if (state['phase'] != 'ALL_FIVE_REGISTERED_B2_PPO_SEEDS_COMPLETE'
            or [r['seed'] for r in state['completed']] != [11, 22, 33, 44, 55]):
        raise ValueError('All five B2 seeds must complete before cohort inference')


def endpoint(batch, method, count):
    folder = batch / method
    raw, evaluation, summary = [read(folder / name) for name in
                               ('endpoint_raw_audit.json', 'evaluation.json', 'summary.json')]
    for name, key in (('evaluation_trace.jsonl.gz', 'evaluation_trace_sha256'),
                      ('config.json', 'config_sha256')):
        if sha(folder / name) != raw[key]:
            raise ValueError('Raw endpoint audit changed: ' + name)
    if sha(batch / 'protocol.json') != raw['protocol_sha256']:
        raise ValueError('Raw audit protocol changed')
    for name, key in (('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256')):
        if sha(folder / name) != evaluation[key]:
            raise ValueError('Evaluated learning artifact changed')
    episodes = evaluation['episodes']
    if (len(episodes) != count or sorted(r['case_id'] for r in episodes) != list(range(count))
            or summary['optimizer_steps'] <= 0 or summary['stopped_reason'] is not None
            or summary['physics_steps'] != raw['training_control_frames_verified']):
        raise ValueError('Incomplete actual PPO or evaluated condition inventory')
    return {r['case_id']: r for r in episodes}, {
        'batch': batch.name, 'method': method, 'model_sha256': evaluation['model_sha256'],
        'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'),
        'training_controls': summary['physics_steps'], 'actual_Adam_steps': summary['optimizer_steps'],
        'raw_controls': raw['total_applied_controls_verified']}


def run(pairs_manifest, execution, output):
    pairs_manifest, execution, output = map(Path, (pairs_manifest, execution, output))
    state = read(execution / 'status.json'); require_complete_cohort(state)
    if (execution / 'failure.json').exists():
        raise ValueError('Retained B2 cohort failure')
    pairs = read(pairs_manifest)['pairs']
    if [r['seed'] for r in pairs] != [11, 22, 33, 44, 55]:
        raise ValueError('Require the canonical all-five H inventory')
    if output.exists():
        raise FileExistsError('Preserve the previous planner comparison')
    results = []; inventory = None
    for pair, trained in zip(pairs, state['completed']):
        h, b2 = resolve_batch(ROOT, pair['h']), resolve_batch(ROOT, 'runs/' + trained['batch'])
        protocols = [read(p / 'protocol.json') for p in (h, b2)]
        if any(p['seed'] != pair['seed'] for p in protocols):
            raise ValueError('Actual policy seed differs from the paired inventory')
        for key in ('seed', 'routes', 'native_case_manifest', 'native_validation', 'calibration_sha256',
                    'shared_endpoint_mask_v1', 'rollout_actions', 'ppo_epochs', 'episode_seconds', 'road_geometry_version'):
            if protocols[0].get(key) != protocols[1].get(key):
                raise ValueError('Pair differs in ' + key)
        cases = protocols[0]['native_case_manifest']['cases']
        if inventory is None: inventory = cases
        if cases != inventory: raise ValueError('Conditions differ across seeds')
        for p, method in ((h, 'H'), (b2, 'B2')):
            budget = protocols[0]['method_budgets']['H']
            protocol = read(p / 'protocol.json')
            if protocol['method_budgets'][method] != budget or budget != 20480:
                raise ValueError('Unequal requested physical training budget')
        verify_planner_change(read(h / 'H/config.json')['env'], read(b2 / 'B2/config.json')['env'])
        origins = [p / protocol['recovery'][m]['checkpoint'] for p, protocol, m in
                   zip((h, b2), protocols, ('H', 'B2'))]
        verify_learner(origins[1], origins[0])
        for name in ('model.zip', 'normalization.pkl', 'resume_state.pkl'):
            if sha(origins[0] / name) != sha(origins[1] / name):
                raise ValueError('Initial weights/statistics/RNG differ')
        critical = ('env', 'planning', 'mpc', 'trajectory', 'reward', 'road_grid', 'road_grid_batch',
                    'native_route', 'config', 'candidate_filter', 'road_candidate_policy', 'carla_actuation')
        for name in critical:
            relative = 'source/navstack/' + name + '.py'
            if sha(h / relative) != sha(b2 / relative):
                raise ValueError('Unmatched physical/policy implementation: ' + name)
        left, hp = endpoint(h, 'H', len(cases)); right, bp = endpoint(b2, 'B2', len(cases))
        if any(not budget <= p['training_controls'] <= budget + 320 for p in (hp, bp)):
            raise ValueError('Training budget or one-rollout overshoot differs')
        families = {}
        for label, kind in (('curve_parking', None), ('static_obstacle', 'static_vehicle'), ('dynamic_interaction', 'crossing_walker')):
            ids = [i for i, c in enumerate(cases) if (c['hazard'] is None if kind is None else
                   c['hazard'] is not None and c['hazard']['kind'] == kind)]
            if len(ids) * 3 != len(cases): raise ValueError('Require balanced core families')
            families[label] = {m: sum(rows[i]['success'] for i in ids) for m, rows in (('H', left), ('B2', right))}
        counts = {m: sum(r['success'] for r in rows.values()) for m, rows in (('H', left), ('B2', right))}
        joint = [i for i in left if left[i]['success'] and right[i]['success']]
        operating = {}
        for method, rows in (('H', left), ('B2', right)):
            operating[method] = {
                'termination_counts': dict(Counter(r['termination_reason'] for r in rows.values())),
                'all_episode_totals': {key: sum(r[key] for r in rows.values()) for key in
                                      ('physics_steps', 'simulation_seconds', 'progress', 'interventions', 'plan_failures', 'mpc_failures')},
                'jointly_successful_cases': joint,
                'joint_success_simulation_seconds': sum(rows[i]['simulation_seconds'] for i in joint),
                'episodes': [{k: r[k] for k in ('case_id', 'success', 'termination_reason', 'physics_steps',
                             'simulation_seconds', 'progress', 'interventions', 'plan_failures', 'mpc_failures',
                             'control_cycle_times', 'decision_cycle_times')} for r in rows.values()]}
        results.append({'seed': pair['seed'], 'cases': len(cases), 'successes': counts, 'families': families,
                        'operating_metrics': operating,
                        'unsafe_episodes': {m: sum(r['termination_reason'] in ('collision', 'out_of_bounds') for r in rows.values())
                                            for m, rows in (('H', left), ('B2', right))},
                        'difference': (counts['H'] - counts['B2']) / len(cases), 'proof': [hp, bp],
                        'walker_bridge_source_equal': sha(h / 'source/scripts/carla_bridge37.py') == sha(b2 / 'source/scripts/carla_bridge37.py')})
    result = {'scope': __doc__, 'pairs_manifest_sha256': sha(pairs_manifest),
              'B2_execution_state_sha256': sha(execution / 'status.json'), 'results': results,
              'successes': {m: sum(r['successes'][m] for r in results) for m in ('H', 'B2')},
              'episodes_each_method': sum(r['cases'] for r in results), 'unique_known_conditions': len(inventory),
              'statistics': paired_seed_statistics([r['difference'] for r in results]),
              'limits': 'Retrospective DEVELOPMENT comparison with shared H teacher priors;historical tuning costs not equalized. Same12known conditions repeated across five independently trained policies. No fresh final confirmation or PPO-vs-rule/feedback necessity claim.',
              'new_controls': 0, 'optimizer_updates': 0, 'research_acceptance': 'UNMET'}
    with output.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('successes', 'statistics')})); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs-manifest', default=str(ROOT / 'protocols/native_feedback_development_pairs_20261031.json'))
    parser.add_argument('--execution', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.pairs_manifest, args.execution, args.output)

"""Compare complete known H/H0 inventories with the shared-lowlevel B3 rule."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.fixed_shared_native_rule import without_functions
from scripts.summarize_outcome_development import summarize as summarize_pair
from navstack.shared_rule_registration import registered
from navstack.metrics import timing_summary

SHARED_PROTOCOL_KEYS = ('routes', 'native_case_manifest', 'native_validation', 'calibration_sha256',
    'reference_road_refinement', 'reference_steering_transition', 'reference_option_commitment',
    'reference_compute_profile', 'reference_joint_speed', 'reference_point_constraints',
    'mpc_reference_effort', 'goal_parking_assist', 'reference_context', 'shared_endpoint_mask_v1',
    'shared_mask_ablation', 'initialization_gate_profile', 'unsafe_event_penalty', 'reward_intervention')


def recorded_costs(trace, digest, period, selection_key):
    if sha(trace) != digest:
        raise ValueError('Timing trace differs from its completed raw audit')
    selection, full = [], []
    search = dict(search_calls=0, search_nodes=0, search_seconds=0.)
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            a, b = row[selection_key], row['full_decision_seconds']
            if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a <= b:
                raise ValueError('Invalid recorded full-cycle timing')
            selection.append(a); full.append(b)
            for key, value in row['info'].get('rule_candidate_compute', {}).items():
                search[key] += value
    return dict(decisions=len(full), selection_seconds=sum(selection),
        full_decision_seconds=sum(full), selection_times=timing_summary(selection, period),
        full_decision_times=timing_summary(full, period), rule_extra_nominal_search=search,
        scope='Measured policy selection plus environment step;excludes reset,serialization and process setup')


def paired_cases(left, right):
    """Descriptive pairing only: these conditions are not independent seeds."""
    if [e['case_id'] for e in left] != [e['case_id'] for e in right]:
        raise ValueError('Complete ordered condition inventories must match')
    return dict(
        both_success=sum(a['success'] and b['success'] for a, b in zip(left, right)),
        learned_only_success=sum(a['success'] and not b['success'] for a, b in zip(left, right)),
        rule_only_success=sum(b['success'] and not a['success'] for a, b in zip(left, right)),
        both_failed=sum(not a['success'] and not b['success'] for a, b in zip(left, right)),
        learned_minus_rule_controls=sum(a['physics_steps']-b['physics_steps'] for a, b in zip(left, right)),
        learned_minus_rule_interventions=sum(a['interventions']-b['interventions'] for a, b in zip(left, right)),
        learned_minus_rule_planning_failures=sum(a['plan_failures']-b['plan_failures'] for a, b in zip(left, right)))


def summarize(h_execution, rule_execution):
    h_execution, rule_execution = Path(h_execution).resolve(), Path(rule_execution).resolve()
    archive = rule_execution.parent
    if h_execution.parent != archive:
        raise ValueError('Use the same explicit evidence archive')
    learned = summarize_pair(h_execution)
    done = read(rule_execution/'complete.json')
    if len(done['results']) != 2 or {r['phase'] for r in done['results']} != {'placements30', 'known24'}:
        raise ValueError('Both complete prespecified rule inventories are required')
    rule, paired = {}, {}
    for item in done['results']:
        phase = item['phase']; count = 30 if phase == 'placements30' else 24
        batch = archive/item['batch']
        if batch.resolve().parent != archive:
            raise ValueError('Batch must remain in the registered archive')
        require_finished_parent(batch)
        p, raw, engine, result = [read(batch/name) for name in
            ('protocol.json', 'raw_audit.json', 'strict_delivery.json', 'summary.json')]
        if (not registered(p) or raw['protocol_sha256'] != sha(batch/'protocol.json')
                or raw['summary_sha256'] != sha(batch/'summary.json')
                or item['raw_audit_sha256'] != sha(batch/'raw_audit.json')
                or item['engine_sha256'] != sha(batch/'strict_delivery.json')
                or not raw['batch_complete'] or raw['cases_verified'] != count
                or not engine['strict_delivery_gate_passed']
                or raw['evaluation_controls_verified'] != engine['counts']['controls']
                or item['actual_controls'] != raw['evaluation_controls_verified']):
            raise ValueError('Complete bound rule raw/control proof required')
        eps = result['episodes']
        if [e['case_id'] for e in eps] != list(range(count)):
            raise ValueError('Preserve all rule outcomes in registered order')
        for name, digest in p['source_hashes'].items():
            if sha(batch/'source'/name) != digest:
                raise ValueError('Frozen rule source changed: '+name)
        if sha(batch/'calibration/calibration.json') != p['calibration_sha256']:
            raise ValueError('Rule calibration changed')
        families = {}
        manifest = p['native_case_manifest'] if phase == 'placements30' else p['native_validation']['manifest']
        for case, ep in zip(manifest['cases'], eps):
            family = {None: 'curve_parking', 'static_vehicle': 'static_obstacle',
                'crossing_walker': 'dynamic_interaction'}[(case.get('hazard') or {}).get('kind')]
            stats = families.setdefault(family, dict(cases=0, successes=0, unsafe=0))
            stats['cases'] += 1; stats['successes'] += ep['success']
            stats['unsafe'] += ep['termination_reason'] in ('collision', 'out_of_bounds')
        rule[phase] = dict(batch=batch.name, cases=count, successes=sum(e['success'] for e in eps),
            termination_counts=dict(Counter(e['termination_reason'] for e in eps)), families=families,
            actual_controls=raw['evaluation_controls_verified'],
            interventions=sum(e['interventions'] for e in eps),
            plan_failures=sum(e['plan_failures'] for e in eps),
            shared_parking_controls=sum(e['goal_parking_controls'] for e in eps),
            world_ticks=result['world_ticks'], reset_ticks=result['reset_ticks'],
            recorded_evaluation_wall_seconds=result['wall_seconds'], neural_parameters=0,
            recorded_costs=recorded_costs(batch/'evaluation_trace.jsonl.gz', raw['trace_sha256'],
                p['config']['decision_dt'], 'selection_seconds'),
            raw_audit_sha256=sha(batch/'raw_audit.json'), all_cases=eps)
        paired[phase] = {}
        for method in ('H', 'H0'):
            hbatch = archive/learned['results'][method+'_'+phase]['batch']
            hp = read(hbatch/'protocol.json'); hev = read(hbatch/'H/evaluation.json')
            for key in SHARED_PROTOCOL_KEYS:
                if hp.get(key) != p.get(key):
                    raise ValueError('Paired tasks or calibration differ: '+key)
            if p['config'] != {**read(hbatch/'H/config.json')['env'], 'method': 'B3'}:
                raise ValueError('Rule environment differs beyond method label')
            exceptions = p['fixed_shared_rule']['unchanged_lowlevel_AST_except']
            for name, digest in hp['source_hashes'].items():
                if name.startswith('navstack/'):
                    if name in exceptions:
                        if without_functions(hbatch/'source'/name, exceptions[name]) != without_functions(batch/'source'/name, exceptions[name]):
                            raise ValueError('Shared lowlevel AST differs: '+name)
                    elif p['source_hashes'].get(name) != digest:
                        raise ValueError('Shared native module differs: '+name)
            paired[phase][method] = paired_cases(hev['episodes'], eps)
            # Preserve full policy-including cycles, rather than comparing an
            # environment-only learned cycle with a policy-including rule cycle.
            learned['results'][method+'_'+phase]['policy_including_case_timings'] = [
                dict(case_id=e['case_id'], full_decision_times=e['full_decision_times'],
                     policy_prediction_times=e['policy_prediction_times']) for e in hev['episodes']]
            hraw = read(hbatch/'H/endpoint_raw_audit.json')
            learned['results'][method+'_'+phase]['recorded_costs'] = recorded_costs(
                hbatch/'H/evaluation_trace.jsonl.gz', hraw['evaluation_trace_sha256'],
                p['config']['decision_dt'], 'policy_prediction_seconds')
            learned['results'][method+'_'+phase]['neural_parameters'] = read(hbatch/'H/policy_architecture.json')['parameters']
    return dict(scope=__doc__, research_acceptance='UNMET', independent_learned_training_seeds=1,
        registered_known_entries_per_method=54, new_final_conditions=0, new_PPO_updates=0,
        learned=learned, rule=rule, descriptive_pairs=paired,
        rule_actual_evaluation_controls=sum(v['actual_controls'] for v in rule.values()),
        execution_complete_sha256=sha(rule_execution/'complete.json'), script_sha256=sha(Path(__file__)),
        limits='Known development conditions and one continued learned seed;no independent-seed confidence interval or final confirmation.Rule compute includes candidate scoring and nominal searches;full_decision_times includes policy selection plus environment step for both methods.Shared workstation timings are not isolated real-time benchmarks.All failures retained.No PPO superiority follows merely from completed execution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('h-execution', 'rule-execution', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve completed comparison reports')
    atomic_json(output, summarize(args.h_execution, args.rule_execution))

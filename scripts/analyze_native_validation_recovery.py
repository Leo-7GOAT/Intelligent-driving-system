"""Join all24fixed validation conditions across a preserved initialization failure and its explicit remainder."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha
from scripts.audit_native_control_delivery import analyze as delivery_audit
from scripts.analyze_public_native_replay import parking_trace
from scripts.evaluate_native_rule import require_finished_parent


def analyze(parent, child, prefix_path):
    parent, child, prefix_path = Path(parent).resolve(), Path(child).resolve(), Path(prefix_path).resolve()
    require_finished_parent(child)
    prefix = read(prefix_path)
    p, cp = read(parent / 'protocol.json'), read(child / 'protocol.json')
    spec = cp['validation_remainder']
    raw = read(child / 'B1/endpoint_raw_audit.json')
    summary = read(child / 'B1/summary.json')
    evaluation = read(child / 'B1/evaluation.json')
    starts = read(child / 'B1/measured_start_inference.json')
    original_ids = spec['original_case_ids']
    if (prefix['protocol_sha256'] != sha(parent / 'protocol.json')
            or spec['parent_protocol_sha256'] != prefix['protocol_sha256']
            or prefix['remaining_case_ids'] != original_ids
            or raw['protocol_sha256'] != sha(child / 'protocol.json')
            or summary['physics_steps'] != 0 or summary['optimizer_steps'] != 0
            or sorted(e['case_id'] for e in evaluation['episodes']) != list(range(len(original_ids)))
            or sorted(e['case_id'] for e in starts['results']) != list(range(len(original_ids)))
            or starts['batch_protocol_sha256'] != sha(child / 'protocol.json')):
        raise ValueError('Require the exact retained prefix and every predeclared unfinished case')
    for name, digest in cp['source_hashes'].items():
        if sha(child / 'source' / name) != digest:
            raise ValueError('Frozen remainder source changed')
    for name in ('model.zip', 'normalization.pkl'):
        if sha(parent / 'B1' / name) != sha(child / 'B1' / name):
            raise ValueError('Frozen checkpoint changed between initialization versions')
    if not delivery_audit(child)['strict_delivery_gate_passed']:
        raise ValueError('Actual engine acknowledgements must all match')
    parking = parking_trace(child, list(range(len(original_ids))))
    episodes = {r['case_id']: {**r['metrics'], 'case_id': r['case_id']} for r in prefix['completed_cases']}
    for e in evaluation['episodes']:
        original = original_ids[e['case_id']]
        if original in episodes:
            raise ValueError('A completed condition was replayed or counted twice')
        episodes[original] = {**e, 'case_id': original}
    if sorted(episodes) != list(range(24)):
        raise ValueError('All24original development-validation conditions are required')
    groups = {}
    for case, e in sorted(episodes.items()):
        hazard = p['native_case_manifest']['cases'][case]['hazard']
        name = 'curve_parking' if hazard is None else hazard['kind']
        groups.setdefault(name, []).append(e)
    count = prefix['all_recorded_actual_controls_verified'] + raw['total_applied_controls_verified']
    return {'scope': __doc__, 'script_sha256': sha(Path(__file__)),
            'parent_batch': parent.name, 'remainder_batch': child.name,
            'parent_protocol_sha256': sha(parent / 'protocol.json'), 'remainder_protocol_sha256': sha(child / 'protocol.json'),
            'prefix_audit_sha256': sha(prefix_path), 'remainder_raw_audit_sha256': sha(child / 'B1/endpoint_raw_audit.json'),
            'model_sha256': evaluation['model_sha256'], 'normalization_sha256': evaluation['normalization_sha256'],
            'independently_trained_policies': 1, 'unique_existing_validation_conditions': 24,
            'successes': sum(e['success'] for e in episodes.values()),
            'termination_counts': dict(Counter(e['termination_reason'] for e in episodes.values())),
            'by_core': {name: {'cases': len(rows), 'successes': sum(e['success'] for e in rows),
                'termination_counts': dict(Counter(e['termination_reason'] for e in rows))} for name, rows in groups.items()},
            'actual_controls_verified': count, 'remainder_measured_start_actions_verified': len(starts['results']),
            'all_engine_controls_acknowledged': True, 'remainder_parking': parking,
            'startup_initialization_mixed': True,
            'startup_versions': {'completed_original_case_ids': prefix['completed_case_ids'],
                'resumed_original_case_ids': original_ids, 'resumed_walker_initialization': 'native_walker_clearance_020_v1',
                'retained_legacy_walker_case_ids': [16]},
            'case_outcomes': [{k: e[k] for k in ('case_id', 'success', 'termination_reason', 'physics_steps',
                'interventions', 'goal_parking_controls', 'simulation_seconds', 'progress',
                'travel_time_success_only', 'decision_cycle_times', 'control_cycle_times')} for _, e in sorted(episodes.items())],
            'new_PPO_updates': 0, 'new_final_conditions': 0,
            'limits': 'One policy,known developmentvalidation. Complete descriptive recovery record with an explicit pre-control walker-height change in seven cases;not a uniform-harness formal comparison or new final confirmation. Earlier completed failures are all retained.',
            'research_acceptance': 'UNMET'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--child', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.parent, args.child, args.prefix)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('successes', 'termination_counts', 'by_core', 'actual_controls_verified')}))

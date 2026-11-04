"""Audit every known curve, static-obstacle and walker condition of the trained B1 probe."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha
from scripts.audit_native_control_delivery import analyze as audit_delivery
from scripts.analyze_public_native_replay import parking_trace
from scripts.evaluate_native_rule import require_finished_parent


def analyze(batch):
    batch = Path(batch).resolve()
    require_finished_parent(batch)
    p = read(batch / 'protocol.json')
    spec = p['qualified_basic_full_probe']
    raw = read(batch / 'B1/endpoint_raw_audit.json')
    summary = read(batch / 'B1/summary.json')
    evaluation = read(batch / 'B1/evaluation.json')
    inference = read(batch / 'B1/measured_start_inference.json')
    episodes = evaluation['episodes']
    if (not p['evaluation_only'] or summary['physics_steps'] != 0 or summary['optimizer_steps'] != 0
            or raw['training_control_frames_verified'] != 0 or raw['protocol_sha256'] != sha(batch / 'protocol.json')
            or sorted(e['case_id'] for e in episodes) != list(range(12))
            or sorted(r['case_id'] for r in inference['results']) != list(range(12))
            or inference['batch_protocol_sha256'] != sha(batch / 'protocol.json')):
        raise ValueError('Require all twelve registered frozen-policy outcomes and measured starts')
    for name, digest in p['source_hashes'].items():
        if sha(batch / 'source' / name) != digest:
            raise ValueError('Frozen source changed')
    for name, key in (('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256')):
        if sha(batch / 'B1' / name) != sha(batch / 'resume/B1' / name) or sha(batch / 'B1' / name) != evaluation[key]:
            raise ValueError('Frozen learned checkpoint changed')
    if evaluation['model_sha256'] != spec['source_model_sha256'] or inference['model_sha256'] != evaluation['model_sha256']:
        raise ValueError('Require the predeclared trained endpoint')
    delivery = audit_delivery(batch)
    if not delivery['strict_delivery_gate_passed']:
        raise ValueError('Every engine acknowledgement is required')
    parking = parking_trace(batch, list(range(12)))
    if sum(e['controls'] for e in parking['cases']) != raw['total_applied_controls_verified']:
        raise ValueError('Assistance audit must cover every actual control')
    groups = {}
    for e in episodes:
        hazard = p['native_case_manifest']['cases'][e['case_id']]['hazard']
        group = 'curve_parking' if hazard is None else hazard['kind']
        groups.setdefault(group, []).append(e)
    by_core = {name: {'cases': len(rows), 'successes': sum(e['success'] for e in rows),
                     'termination_counts': dict(Counter(e['termination_reason'] for e in rows))}
               for name, rows in groups.items()}
    return {'scope': __doc__, 'script_sha256': sha(Path(__file__)), 'batch': batch.name,
            'protocol_sha256': sha(batch / 'protocol.json'), 'model_sha256': evaluation['model_sha256'],
            'normalization_sha256': evaluation['normalization_sha256'], 'seed': p['seed'],
            'cases': 12, 'successes': sum(e['success'] for e in episodes),
            'termination_counts': dict(Counter(e['termination_reason'] for e in episodes)), 'by_core': by_core,
            'new_actual_controls': raw['total_applied_controls_verified'], 'new_PPO_updates': 0,
            'all_engine_controls_acknowledged': True, 'measured_start_actions_verified': 12,
            'parking': parking, 'lineage_PPO_controls': summary['lineage_control_steps'],
            'raw_audit_sha256': sha(batch / 'B1/endpoint_raw_audit.json'),
            'case_metrics': [{key: e[key] for key in ('case_id', 'success', 'termination_reason', 'physics_steps',
                'interventions', 'goal_parking_controls', 'simulation_seconds', 'progress',
                'travel_time_success_only', 'decision_cycle_times', 'control_cycle_times')} for e in episodes],
            'limits': 'One independently trained warm-started seed, twelve KNOWN development conditions. No full-task PPO updates here. Whole-trajectory policy-action reconstruction is not claimed for the dynamic cases. Not final confirmation, five-seed qualification, or an isolated PPO benefit.',
            'research_acceptance': 'UNMET'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.batch)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in ('successes', 'termination_counts', 'by_core', 'new_actual_controls')}), flush=True)

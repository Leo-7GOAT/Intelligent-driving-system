"""Descriptive complete-inventory H/B3 development comparison; no new driving."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha


def analyze(h, b3, output):
    h, b3, output = map(Path, (h, b3, output))
    if output.exists():
        raise FileExistsError('Preserve previous comparison')
    hp, bp = read(h/'protocol.json'), read(b3/'protocol.json')
    config = read(h/'H/config.json')['env']
    if config != bp['config'] or any(hp.get(k) != bp.get(k) for k in
            ('routes', 'native_case_manifest', 'native_validation', 'calibration_sha256')):
        raise ValueError('Shared environment or registered condition inventory differs')
    common_sources = {}
    for name in ('navstack/env.py', 'navstack/config.py', 'navstack/planning.py',
                 'navstack/mpc.py', 'navstack/trajectory.py', 'navstack/shield.py',
                 'navstack/carla_bridge.py', 'scripts/carla_bridge37.py'):
        digest = sha(h/'source'/name)
        if (digest != sha(b3/'source'/name) or digest != hp['source_hashes'][name]
                or digest != bp['source_hashes'][name]):
            raise ValueError('Registered shared stack source differs: '+name)
        common_sources[name] = digest
    inventory = hp.get('native_validation', {}).get('manifest', hp.get('native_case_manifest'))
    if inventory is None:
        raise ValueError('Require an explicit complete development inventory')
    expected = set(range(len(inventory['cases'])))
    ha, ba = read(h/'H/endpoint_raw_audit.json'), read(b3/'raw_audit.json')
    for base, audit, bindings in ((h, ha, [('protocol.json', 'protocol_sha256'),
            ('H/evaluation.json', 'evaluation_sha256'), ('H/evaluation_trace.jsonl.gz', 'evaluation_trace_sha256')]),
            (b3, ba, [('protocol.json', 'protocol_sha256'), ('summary.json', 'summary_sha256'),
            ('evaluation_trace.jsonl.gz', 'trace_sha256')])):
        for name, key in bindings:
            if sha(base/name) != audit[key]:
                raise ValueError('Completed raw audit binding changed: '+name)
    rows = [read(h/'H/evaluation.json')['episodes'], read(b3/'summary.json')['episodes']]
    for episodes in rows:
        if len(episodes) != len(expected) or {r['case_id'] for r in episodes} != expected:
            raise ValueError('Require every condition exactly once, including failures')
    if sum(r['physics_steps'] for r in rows[0]) != ha['evaluation_control_frames_verified'] or sum(
            r['physics_steps'] for r in rows[1]) != ba['evaluation_controls_verified']:
        raise ValueError('Episode controls differ from raw audit')
    aggregates = {}
    for method, episodes in zip(('H', 'B3'), rows):
        aggregates[method] = {
            'successes': sum(r['success'] for r in episodes), 'cases': len(episodes),
            'termination_counts': dict(Counter(r['termination_reason'] for r in episodes)),
            **{name: sum(r[name] for r in episodes) for name in
               ('physics_steps', 'interventions', 'plan_failures', 'mpc_failures', 'progress')},
            'simulation_seconds_all_outcomes': sum(r['physics_steps'] for r in episodes)*config['control_dt'],
            'success_only_travel_seconds': [r['travel_time_success_only'] for r in episodes if r['success']]}
    indexed = [{r['case_id']: r for r in episodes} for episodes in rows]
    pairs = [{'case_id': case, 'H_success': indexed[0][case]['success'],
              'B3_success': indexed[1][case]['success'],
              'H_minus_B3_controls': indexed[0][case]['physics_steps']-indexed[1][case]['physics_steps'],
              'H_minus_B3_interventions': indexed[0][case]['interventions']-indexed[1][case]['interventions']}
             for case in sorted(expected)]
    result = {'scope': __doc__, 'H_batch': h.name, 'B3_batch': b3.name,
              'script_sha256': sha(Path(__file__)),
              'H_raw_audit_sha256': sha(h/'H/endpoint_raw_audit.json'),
              'B3_raw_audit_sha256': sha(b3/'raw_audit.json'),
              'identical_resolved_environment': True, 'common_stack_sha256': common_sources,
              'aggregates': aggregates, 'pairs': pairs,
              'limits': 'One trained H seed on reused development conditions. B3 has no learned seed. No independent-seed interval or superiority claim. All-outcome duration includes early failures; success-only times are explicitly separate. No wall-clock or real-time superiority inference.',
              'new_controls': 0, 'optimizer_updates': 0, 'research_acceptance': 'UNMET'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(aggregates))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('h', 'b3', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    analyze(args.h, args.b3, args.output)

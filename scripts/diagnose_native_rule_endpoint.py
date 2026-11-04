"""Pair an audited rule run with H and probe recorded failed-rule states offline."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import write_json
from scripts.analyze_native_gradient_pair import initial_cases
from scripts.analyze_native_rollout_pair import checked_endpoint
from scripts.audit_native_action_availability import state_references, verify_source
from scripts.audit_native_rule import audit, read, sha


def pair_tasks(learned, rule):
    if not learned or set(learned) != set(rule):
        raise ValueError('Incomplete paired task inventory')
    pairs = []
    for case in sorted(learned):
        a, b = learned[case], rule[case]
        if {k: v for k, v in a.items() if k != 'initial'} != {k: v for k, v in b.items() if k != 'initial'}:
            raise ValueError('Measured road or task differs')
        x, y = a['initial'], b['initial']
        pairs.append({'case_id': case, 'road_sha256': a['road_sha256'],
            'measured_start_position_difference_m': math.hypot(x['x']-y['x'], x['y']-y['y']),
            'measured_start_yaw_difference_rad': (y['yaw']-x['yaw']+math.pi) % (2*math.pi)-math.pi,
            'measured_start_speed_difference_mps': y['v']-x['v']})
    return pairs


def analyze(batch, parent, output):
    batch, parent, output = Path(batch), Path(parent), Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = read(batch/'protocol.json')
    if protocol['parent_batch'] != parent.name or protocol['parent_protocol_sha256'] != sha(parent/'protocol.json'):
        raise ValueError('Wrong parent experiment')
    if audit(batch) != read(batch/'raw_audit.json'):
        raise ValueError('Rule raw audit is stale')
    name = protocol['source_method']
    old = read(parent/'protocol.json')
    job = next(row for row in old['jobs'] if row['name'] == name)
    if job['method'] != 'H':
        raise ValueError('Comparison requires an H rollout endpoint')
    learned = checked_endpoint(parent, name, old, job['rollout_actions'])
    if read(parent/name/'config.json')['env'] != protocol['config']:
        raise ValueError('Rule and H environment configurations differ')
    for source, digest in protocol['source_hashes'].items():
        if source.startswith('navstack/'):
            if old['source_hashes'][source] != digest:
                raise ValueError('Rule and H production sources differ')
            verify_source(ROOT/source, batch/'source'/source, digest)
    paired = pair_tasks(initial_cases(parent/name), initial_cases(batch))
    summary = read(batch/'summary.json')
    config = EnvConfig.from_dict(protocol['config'])
    with gzip.open(batch/'evaluation_trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    results = []
    for metrics in summary['episodes']:
        case = metrics['case_id']
        decisions = [r for r in rows if r['case_id'] == case]
        scene = read(batch/f'scene_{case:04d}.json')['scene']
        state = decisions[0]['info']['initial_scenario']['initial']
        first_wait, before_last = None, None
        for index, row in enumerate(decisions):
            before_last = state
            if first_wait is None and row['policy_action'] == 0:
                first_wait = (index, state)
            state = row['info']['trace'][-1]['state']
        probes = []
        if not metrics['success']:
            selected = [('before_last_decision', len(decisions)-1, before_last)]
            if first_wait is not None:
                selected.insert(0, ('before_first_wait', *first_wait))
            for stage, index, pose in selected:
                result = state_references(config, scene, pose)
                action = int(decisions[index]['policy_action'])
                if result['actions'][action]['mode'] != decisions[index]['info']['mode']:
                    raise ValueError('Recorded selected reference mode does not reproduce')
                probes.append({'stage': stage, 'decision': index, 'recorded_action': action, **result})
        h = next(r for r in learned['cases'] if r['case_id'] == case)
        results.append({'case_id': case, 'H_success': h['success'], 'rule_metrics': metrics,
            'mode_counts': dict(Counter(r['info']['mode'] for r in decisions)),
            'first_wait_decision': None if first_wait is None else first_wait[0],
            'final_state': state, 'final_goal_distance_m': math.hypot(state['x']-scene['goal'][0], state['y']-scene['goal'][1]),
            'final_goal_yaw_error_rad': abs((state['yaw']-scene['goal'][2]+math.pi) % (2*math.pi)-math.pi),
            'failed_case_reference_probes': probes})
    result = {'scope': 'Same measured training-road tasks and unchanged low level; rule success is not PPO success. No seed interval. Offline search failures are not global infeasibility proofs, and alternative references are not executed outcomes.',
        'batch': batch.name, 'parent_batch': parent.name, 'H_method': name,
        'rule_raw_audit_sha256': sha(batch/'raw_audit.json'),
        'H_raw_audit_sha256': sha(parent/name/'endpoint_raw_audit.json'),
        'script_sha256': sha(Path(__file__)), 'availability_script_sha256': sha(ROOT/'scripts/audit_native_action_availability.py'),
        'paired_tasks': paired, 'cases': results, 'new_controls': 0, 'optimizer_updates': 0}
    write_json(output, result)
    print(json.dumps({'cases': len(results), 'rule_successes': sum(r['rule_metrics']['success'] for r in results),
                      'H_successes': sum(r['H_success'] for r in results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.batch, args.parent, args.output)

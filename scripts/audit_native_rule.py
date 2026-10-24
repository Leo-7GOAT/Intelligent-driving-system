"""Bind a complete native rule diagnostic to measured CARLA frames, not PPO costs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha, read_prefix, _verify_ticks
from navstack.metrics import timing_summary


def audit(batch):
    batch = Path(batch)
    protocol, summary = read(batch/'protocol.json'), read(batch/'summary.json')
    if summary['status'] != 'complete' or summary['policy_kind'] != 'rule_only_B3':
        raise ValueError('Requires a completed explicitly labelled rule diagnostic')
    if any(record.get(key) != 0 for record in (protocol, summary)
           for key in ('training_physics_steps', 'optimizer_updates')):
        raise ValueError('Rule evaluation cannot claim PPO training or optimizer updates')
    for name, digest in protocol['source_hashes'].items():
        if sha(batch/'source'/name) != digest:
            raise ValueError('Frozen source changed')
    if (sha(batch/'calibration/calibration.json') != protocol['calibration_sha256']
            or sha(batch/'source_config.json') != protocol['source_config_sha256']
            or read(batch/'source_config.json')['env'] != protocol['config']):
        raise ValueError('Calibration or source configuration changed')
    paths = sorted((batch/'infrastructure').glob('*_responses.jsonl.gz'))
    if len(paths) != 1:
        raise ValueError('Expected exactly one bridge stream with no automatic retry')
    wrappers, error = read_prefix(paths[0])
    if error:
        raise ValueError('Incomplete bridge response stream: '+error)
    actual, acknowledgements, owners, resets = {}, {}, {}, {}
    for wrapper in wrappers:
        response = wrapper.get('response', {})
        episode, request = response.get('episode_id'), response.get('request_id')
        if episode is not None and request is not None:
            key = (episode, request)
            if key in acknowledgements and acknowledgements[key] != response:
                raise ValueError('Conflicting bridge acknowledgement')
            acknowledgements[key] = response
        if episode is not None and 'reset_ticks' in response:
            if episode in resets and resets[episode] != response['reset_ticks']:
                raise ValueError('Conflicting reset evidence')
            resets[episode] = response['reset_ticks']
        if 'vehicle_control' in response:
            key = (episode, response['frame'])
            if key in actual:
                raise ValueError('Duplicate applied bridge frame')
            actual[key], owners[key] = response, paths[0].name
    rows, error = read_prefix(batch/'evaluation_trace.jsonl.gz')
    if error:
        raise ValueError('Incomplete rule trace: '+error)
    cases, used, previous, episodes = {}, set(), {}, set()
    for row in rows:
        if row['policy_kind'] != 'rule_only_B3':
            raise ValueError('Trace mislabels rule selection')
        info = row['info']
        case = cases.setdefault(row['case_id'], {'actions': 0, 'controls': 0, 'terminal': None,
                                                'initial': info['initial_scenario'], 'episodes': set(),
                                                'selection_times': [], 'full_times': []})
        if case['terminal'] is not None or case['initial'] != info['initial_scenario']:
            raise ValueError('Case changed or contains controls after termination')
        if not 0 <= int(row['policy_action']) < 31 or int(row['policy_action']) != row['policy_action']:
            raise ValueError('Rule selected outside the shared action catalog')
        if protocol.get('rule_speed_policy', 'native_score') == 'paired_low_speed':
            proposal = row.get('rule_proposal')
            if type(proposal) is not int or not 0 <= proposal < 31:
                raise ValueError('Missing valid original rule proposal for speed ablation')
            expected_action = 0 if proposal == 0 else 2*((proposal-1)//2)+1
            if row.get('rule_speed_policy') != 'paired_low_speed' or row['policy_action'] != expected_action:
                raise ValueError('Recorded action differs from registered lower-speed selection')
        if (not all(math.isfinite(row[k]) for k in ('selection_seconds', 'full_decision_seconds'))
                or row['selection_seconds'] < 0 or row['full_decision_seconds'] < row['selection_seconds']):
            raise ValueError('Invalid rule compute accounting')
        controls = _verify_ticks(info['trace'], actual, acknowledgements, owners, used, previous, episodes)
        if controls != info['executed_substeps']:
            raise ValueError('Substep count differs from raw trace')
        case['episodes'].update(t['backend_evidence']['episode_id'] for t in info['trace'])
        case['actions'] += 1
        case['controls'] += controls
        case['selection_times'].append(row['selection_seconds'])
        case['full_times'].append(row['full_decision_seconds'])
        if info['termination_reason']:
            case['terminal'] = info['episode_metrics']
    expected = {r['case_id']: r for r in summary['episodes']}
    inventory = set(range(len(protocol['routes'])))
    if len(expected) != len(summary['episodes']) or set(expected) != inventory or set(cases) != inventory:
        raise ValueError('Incomplete or duplicated registered case inventory')
    scene_hashes = {}
    for case_id in sorted(cases):
        case, recorded = cases[case_id], expected[case_id]
        metrics = {k: v for k, v in recorded.items()
                   if k not in ('case_id', 'rule_selection_times', 'full_decision_times')}
        if case['terminal'] != metrics or case['controls'] != metrics['physics_steps'] or case['actions'] != metrics['actions']:
            raise ValueError('Terminal metrics do not match the complete trace')
        if len(case['episodes']) != 1:
            raise ValueError('One case must have exactly one measured episode')
        for key, values in [('rule_selection_times', case['selection_times']), ('full_decision_times', case['full_times'])]:
            if recorded[key] != timing_summary(values, protocol['config']['decision_dt']):
                raise ValueError('Recorded timing summary differs from per-decision costs')
        path = batch/f'scene_{case_id:04d}.json'
        scene = read(path)
        if (scene['initial'] != case['initial']
                or scene['scene']['native_layout']['geometry_id'] != protocol['routes'][case_id]['geometry_id']
                or scene['scene']['road_grid']['sha256'] != case['initial']['road_sha256']):
            raise ValueError('Scene identity differs from the registered route or recorded task')
        scene_hashes[path.name] = sha(path)
    if not actual or used != set(actual) or set(resets) != episodes or len(episodes) != len(inventory):
        raise ValueError('Unaccounted applied controls or resets')
    reset_ticks = sum(resets.values())
    if (summary['evaluation_controls'] != len(used) or summary['reset_ticks'] != reset_ticks
            or summary['world_ticks'] != len(used)+reset_ticks):
        raise ValueError('World/reset/control accounting mismatch')
    return {'scope': 'Raw frame verification of rule-only native evaluation; no learned-policy or training claim',
            'batch': batch.name, 'protocol_sha256': sha(batch/'protocol.json'),
            'summary_sha256': sha(batch/'summary.json'), 'trace_sha256': sha(batch/'evaluation_trace.jsonl.gz'),
            'bridge_response_sha256': sha(paths[0]), 'scene_hashes': scene_hashes,
            'evaluation_controls_verified': len(used), 'reset_ticks_verified': reset_ticks,
            'world_ticks_verified': len(used)+reset_ticks, 'cases_verified': len(cases),
            'termination_counts': dict(Counter(r['termination_reason'] for r in summary['episodes'])),
            'training_physics_steps': 0, 'optimizer_updates': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.batch)
    output = args.output or args.batch/'raw_audit.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))

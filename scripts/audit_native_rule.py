"""Bind a complete native rule diagnostic to measured CARLA frames, not PPO costs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha, read_prefix, _verify_ticks, verify_initial_actuator
from navstack.metrics import timing_summary
from navstack.native_cases import registered_native_cases, verify_registered_native_begin


def audit(batch, allow_completed_prefix=False):
    batch = Path(batch)
    protocol = read(batch/'protocol.json')
    registered_cases = registered_native_cases(protocol,'evaluation') if protocol.get('native_validation') else registered_native_cases(protocol)
    partial = allow_completed_prefix and not (batch/'summary.json').exists()
    if partial:
        failure = read(batch/'failure.json')
        guard = read(batch/'guard_final.json')
        cleanup = read(batch/'infrastructure/server_stop.json')
        if (not failure.get('error') or guard.get('exit_code') != 1
                or cleanup.get('status') != 'owned_empty_server_stopped'):
            raise ValueError('Partial audit requires a recorded failure and normal owned cleanup')
        results = [read(p) for p in sorted(batch.glob('case_*.json'))]
        ids = [r['case_id'] for r in results]
        if not ids or ids != list(range(len(ids))) or len(ids) >= len(registered_cases):
            raise ValueError('Partial audit requires a strict completed case prefix')
        summary = {'status': 'complete', 'policy_kind': 'rule_only_B3', 'episodes': results,
                   'training_physics_steps': 0, 'optimizer_updates': 0}
    else:
        summary = read(batch/'summary.json')
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
    if partial:
        errors = [w['response'] for w in wrappers if w.get('response', {}).get('ok') is False]
        if (len(errors) != 1 or wrappers[-1].get('response') != errors[0]
                or not errors[0].get('infrastructure_failure')):
            raise ValueError('Partial audit requires exactly one terminal infrastructure error')
        summary.update(evaluation_controls=len(actual), reset_ticks=sum(resets.values()),
                       world_ticks=len(actual)+sum(resets.values()))
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
        actuator_observation = protocol['config'].get('actuator_observation', 'off')
        controls = _verify_ticks(info['trace'], actual, acknowledgements, owners, used, previous, episodes, actuator_observation)
        if actuator_observation != 'off':
            verify_initial_actuator(info['initial_scenario'], info['trace'][0]['backend_evidence']['episode_id'], acknowledgements)
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
    inventory = set(range(len(summary['episodes']) if partial else len(registered_cases)))
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
                or scene['scene']['native_layout']['geometry_id'] != registered_cases[case_id]['native_route']['geometry_id']
                or scene['scene']['road_grid']['sha256'] != case['initial']['road_sha256']):
            raise ValueError('Scene identity differs from the registered route or recorded task')
        scene_hashes[path.name] = sha(path)
        if protocol.get('native_hazard') or protocol.get('native_case_manifest'):
            from navstack.native_hazards import verify_native_hazard_scene
            spec=registered_cases[case_id].get('native_hazard')
            if spec is not None:
                verify_native_hazard_scene(scene['scene']['native_layout'],spec,
                    scene['scene'].get('native_hazard',{}),scene['scene']['static'],scene['initial']['dynamic'])
            episode=next(iter(case['episodes']))
            begin=next((r for r in acknowledgements.values() if r.get('episode_id')==episode and 'reset_ticks' in r),None)
            if begin is None or begin.get('scene')!=scene['scene']:
                raise ValueError('Native hazard scene differs from raw reset acknowledgement')
            if protocol.get('native_validation'):verify_registered_native_begin(begin,protocol,case_id,partition='evaluation')
            else:verify_registered_native_begin(begin,protocol,case_id)
            if protocol.get('native_case_manifest') and scene['initial']['dynamic'] != begin['dynamic']:
                raise ValueError('Registered case initial actor observations differ from measured reset')
            identities=[(a['actor_id'],a['actor_type']) for a in begin.get('native_dynamic_actors',[])]
            if any([(a['actor_id'],a['actor_type']) for a in r.get('native_dynamic_actors',[])]!=identities
                   for (owner,_),r in actual.items() if owner==episode):
                raise ValueError('Native rule actor identity changes within measured episode')
    if not actual or used != set(actual) or set(resets) != episodes or len(episodes) != len(inventory):
        raise ValueError('Unaccounted applied controls or resets')
    reset_ticks = sum(resets.values())
    if (summary['evaluation_controls'] != len(used) or summary['reset_ticks'] != reset_ticks
            or summary['world_ticks'] != len(used)+reset_ticks):
        raise ValueError('World/reset/control accounting mismatch')
    return {'scope': ('Incomplete rule diagnostic: only completed prefix verified; missing cases are infrastructure failures, not successes'
                      if partial else 'Raw frame verification of rule-only native evaluation; no learned-policy or training claim'),
            'batch_complete': not partial, 'registered_cases': len(registered_cases),
            'uncompleted_cases': sorted(set(range(len(registered_cases)))-inventory),
            'batch': batch.name, 'protocol_sha256': sha(batch/'protocol.json'),
            'summary_sha256': None if partial else sha(batch/'summary.json'),
            'failure_sha256': sha(batch/'failure.json') if partial else None,
            'trace_sha256': sha(batch/'evaluation_trace.jsonl.gz'),
            'bridge_response_sha256': sha(paths[0]), 'scene_hashes': scene_hashes,
            'evaluation_controls_verified': len(used), 'reset_ticks_verified': reset_ticks,
            'world_ticks_verified': len(used)+reset_ticks, 'cases_verified': len(cases),
            'termination_counts': dict(Counter(r['termination_reason'] for r in summary['episodes'])),
            'training_physics_steps': 0, 'optimizer_updates': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-completed-prefix', action='store_true')
    args = parser.parse_args()
    result = audit(args.batch, args.allow_completed_prefix)
    output = args.output or args.batch/'raw_audit.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))

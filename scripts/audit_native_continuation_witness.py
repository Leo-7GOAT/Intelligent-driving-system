"""Locate loss of local references and goal connectors in measured policy history.

This is an offline hypothesis check. A rejected bounded connector means unknown,
not unreachable; a connector at a proposed target is not an executed continuation.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.planning import analytic_connection
from navstack.road_grid import RoadGrid
from navstack.world import World
from scripts.audit_native_action_availability import state_references, verify_source
from scripts.audit_native_method_endpoint import read, sha
from navstack.experiments import write_json


def probe_state(config, scene, state):
    """Query unchanged reference production and sufficient spatial connectors."""
    references = state_references(config, scene, state)
    world = World(scene['width'], scene['height'], scene['static'],
                  road=RoadGrid(scene['road_grid']))
    goal = scene['goal']
    pose = (state['x'], state['y'], state['yaw'])
    cache, continuing = {}, []
    for row in references['actions']:
        if row['mode'] != 'reference':
            continue
        target = tuple(row['target'])
        if target not in cache:
            cache[target] = analytic_connection(target, goal, world, config.vehicle) is not None
        if cache[target]:
            continuing.append(row['action'])
    return {
        'state_goal_connector': analytic_connection(pose, goal, world, config.vehicle) is not None,
        'reference_actions': references['reference_actions'],
        'reference_actions_with_target_goal_connector': continuing,
        'actions': references['actions'],
    }


def summarize(rows):
    def first(predicate):
        return next((row['decision'] for row in rows if predicate(row)), None)
    return {
        'decisions': len(rows),
        'first_no_moving_reference': first(lambda r: not r['reference_actions']),
        'first_no_state_goal_connector': first(lambda r: not r['state_goal_connector']),
        'no_moving_reference_decisions': sum(not r['reference_actions'] for r in rows),
        'no_state_goal_connector_decisions': sum(not r['state_goal_connector'] for r in rows),
        'selected_reference_without_target_connector_despite_alternative': sum(
            r['recorded_mode'] == 'reference'
            and bool(r['reference_actions_with_target_goal_connector'])
            and r['recorded_action'] not in r['reference_actions_with_target_goal_connector']
            for r in rows),
        'failed_selection_despite_reference': sum(
            r['recorded_mode'] == 'plan_failed' and bool(r['reference_actions']) for r in rows),
    }


def audit(batch, method, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise FileExistsError(output)
    folder = batch / method
    protocol = read(batch / 'protocol.json')
    source_hashes = {k: v for k, v in protocol['source_hashes'].items() if k.startswith('navstack/')}
    for name, digest in source_hashes.items():
        verify_source(ROOT / name, batch / 'source' / name, digest)
    receipt = read(folder / 'endpoint_raw_audit.json')
    for name, key in [('config.json', 'config_sha256'),
                      ('evaluation.json', 'evaluation_sha256'),
                      ('evaluation_trace.jsonl.gz', 'evaluation_trace_sha256')]:
        if sha(folder / name) != receipt[key]:
            raise ValueError('Input differs from completed raw audit: ' + name)
    config = EnvConfig.from_dict(read(folder / 'config.json')['env'])
    scenes, scene_hashes = {}, {}
    for path in sorted(folder.glob('scene_*.json')):
        scene = read(path)['scene']
        identity = scene['road_grid']['sha256']
        if identity not in scenes:
            scenes[identity] = scene
            scene_hashes[path.name] = sha(path)
    expected = {r['case_id']: r['actions'] for r in read(folder / 'evaluation.json')['episodes']}
    previous, cases = {}, {}
    with gzip.open(folder / 'evaluation_trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            recorded = json.loads(line)
            case, info = recorded['case_id'], recorded['info']
            initial = info['initial_scenario']
            scene = scenes[initial['road_sha256']]
            if (any(scene[k] != initial[k] for k in ('goal', 'static'))
                    or scene.get('dynamic', []) != initial['dynamic']):
                raise ValueError('Recorded task geometry mismatch')
            state = previous.get(case, initial['initial'])
            probe = probe_state(config, scene, state)
            action = int(recorded['policy_action'])
            selected = probe.pop('actions')[action]
            if (selected['mode'] != info['mode']
                    or (selected['plan'] or {}).get('failure_reason') != (info['plan'] or {}).get('failure_reason')):
                raise ValueError('Selected reference mode/failure did not reproduce')
            rows = cases.setdefault(case, [])
            rows.append({'decision': len(rows), 'state': state,
                         'recorded_action': action, 'recorded_mode': info['mode'], **probe})
            previous[case] = info['trace'][-1]['state']
            if len(rows) % 30 == 0:
                print(json.dumps({'case_id': case, 'decisions_checked': len(rows)}), flush=True)
    if {k: len(v) for k, v in cases.items()} != expected:
        raise ValueError('Incomplete decision inventory')
    report = {
        'scope': 'Offline unchanged production references and bounded analytic goal connectors at every measured evaluation decision. False connector means unknown, not infeasible. Proposed target is not next actual state. No physical execution, action filtering, policy update or causal effect is claimed.',
        'batch': batch.name, 'method': method, 'script_sha256': sha(Path(__file__)),
        'reference_script_sha256': sha(ROOT / 'scripts/audit_native_action_availability.py'),
        'source_hashes': source_hashes, 'scene_hashes': scene_hashes,
        'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'),
        'evaluation_trace_sha256': receipt['evaluation_trace_sha256'],
        'config_sha256': receipt['config_sha256'], 'new_controls': 0, 'optimizer_updates': 0,
        'cases': [{'case_id': case, 'summary': summarize(rows), 'decisions': rows}
                  for case, rows in sorted(cases.items())],
    }
    write_json(output, report)
    print(json.dumps({'method': method, 'cases': [
        {'case_id': r['case_id'], **r['summary']} for r in report['cases']]}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.batch, args.method, args.output)

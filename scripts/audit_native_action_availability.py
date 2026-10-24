"""Rebuild all action references at recorded native states; never execute a plant."""
import argparse
from collections import Counter
import gzip
import math
from pathlib import Path
import sys
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import write_json
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehiclePlant, VehicleState
from navstack.world import World
from scripts.audit_native_method_endpoint import read, sha


def verify_source(current, frozen, digest):
    # Git checkouts may normalize CRLF. Require exact archived bytes and
    # identical current source text after universal-newline decoding only.
    if sha(frozen) != digest or current.read_text(encoding='utf-8') != frozen.read_text(encoding='utf-8'):
        raise ValueError('Current/frozen native source mismatch: ' + current.name)


def state_references(config, scene, state):
    """Use production decoding, search and time parameterization without reset/step."""
    if scene.get('dynamic'):
        raise ValueError('Moving actors need per-frame reconstruction; unsupported here')
    if config.method != 'H' or config.planner.max_seconds is not None:
        raise ValueError('Requires hierarchical node-budget planning')
    env = NavigationEnv(config)
    env.world = World(scene['width'], scene['height'], scene['static'], road=RoadGrid(scene['road_grid']))
    env.plant = VehiclePlant(config.vehicle, VehicleState(**state))
    env.goal = scene['goal']
    env.episode = {'plan_times': [], 'plan_failures': 0}
    env._rule_plan_cache = None
    initial_distance = math.dist((state['x'], state['y']), scene['goal'][:2])
    rows = []
    for action in range(env.action_space.n):
        target, speed = env.decode_action(action)
        mode, detail = env._prepare_reference(action)
        # Wall time varies by host/load and is not a deterministic feasibility result.
        detail = None if detail is None else {k: v for k, v in detail.items() if k != 'planning_time'}
        row = {'action': action, 'target': target, 'target_speed': speed, 'mode': mode, 'plan': detail}
        if env.reference is not None:
            ref = env.reference
            distance = np.hypot(ref.x - env.goal[0], ref.y - env.goal[1])
            yaw = np.abs(np.arctan2(np.sin(ref.yaw-env.goal[2]), np.cos(ref.yaw-env.goal[2])))
            row.update(reference_end_progress_m=initial_distance-float(distance[-1]),
                       reference_min_goal_distance_m=float(distance.min()),
                       reference_contains_parking_pose=bool(np.any(
                           (distance <= config.goal_tolerance) & (yaw <= config.goal_yaw_tolerance)
                           & (np.abs(ref.v) <= config.goal_speed))))
        rows.append(row)
    assert env.state == VehicleState(**state), 'Offline query changed the measured state'
    return {'state': state, 'goal_distance_m': initial_distance,
            'mode_counts': dict(Counter(r['mode'] for r in rows)),
            'reference_actions': [r['action'] for r in rows if r['mode'] == 'reference'],
            'progress_reference_actions': [r['action'] for r in rows
                if r.get('reference_end_progress_m', 0) > 0],
            'parking_reference_actions': [r['action'] for r in rows
                if r.get('reference_contains_parking_pose', False)], 'actions': rows}


def audit(batch, method, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise FileExistsError(output)
    folder = batch / method
    protocol = read(batch / 'protocol.json')
    # A changed planner/decoder must not masquerade as the frozen learner's interface.
    source_hashes = {name: digest for name, digest in protocol['source_hashes'].items()
                     if name.startswith('navstack/')}
    for name, digest in source_hashes.items():
        verify_source(ROOT / name, batch / 'source' / name, digest)
    receipt = read(folder / 'endpoint_raw_audit.json')
    for name, key in [('config.json', 'config_sha256'),
                      ('evaluation_trace.jsonl.gz', 'evaluation_trace_sha256')]:
        if sha(folder / name) != receipt[key]:
            raise ValueError('Input differs from completed frame audit: ' + name)
    config = EnvConfig.from_dict(read(folder / 'config.json')['env'])
    scenes = {}
    for path in sorted(folder.glob('scene_*.json')):
        scene = read(path)['scene']
        scenes.setdefault(scene['road_grid']['sha256'], (path, scene))
    cases = {}
    with gzip.open(folder / 'evaluation_trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            case = cases.setdefault(row['case_id'], {'initial': row['info']['initial_scenario'],
                                                    'previous_state': row['info']['initial_scenario']['initial']})
            case.update(last_before=case['previous_state'], last_row=row,
                        previous_state=row['info']['trace'][-1]['state'])
    if len(cases) != len(protocol['routes']):
        raise ValueError('Incomplete frozen evaluation inventory')
    results, scene_hashes = [], {}
    for case_id, case in sorted(cases.items()):
        initial = case['initial']
        path, scene = scenes[initial['road_sha256']]
        if any(scene[key] != initial[key] for key in ('goal', 'static')) or scene.get('dynamic', []) != initial['dynamic']:
            raise ValueError('Recorded task geometry mismatch')
        scene_hashes[path.name] = sha(path)
        last = case['last_row']
        for stage, state in [('initial', initial['initial']), ('before_last_decision', case['last_before'])]:
            result = state_references(config, scene, state)
            result.update(case_id=case_id, stage=stage, road_sha256=initial['road_sha256'])
            if stage == 'before_last_decision':
                action = int(last['policy_action'])
                reconstructed = result['actions'][action]
                if reconstructed['mode'] != last['info']['mode']:
                    raise ValueError('Selected action mode does not reproduce')
                recorded_reason = (last['info']['plan'] or {}).get('failure_reason')
                if (reconstructed['plan'] or {}).get('failure_reason') != recorded_reason:
                    raise ValueError('Selected action failure reason does not reproduce')
                result.update(recorded_action=action, recorded_mode=last['info']['mode'],
                              selected_mode_and_failure_reproduced=True,
                              recorded_termination=last['info']['termination_reason'])
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ('actions', 'state')}), flush=True)
    report = {'scope': 'Offline production reference availability at recorded CARLA states; no ticks, policy updates, MPC execution, dynamic-scene coverage or task-completion proof. Search failure is not proof of infeasibility.',
              'batch': batch.name, 'method': method, 'source_hashes': source_hashes,
              'analysis_script_sha256': sha(Path(__file__)), 'scene_hashes': scene_hashes,
              'endpoint_raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'),
              'evaluation_trace_sha256': receipt['evaluation_trace_sha256'],
              'config_sha256': receipt['config_sha256'], 'new_controls': 0, 'optimizer_updates': 0,
              'states': results}
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.batch, args.method, args.output)

"""Offline candidate coverage at measured CARLA states; no vehicle execution."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import write_json
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehiclePlant, VehicleState
from navstack.world import World


def match_scenes(routes, scenes):
    """Evaluation order is protocol order, not the resumed training route order."""
    by_geometry = {}
    for scene in scenes:
        by_geometry.setdefault(scene['scene']['native_layout']['geometry_id'], scene)
    missing = [route['geometry_id'] for route in routes if route['geometry_id'] not in by_geometry]
    if missing:
        raise ValueError('Missing measured training geometry for evaluation routes: '+repr(missing))
    return [by_geometry[route['geometry_id']] for route in routes]


def audit(batch, output):
    batch = Path(batch)
    config_data = json.loads((batch/'H/config.json').read_text(encoding='utf-8'))['env']
    protocol = json.loads((batch/'protocol.json').read_text(encoding='utf-8'))
    scenes = match_scenes(protocol['routes'],
        [json.loads(p.read_text(encoding='utf-8')) for p in sorted((batch/'H').glob('scene_*.json'))])
    terminal = {}
    trace = batch/'H/evaluation_trace.jsonl.gz'
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            terminal[row['case_id']] = row['info']['trace'][-1]['state']
    results = []
    for index, measured in enumerate(scenes):
        scene = measured['scene']
        for label, state in [('initial', measured['initial']['initial']), ('terminal', terminal[index])]:
            for mode in ('ego_arcs', 'goal_frame_terminal'):
                config = EnvConfig.from_dict({**config_data, 'subgoal_mode': mode})
                env = NavigationEnv(config)
                env.world = World(scene['width'], scene['height'], road=RoadGrid(scene['road_grid']))
                # Only decode and query geometry. This plant is never stepped;
                # all poses come from recorded CARLA, not an alternate simulator.
                env.plant = VehiclePlant(config.vehicle, VehicleState(**state))
                env.goal = scene['goal']
                env.execution_outcome = np.zeros(6)
                features = env._execution_feedback()[:93].reshape(31, 3)
                targets = [env.decode_action(a)[0] for a in range(1, 31)]
                results.append({'case_id': index, 'state_label': label, 'state': asdict(env.state),
                    'geometry_id': scene['native_layout']['geometry_id'],
                    'subgoal_mode': mode, 'start_free': bool(env.world.free((env.state.x, env.state.y, env.state.yaw), config.vehicle)),
                    'clear_actions': [int(a) for a in np.flatnonzero(features[:, 0]) if a],
                    'analytic_witness_actions': [int(a) for a in np.flatnonzero(features[:, 1]) if a],
                    'unique_targets': len(set(targets)), 'road_sha256': scene['road_grid']['sha256']})
    result = {'scope': 'Offline coverage of two existing candidate catalogs at eight recorded states; analytic rejection is not proof of no path',
        'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(), 'results': results}
    write_json(output, result)
    for row in results:
        print(row['case_id'], row['state_label'], row['subgoal_mode'], 'start', row['start_free'],
              'clear', len(row['clear_actions']), 'witness', len(row['analytic_witness_actions']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.batch, args.output)

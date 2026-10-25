"""Reconstruct every frozen categorical decision from recorded CARLA feedback."""
import argparse
from collections import deque
import gzip
import json
import math
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.env import ACTIONS, NavigationEnv
from navstack.experiments import write_json
from navstack.independent_clip_ppo import load_ppo
from navstack.road_grid import RoadGrid
from navstack.vehicle import Command, VehicleState, wrap
from navstack.world import World
from scripts.audit_native_action_availability import verify_source
from scripts.audit_native_method_endpoint import read, sha


def observation_env(config, scene, initial):
    if config.method != 'H' or config.vehicle.input_delay_steps:
        raise ValueError('Sequence reconstruction requires hierarchical actions and no input queue')
    if initial['dynamic'] or scene.get('dynamic'):
        raise ValueError('Dynamic actors require per-frame observation reconstruction')
    if scene['goal'] != initial['goal'] or scene['static'] != initial['static']:
        raise ValueError('Task geometry mismatch')
    road = RoadGrid(scene['road_grid'])
    if road.sha256 != initial['road_sha256']:
        raise ValueError('Recorded road mismatch')
    env = NavigationEnv(config)
    env.world = World(scene['width'], scene['height'], scene['static'], road=road)
    env.goal = initial['goal']
    env.navigation_route = scene['native_layout']['path'] if config.subgoal_mode == 'route_frame_terminal' else None
    env.last_behavior = np.zeros(4)
    env.execution_outcome = np.zeros(6, dtype=np.float32)
    # No step method exists: this object only holds already measured feedback.
    env.plant = SimpleNamespace(state=VehicleState(**initial['initial']),
                               previous_command=Command(), delay_queue=deque(), evidence=initial.get('actuator_evidence',{}))
    return env


def advance_observation(env, row):
    """Mirror observable history fields, with states/commands from audited frames."""
    info, action = row['info'], int(row['policy_action'])
    trace = info['trace']
    initial_speed = env.state.v
    if action and (env.config.subgoal_mode.startswith('goal_frame') or env.config.subgoal_mode == 'route_frame_terminal'):
        target, speed = env.decode_action(action)
        dx, dy = target[0]-env.state.x, target[1]-env.state.y
        co, si = math.cos(env.state.yaw), math.sin(env.state.yaw)
        env.last_behavior = np.array([co*dx+si*dy,-si*dx+co*dy,wrap(target[2]-env.state.yaw),speed])
    else:
        env.last_behavior = np.array(ACTIONS[action])
    env.plant.state = VehicleState(**trace[-1]['state'])
    env.plant.previous_command = Command(**trace[-1]['command'])
    env.plant.evidence = trace[-1].get('backend_evidence',{})
    if env.config.execution_feedback != 'off':
        env.execution_outcome = np.array([
            float(info['mode'] == 'reference'), float(info['mode'] == 'plan_failed'),
            float(info['mode'] == 'dynamic_brake'),
            sum(t['controller_status'] == 'failed' for t in trace) / len(trace),
            sum(t['intervention'] for t in trace) / len(trace),
            (env.state.v-initial_speed)/env.config.vehicle.max_speed], dtype=np.float32)


def analyze(batch, method, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise FileExistsError(output)
    folder = batch/method
    protocol, audit = read(batch/'protocol.json'), read(folder/'endpoint_raw_audit.json')
    for name, digest in protocol['source_hashes'].items():
        if name.startswith('navstack/'):
            verify_source(ROOT/name, batch/'source'/name, digest)
    for name, key in [('protocol.json', 'protocol_sha256'), ('config.json', 'config_sha256'),
                      ('evaluation.json', 'evaluation_sha256'),
                      ('evaluation_trace.jsonl.gz', 'evaluation_trace_sha256')]:
        if sha((batch if name == 'protocol.json' else folder)/name) != audit[key]:
            raise ValueError('Input differs from completed raw audit: '+name)
    evaluation = read(folder/'evaluation.json')
    for name, key in [('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256')]:
        if sha(folder/name) != evaluation[key]:
            raise ValueError('Frozen inference artifact mismatch: '+name)
    availability = read(folder/'action_availability_v2.json')
    if availability['endpoint_raw_audit_sha256'] != sha(folder/'endpoint_raw_audit.json'):
        raise ValueError('Availability evidence is not bound to this endpoint')
    available = {(r['case_id'], r['stage']): r for r in availability['states']}
    scenes = {}
    for path in sorted(folder.glob('scene_*.json')):
        scene = read(path)['scene']
        scenes.setdefault(scene['road_grid']['sha256'], scene)
    config = EnvConfig.from_dict(read(folder/'config.json')['env'])
    torch.set_num_threads(1)
    model = load_ppo(folder/'model.zip', device='cpu')
    with (folder/'normalization.pkl').open('rb') as stream:
        norm = pickle.load(stream)
    norm.training = False
    before = (norm.obs_rms.mean.copy(), norm.obs_rms.var.copy(), float(norm.obs_rms.count))
    expected = {r['case_id']: r['actions'] for r in evaluation['episodes']}
    environments, cases = {}, {}
    with gzip.open(folder/'evaluation_trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            case_id = row['case_id']
            if case_id not in environments:
                initial = row['info']['initial_scenario']
                environments[case_id] = observation_env(config, scenes[initial['road_sha256']], initial)
                cases[case_id] = []
            env = environments[case_id]
            obs = norm.normalize_obs(env._observation()[None])
            with torch.no_grad():
                distribution = model.policy.get_distribution(torch.as_tensor(obs)).distribution
                probabilities = distribution.probs.numpy()[0]
            predicted = int(model.predict(obs, deterministic=True)[0][0])
            if predicted != int(row['policy_action']):
                raise ValueError(f'Inference mismatch at case {case_id}, decision {len(cases[case_id])}')
            result = {'decision': len(cases[case_id]), 'action': predicted,
                      'mode': row['info']['mode'], 'maximum_probability': float(probabilities.max()),
                      'entropy': float(distribution.entropy()[0])}
            stage = 'initial' if not cases[case_id] else (
                'before_last_decision' if len(cases[case_id]) == expected[case_id]-1 else None)
            if stage:
                choices = available[case_id, stage]
                if choices['state'] != vars(env.state):
                    raise ValueError('Probability and reference queries use different states')
                indices = choices['reference_actions']
                result.update(stage=stage, probabilities=probabilities.tolist(),
                              reference_actions=indices,
                              reference_probability_mass=float(probabilities[indices].sum()),
                              uniform_reference_probability_mass=len(indices)/len(probabilities),
                              wait_probability=float(probabilities[0]))
            cases[case_id].append(result)
            advance_observation(env, row)
    if {k: len(v) for k, v in cases.items()} != expected:
        raise ValueError('Incomplete decision sequence')
    np.testing.assert_array_equal(before[0], norm.obs_rms.mean)
    np.testing.assert_array_equal(before[1], norm.obs_rms.var)
    assert before[2] == norm.obs_rms.count
    report = {'scope': 'Offline reconstruction of all deterministic categorical decisions from audited CARLA states and observable history; no new execution, stochastic-policy score or causal claim.',
              'batch': batch.name, 'method': method, 'analysis_script_sha256': sha(Path(__file__)),
              'endpoint_raw_audit_sha256': sha(folder/'endpoint_raw_audit.json'),
              'availability_sha256': sha(folder/'action_availability_v2.json'),
              'model_sha256': evaluation['model_sha256'],
              'normalization_sha256': evaluation['normalization_sha256'],
              'decisions_reproduced': sum(expected.values()), 'normalization_unchanged': True,
              'new_controls': 0, 'optimizer_updates': 0,
              'cases': [{'case_id': k, 'decisions': v} for k, v in sorted(cases.items())]}
    write_json(output, report)
    print(json.dumps({'method': method, 'decisions_reproduced': report['decisions_reproduced'],
                      'endpoints': [r for rows in cases.values() for r in rows if r.get('stage')]}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.batch, args.method, args.output)

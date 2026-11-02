"""Sample measured H/H0 inputs and isolate learned feedback use with the physical mask fixed."""
import argparse
from collections import deque
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(batch, method, stride=16):
    batch = Path(batch).resolve()
    if stride != 16 or method not in ('H', 'H0'):
        raise ValueError('Fixed diagnostic sampling: every sixteenth decision plus first/last, H/H0 only')
    if any(k.startswith('navstack.') for k in sys.modules):
        raise RuntimeError('Use a fresh interpreter for each frozen batch/method')
    runtime, folder = batch / 'source', batch / method
    sys.path.insert(0, str(runtime))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.env import ACTIONS, NavigationEnv
    from navstack.native_cases import verify_registered_native_begin
    from navstack.road_grid import RoadGrid
    from navstack.vehicle import Command, VehicleState, wrap
    from navstack.world import MovingObstacle, World
    from stable_baselines3 import PPO
    from scripts.evaluate_native_rule import require_finished_parent

    require_finished_parent(batch)
    p, raw, evaluation = (read(path) for path in
        (batch / 'protocol.json', folder / 'endpoint_raw_audit.json', folder / 'evaluation.json'))
    for name, digest in p['source_hashes'].items():
        if sha(runtime / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for name, key in (('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256')):
        if sha(folder / name) != evaluation[key]:
            raise ValueError('Endpoint bytes differ')
    if raw['protocol_sha256'] != sha(batch / 'protocol.json') or sha(folder / 'evaluation_trace.jsonl.gz') != raw['evaluation_trace_sha256']:
        raise ValueError('Raw audit does not bind this protocol/trace')
    stream_name = raw['bridge_response_stream']
    if Path(stream_name).name != stream_name:
        raise ValueError('Unsafe stream name')
    bridge = batch / 'infrastructure' / stream_name
    if sha(bridge) != raw['bridge_response_sha256']:
        raise ValueError('Measured bridge bytes differ')
    begins = {}
    with gzip.open(bridge, 'rt', encoding='utf-8') as stream:
        for line in stream:
            response = json.loads(line).get('response', {})
            if 'scene' in response:
                begins[response['episode_id']] = response
    with gzip.open(folder / 'evaluation_trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    cfg = EnvConfig.from_dict(read(folder / 'config.json')['env'])
    if not cfg.endpoint_candidate_filter or cfg.vehicle.input_delay_steps:
        raise ValueError('Require the common endpoint mask and observed zero-delay actuator state')
    torch.set_num_threads(1)
    model = PPO.load(folder / 'model.zip', device='cpu')
    model.policy.set_training_mode(False)
    parameters = {k: v.clone() for k, v in model.policy.state_dict().items()}
    with (folder / 'normalization.pkl').open('rb') as stream:
        norm = pickle.load(stream)
    norm.training = False
    rms = (norm.obs_rms.mean.copy(), norm.obs_rms.var.copy(), norm.obs_rms.count)
    if not hasattr(model.policy.mlp_extractor, 'forward_actor_unmasked'):
        raise ValueError('Endpoint extractor required')

    def actors(snapshot):
        identities, states = [], []
        for actor in snapshot:
            identities.append((actor['actor_id'], actor['actor_type']))
            state = actor['state']
            if set(state) != {'x', 'y', 'vx', 'vy', 'radius'} or not all(math.isfinite(x) for x in state.values()):
                raise ValueError('Invalid measured actor state')
            states.append(state)
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate measured actor identity')
        return identities, states

    variants = ('candidate_witnesses', 'previous_execution', 'both')
    records, controls = [], 0
    case_ids = sorted({r['case_id'] for r in rows})
    if case_ids != sorted(e['case_id'] for e in evaluation['episodes']):
        raise ValueError('Evaluation inventory differs')
    for case in case_ids:
        selected = [r for r in rows if r['case_id'] == case]
        first = selected[0]['info']['trace'][0]
        begin = begins[first['backend_evidence']['episode_id']]
        verify_registered_native_begin(begin, p, case, partition='evaluation')
        initial, scene = selected[0]['info']['initial_scenario'], begin['scene']
        identities, states = actors(begin.get('native_dynamic_actors', []))
        if states != initial['dynamic'] or initial['initial'] != begin['state']:
            raise ValueError('Initial measured state differs')
        env = NavigationEnv(cfg)
        env.world = World(scene['width'], scene['height'], scene['static'],
            dynamic=[MovingObstacle(**s) for s in states], road=RoadGrid(scene['road_grid']))
        env.world.road_collision_backend = cfg.road_collision_backend
        env.goal = initial['goal']
        env.navigation_route = scene['native_layout']['path']
        env.last_behavior = np.zeros(4)
        env.execution_outcome = np.zeros(6, dtype=np.float32)
        # No step/reset method exists; only already measured states are assigned.
        env.plant = SimpleNamespace(state=VehicleState(**begin['state']), previous_command=Command(),
            evidence=initial.get('actuator_evidence', {}), delay_queue=deque())
        frame = begin['frame']
        try:
            for index, row in enumerate(selected):
                info, action = row['info'], int(row['policy_action'])
                if index % stride == 0 or index == len(selected) - 1:
                    obs = norm.normalize_obs(env._observation()[None])
                    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], np.array([action]))
                    tensor = torch.as_tensor(obs)
                    with torch.no_grad():
                        features = model.policy.extract_features(tensor)
                        latent = model.policy.mlp_extractor.forward_actor(features)
                        original = model.policy.get_distribution(tensor).distribution
                        baseline = original.probs
                        changes = {}
                        for variant in variants:
                            hidden = features.clone()
                            if variant in ('candidate_witnesses', 'both'):
                                hidden[:, 351:599].reshape(-1, 31, 8)[:, :, 5:] = 0
                            if variant in ('previous_execution', 'both'):
                                hidden[:, 599:605] = 0
                            learned = model.policy.mlp_extractor.forward_actor_unmasked(hidden)
                            # Retain the exact baseline physical availability mask.
                            altered = torch.cat((learned, latent[:, -31:]), dim=1)
                            probs = torch.distributions.Categorical(logits=model.policy.action_net(altered)).probs
                            changes[variant] = {'total_variation': float(.5 * (probs - baseline).abs().sum()),
                                'action': int(probs.argmax(1)[0]), 'argmax_changed': int(probs.argmax(1)[0]) != action}
                    records.append({'case_id': case, 'decision_index': index, 'action': action,
                        'available_actions': int((latent[:, -31:] > .5).sum()),
                        'nonzero_previous_execution_inputs': int(np.count_nonzero(obs[:, 599:605])),
                        'counterfactuals': changes})
                old_speed = env.state.v
                if action:
                    target, speed = env.decode_action(action)
                    dx, dy = target[0] - env.state.x, target[1] - env.state.y
                    co, si = math.cos(env.state.yaw), math.sin(env.state.yaw)
                    env.last_behavior = np.array([co*dx + si*dy, -si*dx + co*dy, wrap(target[2] - env.state.yaw), speed])
                else:
                    env.last_behavior = np.array(ACTIONS[action])
                for tick in info['trace']:
                    evidence = tick['backend_evidence']
                    if tick['backend_frame'] != frame + 1 or evidence['episode_id'] != begin['episode_id']:
                        raise ValueError('Measured frame/episode history differs')
                    ids, states = actors(evidence.get('native_dynamic_actors', []))
                    if ids != identities:
                        raise ValueError('Measured actor identities changed')
                    env.world.dynamic = [MovingObstacle(**s) for s in states]
                    env.plant.state, env.plant.previous_command = VehicleState(**tick['state']), Command(**tick['command'])
                    env.plant.evidence, frame = evidence, tick['backend_frame']
                    controls += 1
                trace, mode = info['trace'], info['mode']
                env.execution_outcome = np.array([float(mode == 'reference'), float(mode == 'plan_failed'),
                    float(mode == 'dynamic_brake'), sum(t['controller_status'] == 'failed' for t in trace)/len(trace),
                    sum(t['intervention'] for t in trace)/len(trace), (env.state.v - old_speed)/cfg.vehicle.max_speed], dtype=np.float32)
        finally:
            env.close()
    if controls != raw['evaluation_control_frames_verified']:
        raise ValueError('Every recorded evaluation frame must be accounted for')
    if any(not torch.equal(v, model.policy.state_dict()[k]) for k, v in parameters.items()):
        raise ValueError('Frozen parameters changed')
    np.testing.assert_array_equal(rms[0], norm.obs_rms.mean)
    np.testing.assert_array_equal(rms[1], norm.obs_rms.var)
    if rms[2] != norm.obs_rms.count:
        raise ValueError('Normalization count changed')
    totals = {variant: {'sampled_decisions': len(records),
        'argmax_changes': sum(r['counterfactuals'][variant]['argmax_changed'] for r in records),
        'mean_total_variation': float(np.mean([r['counterfactuals'][variant]['total_variation'] for r in records])),
        'max_total_variation': max(r['counterfactuals'][variant]['total_variation'] for r in records)} for variant in variants}
    if method == 'H0' and any(v['max_total_variation'] != 0 for v in totals.values()):
        raise ValueError('H0 learned-feedback negative control must be exactly invariant')
    return {'scope': __doc__, 'batch': batch.name, 'method': method, 'script_sha256': sha(__file__),
        'protocol_sha256': sha(batch / 'protocol.json'), 'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'),
        'model_sha256': evaluation['model_sha256'], 'normalization_sha256': evaluation['normalization_sha256'],
        'trace_sha256': raw['evaluation_trace_sha256'], 'sampling_stride': stride, 'cases': len(case_ids),
        'measured_history_controls': controls, 'sampled_actions_exact': len(records),
        'original_physical_mask_retained': True, 'totals': totals, 'records': records,
        'parameters_and_RMS_unchanged': True, 'new_controls': 0, 'new_optimizer_updates': 0,
        'limits': 'Zeros learned normalized feedback inputs only, retaining the original mask. Sampled measured-state inference is not a counterfactual physical rollout, a performance effect or final confirmation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--method', choices=('H', 'H0'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Preserve prior diagnostic evidence')
    result = diagnose(args.batch, args.method)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['totals']), flush=True)

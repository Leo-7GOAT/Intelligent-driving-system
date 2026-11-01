"""Verify a fresh owned physical replay prepared from a downloaded frozen capsule."""
import argparse
from dataclasses import asdict
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.vehicle import Command, VehicleState, wrap
from scripts.audit_native_method_endpoint import read, sha
from scripts.audit_native_control_delivery import analyze as audit_delivery
from scripts.evaluate_native_rule import require_finished_parent


def parking_trace(batch, expected_cases):
    batch = Path(batch)
    p = read(batch / 'protocol.json')
    cfg = EnvConfig.from_dict(read(batch / 'B1/config.json')['env'])
    if not cfg.goal_parking_assist:
        raise ValueError('Require the explicit shared wrapper')
    helper_path = batch / 'source/navstack/parking_assist.py'
    if sha(helper_path) != p['source_hashes']['navstack/parking_assist.py']:
        raise ValueError('Frozen helper differs')
    spec = importlib.util.spec_from_file_location('navstack._replay_parking_assist', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with gzip.open(batch / 'B1/evaluation_trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    episodes = read(batch / 'B1/evaluation.json')['episodes']
    if sorted({r['case_id'] for r in rows}) != list(expected_cases) or sorted(e['case_id'] for e in episodes) != list(expected_cases):
        raise ValueError('Every prospectively registered case is required')
    results = []
    for case in expected_cases:
        selected = [r for r in rows if r['case_id'] == case]
        initial = selected[0]['info']['initial_scenario']
        state = VehicleState(**initial['initial'])
        goal = initial['goal']
        distance, ever, controls, overrides = 0., False, 0, 0
        for row in selected:
            for tick in row['info']['trace']:
                nominal = Command(**tick['unassisted_command'])
                ever |= tick['unassisted_controller_status'] in ('direct', 'solved')
                command, changed, eligible = helper.goal_parking_command(state, goal, nominal, cfg, ever and distance > .1)
                if (asdict(command) != tick['requested_command'] or changed != tick['goal_parking_intervention']
                        or eligible != tick['goal_parking_eligible'] or command.steering != nominal.steering):
                    raise ValueError('Measured frame differs from the frozen common helper')
                if changed and (not tick['intervention'] or tick['controller_status'] != 'goal_parking_assist'):
                    raise ValueError('Unreported assistance')
                after = VehicleState(**tick['state'])
                distance += math.hypot(after.x - state.x, after.y - state.y)
                state = after
                controls += 1
                overrides += int(changed)
        episode = next(e for e in episodes if e['case_id'] == case)
        final = selected[-1]['info']['trace'][-1]
        if episode['physics_steps'] != controls or episode['goal_parking_controls'] != overrides:
            raise ValueError('Control/override counts differ')
        if episode['success'] and (final['intervention'] or math.hypot(state.x - goal[0], state.y - goal[1]) > cfg.goal_tolerance
                                  or abs(wrap(state.yaw - goal[2])) > cfg.goal_yaw_tolerance or abs(state.v) > cfg.goal_speed):
            raise ValueError('Success must satisfy unchanged physical gates without intervention')
        results.append({'case_id': case, 'termination': episode['termination_reason'], 'controls': controls,
                        'parking_overrides': overrides, 'terminal_intervened': final['intervention'], 'final_speed': state.v})
    return {'helper_sha256': sha(helper_path), 'cases': results, 'all_assistance_commands_verified': True,
            'successful_terminal_frames_satisfy_original_unassisted_goal_gate': True}


def analyze(batch):
    batch = Path(batch).resolve()
    require_finished_parent(batch)
    p = read(batch / 'protocol.json')
    declared = p['public_native_reproduction']
    raw = read(batch / 'B1/endpoint_raw_audit.json')
    evaluation = read(batch / 'B1/evaluation.json')
    inference = read(batch / 'B1/measured_start_inference.json')
    sequence = read(batch / 'B1/whole_sequence_inference.json')
    delivery = audit_delivery(batch)
    if (raw['protocol_sha256'] != sha(batch / 'protocol.json') or raw['training_control_frames_verified'] != 0
            or raw['segment_optimizer_steps'] != 0 or not delivery['strict_delivery_gate_passed']
            or evaluation['model_sha256'] != declared['model_sha256']
            or sha(batch / 'B1/model.zip') != sha(batch / 'resume/B1/model.zip')
            or sha(batch / 'B1/normalization.pkl') != sha(batch / 'resume/B1/normalization.pkl')
            or inference['batch_protocol_sha256'] != sha(batch / 'protocol.json') or len(inference['results']) != 1
            or sequence['protocol_sha256'] != sha(batch / 'protocol.json')
            or sequence['total_bit_exact_actions'] != raw['total_applied_controls_verified']):
        raise ValueError('Replay must preserve the published model and bind all physical frames/actions')
    return {'scope': __doc__, 'script_sha256': sha(Path(__file__)), 'batch': batch.name,
            'protocol_sha256': sha(batch / 'protocol.json'), 'model_sha256': evaluation['model_sha256'],
            'normalization_sha256': evaluation['normalization_sha256'], 'registered_reproduction': declared,
            'new_actual_controls': raw['total_applied_controls_verified'], 'new_PPO_updates': 0,
            'all_engine_controls_acknowledged': True, 'all_policy_actions_bit_exact': True,
            'parking': parking_trace(batch, [0]),
            'owned_server_cleanup': read(batch / 'infrastructure/server_stop.json')['status'],
            'raw_audit_sha256': sha(batch / 'B1/endpoint_raw_audit.json'),
            'limits': 'One already-known TRAINcondition from an independent checkout and public download. Not an independent training seed, new final condition, or evidence of added PPO value.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.batch)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)

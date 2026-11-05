"""Probe all actions before one recorded decision; never integrate or drive.

Run in a fresh interpreter so NAVSTACK_AUDIT_SOURCE_ROOT selects the immutable
physical source before navstack is imported. This is a model check, not native
certification of alternatives or a counterfactual task-success experiment.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def selected_rows(rows, case_id, decision_index):
    if type(decision_index) is not int or decision_index < 1:
        raise ValueError('Decision index must be an integer >=1 (zero-based)')
    selected = [row for row in rows if row['case_id'] == case_id]
    if decision_index >= len(selected):
        raise ValueError('Decision index is outside the recorded case')
    return selected[decision_index - 1], selected[decision_index]


def clone_option_env(env, protocol, configure, settings):
    current = copy.deepcopy(env)
    # deepcopy preserves function closures: discard the wrappers that still
    # refer to the original env and bind fresh wrappers to this independent copy.
    for field in ('decode_action', '_prepare_reference', '_reference_option_version'):
        current.__dict__.pop(field, None)
    configure(current, settings(protocol))
    current.episode = dict(plan_times=[], plan_failures=0)
    return current


def reference_endpoint(env):
    ref = env.reference
    if ref is None:
        return None
    pose = [float(ref.x[-1]), float(ref.y[-1]), float(ref.yaw[-1])]
    distance = math.dist(pose[:2], env.goal[:2])
    yaw_error = abs(math.atan2(math.sin(pose[2] - env.goal[2]),
                               math.cos(pose[2] - env.goal[2])))
    return dict(pose=pose, distance_to_goal=distance, yaw_error=yaw_error,
                end_time=float(ref.start_time + ref.t[-1]),
                end_speed=float(ref.v[-1]),
                inside_task_pose_region=(distance < env.config.goal_tolerance
                                         and yaw_error < env.config.goal_yaw_tolerance),
                scope='Nominal reference endpoint only; not achieved vehicle pose')


def probe(batch, case_id, decision_index, output, *, wall_seconds=300.):
    batch, output = Path(batch).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if any(name == 'navstack' or name.startswith('navstack.') for name in sys.modules):
        raise RuntimeError('Use a fresh interpreter to select the frozen source')
    os.environ['NAVSTACK_AUDIT_SOURCE_ROOT'] = str(batch / 'source')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import audit_native_reference_inference as replay
    from scripts.audit_native_method_endpoint import read, read_prefix, sha
    from navstack.research import atomic_json
    from navstack.reference_option_commitment import configure, settings
    from navstack.steering_transition import prepare_reference, settings as steering_settings

    protocol = read(batch / 'protocol.json')
    trace = batch / 'H/evaluation_trace.jsonl.gz'
    rows, error = read_prefix(trace)
    if error:
        raise ValueError('Require a closed complete trace')
    previous, actual = selected_rows(rows, case_id, decision_index)
    registration = dict(case_id=case_id, decision_index=decision_index,
                        protocol_sha256=sha(batch / 'protocol.json'), trace_sha256=sha(trace),
                        script_sha256=sha(Path(__file__)),
                        replay_script_sha256=sha(Path(replay.__file__)),
                        wall_seconds_for_candidates=wall_seconds,
                        new_controls=0, new_updates=0)
    sidecar = output.with_suffix('.registration.json')
    if sidecar.exists():
        raise FileExistsError(sidecar)
    atomic_json(sidecar, registration)
    captured = []
    original = replay.advance_observation

    class CaptureComplete(Exception):
        pass

    def observe(env, row):
        original(env, row)
        if (row['case_id'] == case_id and row['info']['trace'][-1]['backend_frame']
                == previous['info']['trace'][-1]['backend_frame']):
            captured.append(copy.deepcopy(env))
            raise CaptureComplete()

    try:
        replay.advance_observation = observe
        try:
            replay.audit(batch, 'H', all_actions=True)
        except CaptureComplete:
            pass
    finally:
        replay.advance_observation = original
    if len(captured) != 1:
        raise ValueError('Did not capture exactly one measured pre-decision state')
    env = captured[0]
    if env.state.time != previous['info']['trace'][-1]['state']['time']:
        raise ValueError('Captured state does not match recorded pre-decision time')
    action = int(actual['policy_action'])
    order = [action] + [i for i in range(env.action_space.n) if i != action]
    started = time.monotonic()
    results = []
    for candidate in order:
        if time.monotonic() - started > wall_seconds:
            raise TimeoutError('Candidate wall cap exhausted; registration retained')
        current = clone_option_env(env, protocol, configure, settings)
        target, speed = current.decode_action(candidate) if candidate else (None, 0.)
        mode, detail = prepare_reference(current, candidate, current._prepare_reference,
                                         steering_settings(protocol))
        if candidate == action:
            if mode != actual['info']['mode']:
                raise ValueError('Selected preparation mode differs from native record')
            recorded = actual['info'].get('plan')
            for key in ('status', 'expanded_nodes', 'failure_reason'):
                if (detail or {}).get(key) != (recorded or {}).get(key):
                    raise ValueError('Selected preparation differs: ' + key)
        results.append(dict(action=candidate, target=list(target) if target is not None else None,
                            speed=speed, mode=mode, detail=detail,
                            nominal_reference_endpoint=reference_endpoint(current)))
    from collections import Counter
    result = dict(status='COMPLETE', registration=registration, state=vars(env.state),
                  actually_held_action=getattr(env, '_held_reference_action', None),
                  reference_selection=env._reference_selection,
                  actual_action=action, selected_preparation_matches_record=True,
                  mode_counts=dict(Counter(r['mode'] for r in results)), results=results,
                  candidate_wall_seconds=time.monotonic() - started,
                  limits=['Frozen-model preparation only; no alternative native road queries, controls or driving',
                          'Complete prefix inference is checked until capture; this is not a new full-endpoint audit',
                          'Nominal endpoint inside the task pose region does not prove tracking or stopping success',
                          'Search failure is not global infeasibility; no physical source or active training change'])
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--case-id', type=int, required=True)
    parser.add_argument('--decision-index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.batch, args.case_id, args.decision_index, args.output)
    print(json.dumps(dict(status=result['status'], mode_counts=result['mode_counts'])))


if __name__ == '__main__':
    main()

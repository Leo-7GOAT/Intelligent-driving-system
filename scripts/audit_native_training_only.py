"""Bind every training-only transition to measured CARLA frames and saved counters."""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
from navstack.native_cases import verify_registered_native_begin
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read, read_prefix, sha, _verify_ticks
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def audit(batch):
    batch = Path(batch).resolve()
    require_finished_parent(batch)
    p = read(batch / 'protocol.json')
    if p.get('evaluation_only') or p.get('deterministic_training_route_evaluation') is not False:
        raise ValueError('Require the explicitly training-only protocol')
    for name, digest in p['source_hashes'].items():
        if sha(batch / 'source' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    actual, acknowledgements, beginnings, owners, streams = {}, {}, {}, {}, []
    for path in sorted((batch / 'infrastructure').glob('*_responses.jsonl.gz')):
        wrappers, error = read_prefix(path)
        if error:
            raise ValueError('Closed complete bridge stream required: ' + error)
        streams.append(dict(name=path.name, sha256=sha(path)))
        for wrapper in wrappers:
            r = wrapper['response']
            if r.get('episode_id') is not None and r.get('request_id') is not None:
                key = (r['episode_id'], r['request_id'])
                if key in acknowledgements and acknowledgements[key] != r:
                    raise ValueError('Conflicting bridge acknowledgement')
                acknowledgements[key] = r
            if 'reset_ticks' in r:
                if r['episode_id'] in beginnings:
                    raise ValueError('Repeated native episode begin')
                beginnings[r['episode_id']] = r
            if 'vehicle_control' in r:
                key = (r['episode_id'], r['frame'])
                if key in actual:
                    raise ValueError('Duplicate physical frame')
                actual[key] = r
                owners[key] = path.name
    used, methods, world_ticks = set(), {}, 0
    for method in p['methods']:
        folder = batch / method
        summary = read(folder / 'summary.json')
        config = read(folder / 'config.json')['env']
        if summary['stopped_reason'] is not None:
            raise ValueError('Training stopped before its registered endpoint')
        if (folder / 'evaluation_trace.jsonl.gz').exists():
            rows, error = read_prefix(folder / 'evaluation_trace.jsonl.gz')
            if rows or error:
                raise ValueError('Training-only method unexpectedly contains evaluation')
        before = set(used)
        previous, episodes, rewards, hashes = {}, set(), Counter(), {}
        controls = actions = 0
        for path in sorted(folder.glob('transitions*.gz')):
            rows, error = read_prefix(path)
            if error:
                raise ValueError('Closed complete training trace required: ' + error)
            hashes[path.name] = sha(path)
            for row in rows:
                trace = row['trace']
                controls += _verify_ticks(trace, actual, acknowledgements, owners, used, previous,
                                          episodes, config.get('actuator_observation', 'off'))
                actions += 1
                if row['physics_steps'] != controls:
                    raise ValueError('Transition cumulative physical count differs')
                for tick in trace:
                    for key, value in tick['reward_components'].items():
                        if not math.isfinite(value):
                            raise ValueError('Nonfinite recorded reward')
                        rewards[key] += value
        if not hashes or (controls, actions) != (summary['physics_steps'], summary['actions']):
            raise ValueError('Training frame/action inventory differs from saved summary')
        if set(rewards) != set(summary['reward_components']) or any(
                not math.isclose(value, summary['reward_components'][key], rel_tol=1e-10, abs_tol=1e-10)
                for key, value in rewards.items()):
            raise ValueError('Recorded reward totals differ from saved summary')
        for episode in episodes:
            begin = beginnings[episode]
            verify_registered_native_begin(begin, p, begin['scene']['condition'])
            first = min((key for key in used - before if key[0] == episode), key=lambda key: key[1])
            if first[1] != begin['frame'] + 1:
                raise ValueError('First training frame is missing')
        optimization, error = read_prefix(folder / 'optimization.jsonl')
        if error or not optimization:
            raise ValueError('Missing complete optimizer evidence')
        for key in ('physics_steps', 'optimization_phases', 'optimizer_steps'):
            if optimization[-1][key] != summary[key]:
                raise ValueError('Last optimization differs from saved summary: ' + key)
        checkpoint = folder / read(folder / 'latest_checkpoint.json')['folder']
        metadata = verified_checkpoint_metadata(checkpoint)
        if any(metadata[key] != summary[key] for key in ('physics_steps', 'actions', 'optimizer_steps')):
            raise ValueError('Saved checkpoint differs from the finished training counters')
        method_streams = {owners[key] for key in used - before}
        if len(method_streams) != 1:
            raise ValueError('One learner maps to multiple native bridge streams')
        world_ticks += summary['world_ticks']
        methods[method] = dict(controls=controls, actions=actions, optimizer_steps=summary['optimizer_steps'],
            episodes_with_controls=len(episodes), bridge_response_stream=next(iter(method_streams)),
            transition_hashes=hashes, reward_components=dict(rewards),
            model_sha256=sha(folder / 'model.zip'), normalization_sha256=sha(folder / 'normalization.pkl'),
            checkpoint_metadata_sha256=sha(checkpoint / 'metadata.json'))
    if used != set(actual):
        raise ValueError('Bridge has unaccounted controls outside all recorded training traces')
    reset_ticks = sum(r['reset_ticks'] for r in beginnings.values())
    if world_ticks != len(used) + reset_ticks:
        raise ValueError('World tick inventory differs from control and reset evidence')
    return dict(scope=__doc__, batch=batch.name, protocol_sha256=sha(batch / 'protocol.json'),
        script_sha256=sha(Path(__file__)), bridge_streams=streams, methods=methods,
        training_control_frames_verified=len(used), reset_ticks_verified=reset_ticks,
        new_controls=0, new_updates=0,
        limits='Authenticates recorded state,commands,sensor evidence and reward accumulation;does not establish policy superiority or re-execute physics.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-identity', help='Optional exact owned worker identity; wait without restarting or terminating it')
    args = parser.parse_args()
    batch, output = Path(args.batch).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous training audit')
    if args.wait_identity:
        from navstack.owned_process import OwnedProcess
        identity = read(args.wait_identity)
        owner = OwnedProcess(identity['pid'], identity['creation_epoch'], identity['executable'], allow_terminate=False)
        atomic_json(output.with_suffix('.waiting.json'), dict(worker_identity=identity, auditor_pid=os.getpid(),
            source_sha256=sha(Path(__file__)), automatic_retries=0))
        deadline = time.monotonic() + read(batch / 'protocol.json')['resource_limits']['wall_seconds'] + 600
        try:
            while owner.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('Bounded training wait exhausted')
                time.sleep(10)
            if owner.poll() != 0:
                raise ValueError('Training failed; do not infer a completed raw audit')
            while not (batch / 'infrastructure/server_stop.json').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Owned cleanup wait exhausted')
                time.sleep(2)
        finally:
            owner.kernel.CloseHandle(owner.handle)
    result = audit(batch)
    atomic_json(output, result)
    print({key: result[key] for key in ('batch', 'training_control_frames_verified', 'reset_ticks_verified')})


if __name__ == '__main__':
    main()

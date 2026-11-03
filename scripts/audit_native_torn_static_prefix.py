"""Verify only fully retained static episodes inside a disk-interrupted evaluation."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
from navstack.native_cases import verify_registered_native_begin
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read, read_prefix, sha, _verify_ticks, verify_initial_actuator
from scripts.audit_interrupted_native import frozen_evaluation_prefix
from scripts.audit_native_control_delivery import control_errors
from scripts.prepare_native_evaluation_recovery import recorded_lifecycle_gone


def audit(batch, method):
    batch = Path(batch).resolve()
    protocol = read(batch / 'protocol.json')
    if method not in protocol['methods'] or protocol.get('native_validation'):
        raise ValueError('Require a registered static development method')
    final, stop = read(batch / 'guard_final.json'), read(batch / 'infrastructure/server_stop.json')
    worker, watch = read(batch / 'worker_process.json'), read(batch / 'guard.json')
    server = read(batch / 'infrastructure/server_start.json')
    identities = [worker['pid'], watch['pid'], server['shipping_pid'], server['launcher_pid']]
    if (final.get('reason') != 'disk_below_12GiB' or final.get('exit_code') != 1
            or final.get('worker_pid') != worker['pid'] or not server.get('owned_by_task')
            or stop.get('shipping_pid') != server['shipping_pid']
            or stop.get('status') != 'owned_empty_server_stopped'
            or stop.get('post_stop_tcp') != [] or stop.get('post_stop_udp') != []
            or not recorded_lifecycle_gone(identities, stop['epoch'])):
        raise ValueError('Require verified disk stop and ended owned lifecycle')
    folder = batch / method
    prefix = frozen_evaluation_prefix(folder, protocol)
    checkpoint = folder / read(folder / 'latest_checkpoint.json')['folder']
    origin = batch / protocol['recovery'][method]['checkpoint']
    for name in ('model.zip', 'normalization.pkl'):
        if sha(folder / name) != sha(origin / name) or sha(checkpoint / name) != sha(origin / name):
            raise ValueError('Frozen policy bytes differ from their registered source')
    actual, acknowledgements, beginnings, streams = {}, {}, {}, []
    for path in sorted((batch / 'infrastructure').glob('*_responses.jsonl.gz')):
        wrappers, error = read_prefix(path)
        streams.append(dict(name=path.name, sha256=sha(path), read_error=error, complete_rows=len(wrappers)))
        for wrapper in wrappers:
            r = wrapper['response']
            if r.get('episode_id') is not None and r.get('request_id') is not None:
                key = (r['episode_id'], r['request_id'])
                if key in acknowledgements and acknowledgements[key] != r:
                    raise ValueError('Conflicting acknowledgement')
                acknowledgements[key] = r
            if 'reset_ticks' in r:
                beginnings[r['episode_id']] = r
            if 'vehicle_control' in r:
                key = (r['episode_id'], r['frame'])
                if key in actual:
                    raise ValueError('Duplicate applied frame')
                actual[key] = r
    rows, error = read_prefix(folder / 'evaluation_trace.jsonl.gz')
    completed = []
    for case in sorted({r['case_id'] for r in rows}):
        selected = [r for r in rows if r['case_id'] == case]
        terminals = [i for i, r in enumerate(selected) if r['info'].get('episode_metrics') is not None]
        if not terminals:
            continue
        if terminals != [len(selected) - 1]:
            raise ValueError('Controls after terminal or repeated terminal')
        first = selected[0]['info']['trace'][0]
        episode = first['backend_evidence']['episode_id']
        begin = beginnings[episode]
        if begin['dynamic']:
            raise ValueError('This auditor is limited to static measured scenes')
        verify_registered_native_begin(begin, protocol, case)
        verify_initial_actuator(selected[0]['info']['initial_scenario'], episode, acknowledgements)
        if first['backend_frame'] != begin['frame'] + 1:
            raise ValueError('Missing first physical control')
        used, previous, episode_ids = set(), {}, set()
        for row in selected:
            _verify_ticks(row['info']['trace'], actual, acknowledgements, {}, used, previous, episode_ids,
                          read(folder / 'config.json')['env'].get('actuator_observation', 'off'))
        if episode_ids != {episode} or used != {key for key in actual if key[0] == episode}:
            raise ValueError('Completed episode has missing, extra or mixed controls')
        for key in used:
            r = actual[key]
            applied = r['applied_vehicle_control']
            if (r.get('control_delivery') != 'batch_sync_v1'
                    or r.get('applied_control_matches_request') is not True
                    or control_errors(r['vehicle_control'], applied)
                    or applied['hand_brake'] or applied['reverse']):
                raise ValueError('Engine acknowledgement differs from requested control')
        info = selected[-1]['info']
        metric = info['episode_metrics']
        if (metric['physics_steps'] != len(used) or metric['actions'] != len(selected)
                or metric['termination_reason'] != info['termination_reason']):
            raise ValueError('Terminal metrics differ from retained frame inventory')
        completed.append(dict(case_id=case, controls=len(used), metrics=metric))
    if not completed:
        raise ValueError('No complete measured episode retained')
    return dict(scope=__doc__, batch=batch.name, method=method, script_sha256=sha(Path(__file__)),
        protocol_sha256=sha(batch / 'protocol.json'), model_sha256=sha(folder / 'model.zip'),
        evaluation_trace_sha256=sha(folder / 'evaluation_trace.jsonl.gz'), trace_read_error=error,
        frozen_prefix=prefix, bridge_streams=streams, completed_cases=completed,
        completed_case_ids=[r['case_id'] for r in completed],
        remaining_case_ids=[i for i in range(len(protocol['native_case_manifest']['cases']))
                            if i not in {r['case_id'] for r in completed}],
        verified_completed_controls=sum(r['controls'] for r in completed),
        termination_counts=dict(Counter(r['metrics']['termination_reason'] for r in completed)),
        full_evaluation_complete=False, new_controls=0, new_updates=0,
        limits='Torn files are preserved. Only complete retained episodes are authenticated; unreadable tails and other partial episodes remain unverified. No evaluation.json is synthesized.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous prefix audit')
    result = audit(args.batch, args.method)
    atomic_json(output, result)
    print({key: result[key] for key in ('completed_case_ids', 'remaining_case_ids', 'verified_completed_controls', 'termination_counts')})

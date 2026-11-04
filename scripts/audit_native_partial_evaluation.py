"""Authenticate completed cases inside a failed frozen evaluation without synthesizing an endpoint."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, read_prefix, sha, _verify_ticks, verify_initial_actuator
from scripts.audit_interrupted_native import frozen_evaluation_prefix
from scripts.audit_native_control_delivery import analyze as delivery_audit
from scripts.prepare_native_evaluation_recovery import verified_interruption


def audit(batch,method='B1',development_stop=False):
    from navstack.native_cases import verify_registered_native_begin
    batch = Path(batch).resolve()
    if development_stop:
        from scripts.prepare_native_evaluation_recovery import recorded_lifecycle_gone
        final,stop=read(batch/'guard_final.json'),read(batch/'infrastructure/server_stop.json')
        worker,watch=read(batch/'worker_process.json'),read(batch/'guard.json')
        server=read(batch/'infrastructure/server_start.json')
        registration=read(batch/'development_stop_registration.json')
        identities=[worker['pid'],watch['pid'],server['shipping_pid'],server['launcher_pid']]
        if (read(batch/'STOP')!=registration or final.get('exit_code')!=1 or final.get('reason') is not None
                or final.get('worker_pid')!=worker['pid'] or not server.get('owned_by_task')
                or read(batch/'failure.json').get('error')!="RuntimeError('Diagnostic deadline/STOP reached')"
                or not registration['epoch']<final['epoch']
                or stop.get('status')!='owned_empty_server_stopped' or stop.get('shipping_pid')!=server['shipping_pid']
                or stop.get('post_stop_tcp')!=[] or stop.get('post_stop_udp')!=[]
                or not recorded_lifecycle_gone(identities,stop['epoch'])):
            raise ValueError('Require closed cooperative development STOP and ended owned lifecycle')
        reason='Intentional development rejection,not infrastructure failure: '+registration['reason']
    else:reason = verified_interruption(batch)
    p = read(batch / 'protocol.json')
    if method not in p['methods']:raise ValueError('Unregistered method')
    folder=batch/method
    prefix = frozen_evaluation_prefix(folder, p)
    if prefix['evaluation_trace_error'] is not None:
        raise ValueError('Cannot treat a torn trace as complete case evidence')
    actual, acknowledgements, beginnings, streams = {}, {}, {}, []
    for path in (batch / 'infrastructure').glob('*_responses.jsonl.gz'):
        wrappers, error = read_prefix(path)
        if error:
            raise ValueError('Require complete closed bridge streams for this prefix audit')
        streams.append({'name': path.name, 'sha256': sha(path)})
        for wrapper in wrappers:
            r = wrapper.get('response', {})
            if r.get('episode_id') is not None and r.get('request_id') is not None:
                acknowledgements[(r['episode_id'], r['request_id'])] = r
            if 'reset_ticks' in r:
                beginnings[r['episode_id']] = r
            if 'vehicle_control' in r:
                key = (r['episode_id'], r['frame'])
                if key in actual:
                    raise ValueError('Duplicate measured control')
                actual[key] = r
    rows, error = read_prefix(folder/'evaluation_trace.jsonl.gz')
    if error:
        raise ValueError(error)
    used, previous, episode_ids = set(), {}, set()
    config = read(folder/'config.json')['env']
    completed = []
    for case in sorted({r['case_id'] for r in rows}):
        selected = [r for r in rows if r['case_id'] == case]
        first = selected[0]['info']['trace'][0]
        episode = first['backend_evidence']['episode_id']
        verify_registered_native_begin(beginnings[episode], p, case)
        verify_initial_actuator(selected[0]['info']['initial_scenario'], episode, acknowledgements)
        count = 0
        for row in selected:
            count += _verify_ticks(row['info']['trace'], actual, acknowledgements, {}, used, previous,
                                   episode_ids, config.get('actuator_observation', 'off'))
        metric = selected[-1]['info'].get('episode_metrics')
        if metric is None:
            continue
        if metric['physics_steps'] != count:
            raise ValueError('Completed-case metric differs from its measured control inventory')
        completed.append({'case_id': case, 'controls': count, 'success': metric['success'],
                          'termination': metric['termination_reason'], 'metrics': metric})
    if used != set(actual) or len(used) != prefix['evaluation_controls_readable_lower_bound']:
        raise ValueError('Partial trace must account for every actual control')
    delivery = delivery_audit(batch)
    if not delivery['strict_delivery_gate_passed']:
        raise ValueError('Engine control acknowledgement failed')
    return {'scope': __doc__, 'script_sha256': sha(Path(__file__)), 'batch': batch.name,
            'protocol_sha256': sha(batch / 'protocol.json'), 'infrastructure_failure': reason,
            'evaluation_trace_sha256': sha(folder/'evaluation_trace.jsonl.gz'), 'bridge_streams': streams,
            'all_recorded_actual_controls_verified': len(used), 'all_engine_controls_acknowledged': True,
            'completed_cases': completed, 'completed_case_ids': [r['case_id'] for r in completed],
            'remaining_case_ids': [i for i in range(len(p['native_case_manifest']['cases'])) if i not in {r['case_id'] for r in completed}],
            'completed_termination_counts': dict(Counter(r['termination'] for r in completed)),
            'registered_cases': len(p['native_case_manifest']['cases']), 'full_evaluation_complete': False,
            'new_controls_for_audit': 0, 'new_PPO_updates': 0,
            'limits': 'Authenticates retained measured prefix only;does not create evaluation.json,repair original data,declare a completed24-case endpoint or count infrastructure failure as policy success.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method',default='B1');parser.add_argument('--development-stop',action='store_true')
    args = parser.parse_args()
    result = audit(args.batch,args.method,args.development_stop)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('all_recorded_actual_controls_verified', 'completed_termination_counts', 'remaining_case_ids')}))

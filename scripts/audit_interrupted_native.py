"""Preserve readable evidence from an interrupted native batch, never repair logs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time


def prefix_rows(path):
    rows, error = [], None
    try:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                rows.append(json.loads(line))
    except (EOFError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    return rows, error


def frozen_evaluation_prefix(folder,protocol):
    """Authenticate zero-learning evaluation without inventing a heartbeat."""
    if (not protocol.get('evaluation_only') or protocol.get('controlled_budget_per_method')!=0
            or protocol.get('method_budgets',{}).get(folder.name)!=0):
        raise ValueError('Missing heartbeat is not a registered zero-learning evaluation')
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    name=json.loads((folder/'latest_checkpoint.json').read_text(encoding='utf-8'))['folder']
    if Path(name).name!=name:raise ValueError('Invalid frozen checkpoint identity')
    checkpoint=folder/name;metadata=json.loads((checkpoint/'metadata.json').read_text(encoding='utf-8'))
    if (not summary.get('reload_exact') or summary.get('stopped_reason') is not None
            or any(type(record.get(key)) is not int or record[key]!=0
                   for record in (summary,metadata) for key in ('physics_steps','actions','optimizer_steps'))
            or (folder/'optimization.jsonl').read_bytes().strip()):
        raise ValueError('Evaluation prefix contains unverified learning')
    for filename,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),('resume_state.pkl','resume_state_sha256')]:
        if hashlib.sha256((checkpoint/filename).read_bytes()).hexdigest()!=metadata[key]:
            raise ValueError('Frozen evaluation checkpoint changed')
        if filename!='resume_state.pkl' and (folder/filename).read_bytes()!=(checkpoint/filename).read_bytes():
            raise ValueError('Evaluated model/normalizer differs from checkpoint')
    if (folder/'config.json').read_bytes()!=(checkpoint/'config.json').read_bytes():
        raise ValueError('Frozen evaluation configuration changed')
    for path in folder.glob('transitions*.gz'):
        rows,error=prefix_rows(path)
        if rows or error:raise ValueError('Zero-learning evaluation has nonempty/invalid training trace')
    rows,error=prefix_rows(folder/'evaluation_trace.jsonl.gz')
    count=sum(len(row['info']['trace']) for row in rows)
    return {'state':'interrupted_frozen_evaluation','readable_trace_controls':0,
        'evaluation_controls_readable_lower_bound':count,'evaluation_trace_error':error,
        'consumed_controls_lower_bound':count,'unknown_unflushed_tail':error is not None,
        'checkpoint':name,'scope':'Authenticated zero-learning saved model with readable partial evaluation. No synthetic training heartbeat or completed endpoint claim.'}


def audit(batch):
    batch = Path(batch)
    protocol = json.loads((batch/'protocol.json').read_text(encoding='utf-8'))
    evidence, methods = {}, {}
    for path in sorted((batch/'infrastructure').glob('*_responses.jsonl.gz')):
        rows, error = prefix_rows(path)
        evidence[path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'complete_rows': len(rows), 'read_error': error,
            'applied_controls': sum('vehicle_control' in row['response'] for row in rows)}
    for method in protocol['methods']:
        folder = batch/method
        if not folder.exists():
            methods[method] = {'state': 'not_started', 'consumed_controls_lower_bound': 0}
            continue
        # The worker creates config.json before opening the bridge. A verified
        # pre-bridge interpreter failure can therefore leave a folder but no
        # learner heartbeat. Do not treat other missing-heartbeat cases as zero
        # cost: a bridge or any additional artifact makes that inference unsafe.
        if not (folder/'heartbeat.json').exists():
            if protocol.get('evaluation_only'):
                methods[method]=frozen_evaluation_prefix(folder,protocol)
                continue
            failure_path = batch/'failure.json'
            failure = json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else {}
            entries = {p.name for p in folder.iterdir()}
            if (entries == {'config.json'} and not evidence and
                    'FileNotFoundError' in failure.get('error', '') and
                    'CARLA client interpreter not found:' in failure.get('error', '')):
                methods[method] = {'state': 'not_started', 'consumed_controls_lower_bound': 0,
                    'startup_failure': 'Missing client interpreter before bridge creation; config retained'}
                continue
            raise ValueError('Missing heartbeat with unverified interaction history: '+method)
        count, errors = 0, []
        for path in folder.glob('transitions*.gz'):
            rows, error = prefix_rows(path)
            count += sum(len(row['trace']) for row in rows)
            if error:
                errors.append({'file': path.name, 'error': error})
        optimizations, opt_error = prefix_rows(folder/'optimization.jsonl')
        heartbeat = json.loads((folder/'heartbeat.json').read_text(encoding='utf-8'))
        checkpoints = []
        for path in sorted(folder.glob('checkpoint_*/metadata.json')):
            saved = json.loads(path.read_text(encoding='utf-8'))
            for filename,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
                assert hashlib.sha256((path.parent/filename).read_bytes()).hexdigest() == saved[key]
            checkpoints.append({'folder': path.parent.name, 'physics_steps': saved['physics_steps']})
        methods[method] = {'state': 'complete' if (folder/'summary.json').exists() and (folder/'evaluation.json').exists() else 'interrupted',
            'readable_trace_controls': count, 'trace_errors': errors, 'optimization_error': opt_error,
            'last_optimization': {k:optimizations[-1][k] for k in ('physics_steps','optimization_phases','optimizer_steps')} if optimizations else None,
            'consumed_controls_lower_bound': max(count, heartbeat['physics_steps'], max((r['physics_steps'] for r in optimizations), default=0)),
            'checkpoints': checkpoints, 'unknown_unflushed_tail': not (folder/'summary.json').exists()}
    result = {'epoch': time.time(), 'scope': 'Readable-prefix audit only; absent footer/tail is not fabricated or silently repaired. OS liveness must be separately checked before recovery.',
        'batch': batch.name, 'methods': methods, 'bridge_streams': evidence}
    target = batch/'interruption_audit.json'
    if target.exists():
        raise FileExistsError('Preserve previous interruption audit')
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(methods))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True)
    audit(p.parse_args().batch)

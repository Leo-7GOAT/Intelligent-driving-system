"""Audit the selected planner outcomes behind a completed native policy score."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def audit(batch, method, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior planning-outcome evidence')
    folder = batch/method
    raw = json.loads((folder/'endpoint_raw_audit.json').read_text(encoding='utf-8'))
    evaluation = folder/'evaluation.json'
    trace = folder/'evaluation_trace.jsonl.gz'
    for file, key in ((evaluation, 'evaluation_sha256'), (trace, 'evaluation_trace_sha256')):
        if hashlib.sha256(file.read_bytes()).hexdigest() != raw[key]:
            raise ValueError('Input differs from completed raw frame audit')
    episodes = json.loads(evaluation.read_text(encoding='utf-8'))['episodes']
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        decisions = [json.loads(line) for line in stream]
    rows = []
    for episode in episodes:
        records = [r for r in decisions if r['case_id'] == episode['case_id']]
        if len(records) != episode['actions'] or sum(r['info']['executed_substeps'] for r in records) != episode['physics_steps']:
            raise ValueError('Selected decisions do not account for complete episode')
        outcomes = Counter((r['policy_action'], r['info']['mode'], (r['info'].get('plan') or {}).get('failure_reason')) for r in records)
        rows.append({'case_id': episode['case_id'], 'success': episode['success'], 'termination_reason': episode['termination_reason'],
                     'decisions': len(records), 'controls': episode['physics_steps'],
                     'recorded_planning_seconds': sum((r['info'].get('plan') or {}).get('planning_time', 0.) for r in records),
                     'selected_outcomes': [{'action': a, 'mode': m, 'failure_reason': f, 'count': n} for (a, m, f), n in sorted(outcomes.items(), key=str)]})
    result = {'scope': 'Selected-action planning diagnostics on already audited development frames;no counterfactual feasibility or causal learning claim.',
              'batch': batch.name, 'method': method, 'macro_decisions': len(decisions),
              'controls_accounted': sum(r['controls'] for r in rows),
              'recorded_planning_seconds': sum(r['recorded_planning_seconds'] for r in rows), 'episodes': rows,
              'new_physical_controls': 0, 'optimizer_updates': 0, 'new_planner_queries': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'decisions': result['macro_decisions'], 'controls': result['controls_accounted'], 'recorded_planning_seconds': result['recorded_planning_seconds']}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('batch', 'method', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    audit(args.batch, args.method, args.output)

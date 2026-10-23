"""Audit all counterfactual controls and exact unchanged-pair trajectories."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_key(case):
    return json.dumps([case['scenario'], case['seed'], case.get('options', {})], sort_keys=True)


def update_trace(digest, row):
    # Timing instrumentation is intentionally excluded; actual action/state are
    # required to agree exactly with the separately recorded original evaluation.
    digest.update(json.dumps([row['action'], row['info']['trace'][0]['state']], sort_keys=True).encode())


def verify(batch, output, verification_output=None):
    batch, output = Path(batch).resolve(), Path(output).resolve()
    protocol, inputs = read(output/'protocol.json'), read(output/'inputs.json')
    result, completed = read(output/'results.json'), read(output/'completed.json')
    guard = read(output/'guard_final.json')
    assert guard['exit_code'] == 0 and guard['reason'] is None
    assert sha(output/'audit_source.py') == protocol['audit_source_sha256']
    assert sha(batch/'protocol.json') == inputs['parent_protocol_sha256']
    parent = read(batch/'protocol.json')
    for name, digest in parent['source_hashes'].items():
        assert sha(batch/'source'/name) == digest
    targets = [protocol['left'], protocol['right']]
    originals, original_counts = {}, {}
    for target in targets:
        evaluation = read(batch/'B1'/f'validation_{target}.json')
        checkpoint = batch/'B1'/inputs['checkpoints'][str(target)]['checkpoint']
        provenance = inputs['checkpoints'][str(target)]
        assert sha(checkpoint/'model.zip') == provenance['model_sha256'] == evaluation['model_sha256']
        assert sha(checkpoint/'normalization.pkl') == provenance['normalization_sha256']
        assert read(checkpoint/'config.json')['env'] == parent['env']
        digests, counts = {}, Counter()
        with gzip.open(batch/'B1'/f'validation_{target}_trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                key = case_key(parent['cases'][row['case_id']])
                update_trace(digests.setdefault(key, hashlib.sha256()), row)
                counts[key] += 1
        originals[target] = {key: value.hexdigest() for key, value in digests.items()}
        original_counts[target] = counts
    expected_cases = {case_key(case): case for case in inputs['cases']}
    assert len(expected_cases) == len(inputs['cases'])
    vehicle, dt = parent['env']['vehicle'], parent['env']['control_dt']
    counts, terminals, digests = Counter(), {}, {}
    with gzip.open(output/'trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            policy, norm, case = row['policy_target'], row['normalizer_target'], case_key(row['case'])
            assert policy in targets and norm in targets and row['case'] == expected_cases[case]
            key = (policy, norm, case)
            assert key not in terminals, 'Control after a terminal event'
            info = row['info']
            assert info['executed_substeps'] == len(info['trace']) == 1
            tick = info['trace'][0]
            assert tick['controller_status'] == 'direct' and info['mode'] == 'flat'
            action = np.asarray(row['action']).clip(-1, 1)
            command = tick['requested_command']
            assert abs(command['acceleration']-float(action[0])*(vehicle['max_accel'] if action[0]>=0 else vehicle['max_decel'])) < 1e-6
            assert abs(command['steering']-float(action[1])*vehicle['max_steer']) < 1e-6
            counts[key] += 1
            assert abs(tick['state']['time']-counts[key]*dt) < 1e-8
            update_trace(digests.setdefault(key, hashlib.sha256()), row)
            if info['termination_reason']:
                terminals[key] = (info['episode_metrics'], tick['state'])
    assert len(result['episodes']) == len(terminals) == 4*len(expected_cases)
    seen = set()
    for episode in result['episodes']:
        key = (episode['policy_target'], episode['normalizer_target'], case_key(episode['case']))
        assert key not in seen
        seen.add(key)
        metrics, state = terminals[key]
        assert all(episode[k] == v for k, v in metrics.items())
        assert episode['terminal_state'] == state and episode['physics_steps'] == counts[key]
        if key[0] == key[1]:
            assert digests[key].hexdigest() == originals[key[0]][key[2]], 'Unchanged pair trajectory differs'
            assert counts[key] == original_counts[key[0]][key[2]]
    assert {(g['policy_target'],g['normalizer_target']) for g in result['groups']} == {(a,b) for a in targets for b in targets}
    assert len(result['groups']) == completed['groups'] == 4
    assert completed['episodes'] == len(terminals)
    for group in result['groups']:
        episodes = [e for e in result['episodes'] if (e['policy_target'],e['normalizer_target']) == (group['policy_target'],group['normalizer_target'])]
        assert len(episodes) == len(expected_cases)
        assert sum(e['success'] for e in episodes if not e['case']['_probe']) == group['full_task_success']
        assert {str(d):sum(e['success'] for e in episodes if e['case']['label']==f'retention_{d}m') for d in (6,12,18,23)} == group['retention']
    assert sum(counts.values()) == result['actual_controls'] == completed['actual_controls']
    assert result['optimizer_updates'] == result['teacher_queries'] == completed['optimizer_updates'] == 0
    summary = {'scope': 'All factorial commands/clocks/terminals checked; diagonal actions and states exactly reproduce original evaluations',
               'actual_controls': sum(counts.values()), 'episodes':len(terminals),
               'exact_diagonal_controls':sum(count for key,count in counts.items() if key[0]==key[1]),
               'groups':result['groups'], 'trace_sha256':sha(output/'trace.jsonl.gz'),
               'results_sha256':sha(output/'results.json'), 'optimizer_updates':0, 'teacher_queries':0}
    destination=Path(verification_output) if verification_output else output/'verification.json'
    destination.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--verification-output')
    args=parser.parse_args()
    verify(args.batch,args.output,args.verification_output)

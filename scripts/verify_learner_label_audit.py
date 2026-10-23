"""Check query-only diagnostics against actual commands and prior evaluation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(parent, output):
    parent, output = Path(parent), Path(output)
    protocol, completed = read(output/'protocol.json'), read(output/'completed.json')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(output/'executed_script.py') == protocol['script_sha256']
    assert sha(parent/'supervised_checkpoint/model.zip') == protocol['model_sha256']
    assert completed['teacher_executed_controls'] == completed['ppo_updates'] == completed['supervised_updates'] == 0
    original = read(parent/'evaluation/validation_0.json')
    old_indices = {(r['scenario'], r['seed']): r['case_id'] for r in original['episodes'] if 'label' not in r}
    original_rows = {}
    with gzip.open(parent/'evaluation/validation_0_trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            original_rows.setdefault(row['case_id'], []).append(row)
    total, compared = 0, 0
    hashes = {}
    for episode, case in zip(completed['episodes'], protocol['cases'], strict=True):
        assert all(episode[k] == v for k, v in case.items())
        case_id = episode['case_id']
        path = output/f'case_{case_id:03d}.jsonl.gz'
        hashes[path.name] = sha(path)
        terminal, previous, errors, queries = None, None, [], 0.
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for control, line in enumerate(stream):
                row = json.loads(line)
                assert terminal is None and row['executed_by'] == 'learner'
                if previous is not None:
                    assert row['state_before'] == previous
                np.testing.assert_array_equal(row['learner_action'], row['executed_action'])
                errors.append(np.asarray(row['learner_action'], dtype=np.float32)-np.asarray(row['teacher_action'], dtype=np.float32))
                assert row['teacher_query']['control'] == control and row['teacher_query']['teacher_decision']
                queries += row['teacher_query']['query_seconds']
                info = row['info']
                assert len(info['trace']) == info['executed_substeps'] == 1
                tick = info['trace'][0]
                assert tick['controller_status'] == 'direct'
                previous = tick['state']
                assert abs(previous['time']-(control+1)*protocol['env']['control_dt']) < 1e-8
                if case['split'] == 'validation':
                    old = original_rows[old_indices[(case['scenario'], case['seed'])]][control]
                    np.testing.assert_array_equal(row['executed_action'], old['action'])
                    for field in ('state', 'requested_command'):
                        assert tick[field] == old['info']['trace'][0][field]
                    compared += 1
                if info['termination_reason']:
                    terminal = info['episode_metrics']
        assert terminal == episode['metrics']
        assert len(errors) == episode['controls'] == episode['teacher_queries']
        assert abs(queries-episode['teacher_query_seconds']) < 1e-6
        errors = np.asarray(errors)
        np.testing.assert_allclose((errors**2).mean(0), episode['mse_longitudinal_steering'])
        np.testing.assert_allclose(np.quantile(np.abs(errors), .95, axis=0), episode['absolute_error_p95'])
        if case['split'] == 'validation':
            assert len(errors) == len(original_rows[old_indices[(case['scenario'], case['seed'])]])
        total += len(errors)
    assert total == completed['actual_learner_controls'] == completed['teacher_queries']
    result = {'verified_actual_controls': total, 'teacher_executed_controls': 0,
              'validation_controls_identical_to_original_policy_evaluation': compared, 'trace_sha256': hashes,
              'scope': 'Pure policy controls with additional teacher queries; repeated development cases, not independent confirmation'}
    (output/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(args.parent, args.output)

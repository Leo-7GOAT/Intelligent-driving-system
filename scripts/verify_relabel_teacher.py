"""Audit completed current-state labels before permitting a controlled fit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_action_only_change(old, new):
    """Reject changed visits, returns, split, metadata, shape or numeric type."""
    assert set(old) == set(new), 'Dataset fields changed'
    for key in old:
        assert old[key].dtype == new[key].dtype, f'{key} dtype changed'
        assert old[key].shape == new[key].shape, f'{key} shape changed'
        if key != 'actions':
            np.testing.assert_array_equal(old[key], new[key])
    assert np.isfinite(new['actions']).all()
    assert np.abs(new['actions']).max() <= 1


def verify(batch):
    batch = Path(batch).resolve()
    protocol, completed = read(batch/'protocol.json'), read(batch/'completed.json')
    assert completed['ppo_updates'] == completed['supervised_optimizer_steps'] == 0
    assert completed['no_new_teacher_action_executed']
    for folder, key in [('source', 'source_hashes'), ('input', 'input_hashes')]:
        for name, digest in protocol[key].items():
            assert sha(batch/folder/name) == digest
    parent = Path(protocol['parent_directory'])
    records = {(r['group'], r['case_id']): r for r in completed['episodes']}
    assert len(records) == len(completed['episodes'])
    total, seconds, trace_hashes = 0, 0., {}
    vehicle = protocol['env']['vehicle']
    for group in protocol['groups']:
        assert sha(batch/(group+'.npz')) == completed['files'][group]['sha256']
        with np.load(batch/'input'/(group+'.npz')) as old, np.load(batch/(group+'.npz')) as new:
            assert_action_only_change(old, new)
            cases = (read(batch/'input/original_protocol.json')['cases'] if group == 'original'
                     else protocol['collection_cases'])
            source_metrics = read(batch/'input'/('original_completed.json' if group == 'original'
                                               else group+'_collection.json'))['episodes']
            for case_id, case in enumerate(cases):
                record = records.pop((group, case_id))
                assert all(record[k] == v for k, v in case.items())
                indices = np.flatnonzero(old['case_id'] == case_id)
                if group == 'original':
                    executed = old['actions'][indices]
                else:
                    relative = f'round_{int(group[-1]):02d}/collection_{case_id:03d}.jsonl.gz'
                    assert sha(parent/relative) == protocol['original_trace_hashes'][relative]
                    with gzip.open(parent/relative, 'rt', encoding='utf-8') as stream:
                        executed = np.asarray([json.loads(line)['executed_action'] for line in stream], dtype=np.float32)
                assert len(executed) == len(indices)
                np.testing.assert_array_equal(np.load(batch/group/f'case_{case_id:03d}_labels.npy'), new['actions'][indices])
                path = batch/group/f'case_{case_id:03d}.jsonl.gz'
                trace_hashes[path.relative_to(batch).as_posix()] = sha(path)
                terminal, count, query_seconds, maximum, failures = None, 0, 0., 0., 0
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    for control, line in enumerate(stream):
                        row = json.loads(line)
                        assert terminal is None and control < len(indices)
                        index = indices[control]
                        assert row['control'] == control and row['data_index'] == index
                        np.testing.assert_array_equal(row['old_label'], old['actions'][index])
                        np.testing.assert_array_equal(row['new_label'], new['actions'][index])
                        np.testing.assert_array_equal(row['executed_action'], executed[control])
                        query = row['query']
                        assert query['control'] == control and query['teacher_decision']
                        command = query['requested_command']
                        label = np.array([command['acceleration']/(vehicle['max_accel'] if command['acceleration'] >= 0 else vehicle['max_decel']),
                                          command['steering']/vehicle['max_steer']], dtype=np.float32).clip(-1, 1)
                        np.testing.assert_array_equal(label, new['actions'][index])
                        assert 0 <= row['observation_max_difference'] < 1e-5
                        maximum = max(maximum, row['observation_max_difference'])
                        info = row['info']
                        assert info['executed_substeps'] == len(info['trace']) == 1
                        tick = info['trace'][0]
                        assert tick['controller_status'] == 'direct'
                        assert abs(tick['state']['time']-(control+1)*protocol['env']['control_dt']) < 1e-8
                        actual = tick['requested_command']
                        action = executed[control].clip(-1, 1)
                        assert abs(actual['acceleration']-float(action[0])*(vehicle['max_accel'] if action[0] >= 0 else vehicle['max_decel'])) < 1e-6
                        assert abs(actual['steering']-float(action[1])*vehicle['max_steer']) < 1e-6
                        if info['termination_reason']:
                            terminal = info['episode_metrics']
                        query_seconds += query['query_seconds']
                        failures += query['teacher_controller_status'] == 'failed'
                        count += 1
                assert terminal == record['actual_replay_metrics']
                expected = source_metrics[case_id]['direct_replay' if group == 'original' else 'mixture_metrics']
                for key in ('success', 'termination_reason', 'physics_steps', 'actions'):
                    assert terminal[key] == expected[key]
                assert count == len(indices) == record['controls'] == record['teacher_queries']
                assert abs(query_seconds-record['query_seconds']) < 1e-6
                assert maximum == record['max_observation_difference']
                assert failures == record['teacher_control_failures']
                differences = np.abs(new['actions'][indices]-old['actions'][indices])
                np.testing.assert_allclose(differences.mean(0), record['mean_absolute_label_difference'])
                np.testing.assert_allclose(differences.max(0), record['max_absolute_label_difference'])
                total += count
                seconds += query_seconds
    assert not records
    assert total == protocol['expected_controls'] == completed['actual_replay_controls'] == completed['teacher_queries']
    result = {'all_non_action_arrays_identical': True, 'actual_replay_controls': total,
              'teacher_queries': total, 'teacher_query_seconds': seconds, 'ppo_updates': 0,
              'raw_traces_verified': trace_hashes, 'scope': 'Recorded replay, labels, executed commands and dataset invariants; no new physical replay in this verifier'}
    (batch/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    result = verify(parser.parse_args().batch)
    print(json.dumps({k: v for k, v in result.items() if k != 'raw_traces_verified'}))

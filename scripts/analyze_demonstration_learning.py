"""Verify actual demonstration/replay costs and all fixed BC/PPO endpoints."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(dataset, batch, partial=False):
    dataset, batch = Path(dataset), Path(batch)
    dp, dc, protocol = read(dataset/'protocol.json'), read(dataset/'completed.json'), read(batch/'protocol.json')
    hashes = {}
    for root, p in ((dataset, dp), (batch, protocol)):
        for relative, value in p['source_hashes'].items():
            assert digest(root/'source'/relative) == value
    assert digest(dataset/'demonstrations.npz') == digest(batch/'data/demonstrations.npz') == protocol['data_sha256'] == dc['data_sha256']
    assert len(dc['episodes']) == len(dp['cases'])
    data = np.load(dataset/'demonstrations.npz')
    teacher_controls = replay_controls = teacher_decisions = 0
    largest_error = 0.
    for i, (episode, case) in enumerate(zip(dc['episodes'], dp['cases'])):
        assert episode['case_id'] == i and all(episode[k] == v for k, v in case.items())
        teacher, replay = [], []
        for name, collection in ((f'teacher_{i:03d}.jsonl.gz', teacher), (f'direct_replay_{i:03d}.jsonl.gz', replay)):
            path = dataset/name
            hashes[dataset.name+'/'+name] = digest(path)
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    item = json.loads(line)
                    if 'info' in item:
                        collection.extend(item['info']['trace'])
                        teacher_decisions += 1
                    else:
                        collection.append(item)
        assert len(teacher) == len(replay) == episode['teacher']['physics_steps'] == episode['direct_replay']['physics_steps']
        selected = data['case_id'] == i
        assert int(selected.sum()) == len(replay)
        actions = data['actions'][selected]
        for j, (left, right) in enumerate(zip(teacher, replay)):
            error = max(abs(left['state'][k]-right['state'][k]) for k in ('x','y','yaw','v','steering','acceleration','time'))
            largest_error = max(largest_error, error)
            assert error < 1e-3
            command, vehicle = left['requested_command'], dp['env']['vehicle']
            expected = np.array([command['acceleration']/(vehicle['max_accel'] if command['acceleration'] >= 0 else vehicle['max_decel']),
                                 command['steering']/vehicle['max_steer']], dtype=np.float32).clip(-1,1)
            np.testing.assert_array_equal(actions[j], expected)
        teacher_controls += len(teacher)
        replay_controls += len(replay)
    assert (teacher_controls, replay_controls, teacher_decisions) == (dc['teacher_controls'], dc['direct_replay_controls'], dc['teacher_decisions'])
    assert teacher_decisions == len(data['macro_actions'])
    supervision = read(batch/'supervision.json')
    train_count = sum(int(np.sum(data['case_id'] == i)) for i, c in enumerate(dp['cases']) if c['data_split'] == 'train')
    assert train_count == supervision['training_examples']
    assert abs(supervision['normalization_count']-train_count-1e-4) < 1e-6
    assert supervision['ppo_updates'] == 0 and supervision['actor_l2_change_supervised'] > 0
    completed = (batch/'completed.json').exists()
    if not partial and not completed:
        raise ValueError('Experiment incomplete; use --partial for explicitly provisional analysis')
    results = read(batch/'completed.json')['results'] if completed else read(batch/'progress.json')
    assert [r['target_ppo_controls'] for r in results] == protocol['milestones'][:len(results)]
    if completed:
        assert len(results) == len(protocol['milestones'])
    endpoints = []
    for row in results:
        target = row['target_ppo_controls']
        evaluation_path = batch/'B1'/f'validation_{target}.json'
        evaluation = read(evaluation_path)
        checkpoint = batch/'B1'/Path(evaluation['checkpoint']).name
        assert digest(checkpoint/'model.zip') == evaluation['model_sha256']
        summary = read(checkpoint/'training_summary.json')
        assert summary['physics_steps'] == row['actual_ppo_controls']
        assert summary['optimizer_steps'] == row['ppo_optimizer_steps']
        assert row['actual_ppo_controls'] >= target
        if target:
            assert row['ppo_optimizer_steps'] > 0 and row['actor_l2_change_after_supervision'] > 0
        else:
            assert row['ppo_optimizer_steps'] == row['actual_ppo_controls'] == 0
        assert len(evaluation['episodes']) == len(protocol['cases'])
        for episode, case in zip(evaluation['episodes'], protocol['cases']):
            assert all(episode.get(k) == case.get(k) for k in ('scenario','seed','label','options'))
        # Summary success alone is insufficient: retain and match every actual
        # evaluation control and terminal metric from its raw execution trace.
        trace_path = batch/'B1'/f'validation_{target}_trace.jsonl.gz'
        controls, terminal = Counter(), {}
        with gzip.open(trace_path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                case_id, info = item['case_id'], item['info']
                assert case_id not in terminal, 'Controls after termination'
                assert len(info['trace']) == info['executed_substeps']
                controls[case_id] += len(info['trace'])
                if info.get('termination_reason'):
                    terminal[case_id] = info['episode_metrics']
        assert set(terminal) == set(range(len(protocol['cases'])))
        for episode in evaluation['episodes']:
            i = episode['case_id']
            assert controls[i] == episode['physics_steps']
            assert all(episode[k] == v for k, v in terminal[i].items())
        full = [e for e in evaluation['episodes'] if 'label' not in e]
        assert sum(e['success'] for e in full) == row['full_task_success']
        assert {str(d):sum(e['success'] for e in evaluation['episodes'] if e.get('label') == f'retention_{d}m') for d in (6,12,18,23)} == row['retention']
        for path in (evaluation_path, trace_path, checkpoint/'model.zip', checkpoint/'normalization.pkl', checkpoint/'config.json'):
            hashes[batch.name+'/'+path.relative_to(batch).as_posix()] = digest(path)
        endpoints.append({**row, 'full_task_reasons': dict(Counter(e['termination_reason'] for e in full)),
            'scenario_success': {s:sum(e['success'] for e in full if e['scenario'] == s) for s in sorted({e['scenario'] for e in full})},
            'evaluation_controls': sum(e['physics_steps'] for e in evaluation['episodes'])})
    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    history = supervision['history']
    for key in ('train_action_mse','holdout_action_mse'):
        axes[0].semilogy([h['supervised_epoch'] for h in history], [h[key] for h in history], label=key)
    axes[0].set(xlabel='Supervised epochs (not PPO updates)', ylabel='Action MSE', title='Teacher-state prediction')
    axes[0].legend()
    axes[1].plot([e['actual_ppo_controls'] for e in endpoints], [e['full_task_success']/10*100 for e in endpoints], '-o')
    axes[1].set(xlabel='Additional real PPO controls', ylabel='Completion (%)', ylim=(-5,105), title='All 10 development cases')
    fig.suptitle('One demonstration-assisted seed; '+('complete' if completed else 'PROVISIONAL'))
    output = batch/'analysis'
    output.mkdir(exist_ok=True)
    fig.savefig(output/'demonstration_learning.png', dpi=160)
    plt.close(fig)
    result = {'scope': protocol['scope'], 'complete': completed, 'teacher_controls': teacher_controls,
        'direct_replay_controls': replay_controls, 'teacher_decisions': teacher_decisions, 'max_replay_error': largest_error,
        'supervised_optimizer_steps': history[-1]['supervised_optimizer_steps'], 'training_examples': train_count,
        'holdout_examples': supervision['holdout_examples'], 'endpoints': endpoints, 'input_hashes': hashes,
        'limitations': 'Extra supervised data and hold1 frequency; not comparable as pure PPO to earlier hold10 branches. All endpoints retained; not final confirmation.'}
    (output/'verified_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'input_hashes'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--batch', required=True)
    p.add_argument('--partial', action='store_true')
    args = p.parse_args()
    analyze(args.dataset, args.batch, args.partial)

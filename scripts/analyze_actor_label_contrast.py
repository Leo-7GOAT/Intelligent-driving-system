"""Verify a fixed-initialization actor-only label comparison, retaining all cases."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from navstack.experiments import load_run, actor_vector
from scripts.verify_relabel_teacher import assert_action_only_change


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_endpoint(batch):
    protocol, completed = read(batch/'protocol.json'), read(batch/'completed.json')
    guard = read(batch/'guard_final.json')
    assert guard['exit_code'] == 0 and guard['reason'] is None and not (batch/'failure.json').exists()
    assert protocol['value_weight'] == 0 and protocol['epochs'] == 200 and protocol['seed'] == 13
    for folder, key in [('source', 'source_hashes'), ('initial', 'initial_hashes'), ('data', 'data_hashes')]:
        for name, digest in protocol[key].items():
            assert sha(batch/folder/name) == digest
    fit = read(batch/'supervision.json')
    initial, initial_norm = load_run(batch/'initial')
    model, norm = load_run(batch/'supervised_checkpoint')
    try:
        assert sha(batch/'supervised_checkpoint/model.zip') == fit['model_sha256']
        for name, value in initial.policy.named_parameters():
            if 'value_net' in name or name == 'log_std':
                assert torch.equal(value, dict(model.policy.named_parameters())[name])
        change = float(torch.linalg.vector_norm(actor_vector(model)-actor_vector(initial)))
        assert change > 0 and abs(change-fit['actor_l2_change']) < 1e-6
        assert model._n_updates == 0 and not model.policy.optimizer.state
        np.testing.assert_array_equal(initial_norm.obs_rms.mean, norm.obs_rms.mean)
        np.testing.assert_array_equal(initial_norm.obs_rms.var, norm.obs_rms.var)
        assert initial_norm.obs_rms.count == norm.obs_rms.count
        assert fit['history'][-1]['supervised_epoch'] == protocol['epochs']
        assert fit['history'][-1]['supervised_optimizer_steps'] == completed['supervised_optimizer_steps'] == 19400
        assert completed['ppo_updates'] == fit['ppo_updates'] == 0
    finally:
        initial_norm.close()
        norm.close()
    evaluation = read(batch/'evaluation/validation_0.json')
    assert evaluation['model_sha256'] == fit['model_sha256']
    counts, terminals = Counter(), {}
    trace_path = batch/'evaluation/validation_0_trace.jsonl.gz'
    vehicle = protocol['env']['vehicle']
    with gzip.open(trace_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            case_id, info = row['case_id'], row['info']
            assert case_id not in terminals
            assert info['executed_substeps'] == len(info['trace']) == 1
            tick = info['trace'][0]
            assert tick['controller_status'] == 'direct'
            action = np.asarray(row['action']).clip(-1, 1)
            assert abs(tick['requested_command']['acceleration']-float(action[0])*(vehicle['max_accel'] if action[0] >= 0 else vehicle['max_decel'])) < 1e-6
            assert abs(tick['requested_command']['steering']-float(action[1])*vehicle['max_steer']) < 1e-6
            counts[case_id] += 1
            assert abs(tick['state']['time']-counts[case_id]*protocol['env']['control_dt']) < 1e-8
            if info['termination_reason']:
                terminals[case_id] = info['episode_metrics']
    assert len(evaluation['episodes']) == len(protocol['cases']) == len(terminals)
    for i, (episode, case) in enumerate(zip(evaluation['episodes'], protocol['cases'], strict=True)):
        assert all(episode.get(k) == case.get(k) for k in ('scenario', 'seed', 'label', 'options'))
        assert all(episode[k] == value for k, value in terminals[i].items())
        assert counts[i] == episode['physics_steps']
    full = [r for r in evaluation['episodes'] if 'label' not in r]
    assert sum(r['success'] for r in full) == completed['full_task_success']
    assert sum(counts.values()) == completed['evaluation_controls']
    retention = {str(d): sum(r['success'] for r in evaluation['episodes'] if r.get('label') == f'retention_{d}m')
                 for d in (6, 12, 18, 23)}
    assert retention == completed['retention']
    return {'batch': batch.name, 'model_sha256': fit['model_sha256'], 'trace_sha256': sha(trace_path),
            'full_task_success': completed['full_task_success'], 'full_task_cases': len(full),
            'reasons': dict(Counter(r['termination_reason'] for r in full)),
            'retention': completed['retention'], 'supervised_optimizer_steps': 19400, 'actor_l2_change': change,
            'critic_and_normalization_unchanged': True, 'ppo_updates': 0, 'evaluation_controls': sum(counts.values()),
            'episodes': [{k: r[k] for k in ('scenario', 'seed', 'success', 'termination_reason', 'physics_steps')} | {'label': r.get('label')}
                         for r in evaluation['episodes']]}


def analyze(batch, control):
    batch, control = Path(batch), Path(control)
    protocol, control_protocol = read(batch/'protocol.json'), read(control/'protocol.json')
    assert protocol['control_batch'] == control.name
    assert protocol['initial_hashes'] == control_protocol['initial_hashes']
    assert protocol['env'] == control_protocol['env'] and protocol['cases'] == control_protocol['cases']
    for name in ('original.npz', 'round1.npz', 'round2.npz'):
        with np.load(control/'data'/name) as old, np.load(batch/'data'/name) as new:
            assert_action_only_change(old, new)
    endpoints = [verify_endpoint(control), verify_endpoint(batch)]
    result = {'scope': 'One fixed-seed actor-only label contrast; no independent-seed interval or hierarchy claim',
              'only_direct_labels_changed': True, 'endpoints': endpoints,
              'additional_label_costs': protocol['additional_label_costs'],
              'full_task_success_count_difference': endpoints[1]['full_task_success']-endpoints[0]['full_task_success']}
    output = batch/'contrast_analysis'
    output.mkdir(exist_ok=True)
    (output/'verified_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'endpoints'}))
    print(json.dumps([{k: v for k, v in row.items() if k != 'episodes'} for row in endpoints]))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--control', required=True)
    args = parser.parse_args()
    analyze(args.batch, args.control)

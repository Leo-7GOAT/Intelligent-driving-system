"""Audit actual PPO updates with the shared parking wrapper fixed throughout."""
import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha
from scripts.audit_native_control_delivery import analyze as audit_delivery
from scripts.analyze_native_feedback_ppo import adam_step_increments
from scripts.analyze_public_native_replay import parking_trace
from scripts.evaluate_native_rule import require_finished_parent


def analyze(batch):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    batch = Path(batch).resolve()
    require_finished_parent(batch)
    folder = batch / 'B1'
    p = read(batch / 'protocol.json')
    spec = p['shared_parking_PPO']
    raw = read(folder / 'endpoint_raw_audit.json')
    summary = read(folder / 'summary.json')
    evaluation = read(folder / 'evaluation.json')
    inference = read(folder / 'measured_start_inference.json')
    if (summary['physics_steps'] != spec['new_PPO_controls'] or summary['stopped_reason'] is not None
            or raw['training_control_frames_verified'] != summary['physics_steps']
            or raw['protocol_sha256'] != sha(batch / 'protocol.json')
            or sha(batch / 'resume/B1/model.zip') != spec['source_model_sha256']):
        raise ValueError('Require the complete registered actual training course')
    for name, digest in p['source_hashes'].items():
        if sha(batch / 'source' / name) != digest:
            raise ValueError('Frozen source changed during learning')
    initial = load_ppo(batch / 'resume/B1/model.zip', device='cpu')
    final = load_ppo(folder / 'model.zip', device='cpu')
    if final.num_timesteps - initial.num_timesteps != summary['physics_steps']:
        raise ValueError('Actual PPO counter differs from measured training controls')
    adam = adam_step_increments(initial.policy.optimizer.state_dict()['state'], final.policy.optimizer.state_dict()['state'], summary['optimizer_steps'])
    changed = {'actor': 0, 'critic': 0}
    for key, value in initial.policy.state_dict().items():
        if not torch.equal(value, final.policy.state_dict()[key]):
            changed['critic' if 'value_net' in key else 'actor'] += 1
    if not all(changed.values()):
        raise ValueError('Both networks must actually update')
    with (batch / 'resume/B1/normalization.pkl').open('rb') as stream:
        before = pickle.load(stream)
    with (folder / 'normalization.pkl').open('rb') as stream:
        after = pickle.load(stream)
    np.testing.assert_array_equal(before.obs_rms.mean, after.obs_rms.mean)
    np.testing.assert_array_equal(before.obs_rms.var, after.obs_rms.var)
    if before.obs_rms.count != after.obs_rms.count:
        raise ValueError('Frozen observation count changed')
    if (sha(folder / 'model.zip') != evaluation['model_sha256'] or inference['model_sha256'] != evaluation['model_sha256']
            or len(inference['results']) != 4 or inference['batch_protocol_sha256'] != sha(batch / 'protocol.json')):
        raise ValueError('Endpoint inference is not bound to the actual model and protocol')
    delivery = audit_delivery(batch)
    if not delivery['strict_delivery_gate_passed']:
        raise ValueError('Every physical engine control must match')
    episodes = evaluation['episodes']
    parking = parking_trace(batch, [0, 1, 2, 3])
    successes = sum(e['success'] for e in episodes)
    unsafe = sum(e['termination_reason'] in ('collision', 'out_of_bounds') for e in episodes)
    return {'scope': __doc__, 'script_sha256': sha(Path(__file__)), 'batch': batch.name,
            'protocol_sha256': sha(batch / 'protocol.json'),
            'initial_model_sha256': spec['source_model_sha256'], 'final_model_sha256': evaluation['model_sha256'],
            'final_normalization_sha256': evaluation['normalization_sha256'],
            'new_actual_PPO_controls': summary['physics_steps'], 'new_actual_Adam_steps': summary['optimizer_steps'],
            'saved_Adam_proof': adam, 'changed_network_tensors': changed,
            'new_PPO_epoch_updates': final._n_updates - initial._n_updates,
            'new_supervised_updates': 0, 'observation_RMS_mean_variance_count_exact': True,
            'all_engine_controls_acknowledged': True, 'delivery_by_phase': delivery['by_phase'],
            'all_actual_raw_controls': raw['total_applied_controls_verified'], 'parking': parking,
            'training_terminal_outcomes': raw['training_termination_reasons'],
            'endpoint': {'successes': successes, 'cases': 4, 'unsafe_cases': unsafe,
                         'basic_gate_passed': successes >= 3 and unsafe == 0},
            'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'),
            'limits': spec['limits'], 'research_acceptance': 'UNMET'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.batch)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in ('new_actual_PPO_controls', 'new_actual_Adam_steps', 'endpoint')}), flush=True)

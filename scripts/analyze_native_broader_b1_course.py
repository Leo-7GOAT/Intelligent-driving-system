"""Verify actual broader-distribution B1 PPO and every existing validation condition."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha
from scripts.analyze_native_feedback_ppo import adam_step_increments
from scripts.analyze_native_shared_parking_ppo import training_parking_trace
from scripts.analyze_public_native_replay import parking_trace
from scripts.audit_native_control_delivery import analyze as delivery_audit
from scripts.evaluate_native_rule import require_finished_parent


def analyze(batch):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    batch = Path(batch).resolve()
    require_finished_parent(batch)
    p = read(batch / 'protocol.json')
    spec = p['broader_B1_PPO']
    budget = spec['new_actual_PPO_controls']
    recovery = p.get('broader_course_recovery')
    if budget != 20480:
        if (not recovery or type(budget) is not int or not 0 < budget < 20480
                or recovery['saved_original_controls'] + budget != 20480
                or recovery['resumed_controls'] != budget):
            raise ValueError('A reduced segment requires an explicit original-budget recovery')
        parent = batch.parent / p['parent_batch']
        prefix = read(parent / 'verified_saved_prefix.json')
        if (sha(parent / 'verified_saved_prefix.json') != recovery['saved_prefix_sha256']
                or prefix['saved_controls'] != recovery['saved_original_controls']
                or prefix['verified_controls'] < prefix['saved_controls']
                or prefix['all_engine_controls_acknowledged'] is not True
                or sha(parent / 'protocol.json') != recovery['original_protocol_sha256']):
            raise ValueError('Original saved controls lack hash-bound physical evidence')
    folder = batch / 'B1'
    raw, summary, evaluation = (read(folder / n) for n in ('endpoint_raw_audit.json', 'summary.json', 'evaluation.json'))
    starts = read(folder / 'measured_start_inference.json')
    if (summary['physics_steps'] != budget or summary['stopped_reason'] is not None
            or raw['training_control_frames_verified'] != budget or raw['protocol_sha256'] != sha(batch / 'protocol.json')
            or len(p['native_case_manifest']['cases']) != 36 or len(p['native_validation']['manifest']['cases']) != 24
            or sorted(e['case_id'] for e in evaluation['episodes']) != list(range(24))
            or sorted(e['case_id'] for e in starts['results']) != list(range(24))
            or starts['batch_protocol_sha256'] != sha(batch / 'protocol.json')):
        raise ValueError('Require the complete actual budget and every existing validation condition')
    for name, digest in p['source_hashes'].items():
        if sha(batch / 'source' / name) != digest:
            raise ValueError('Frozen source changed')
    if sha(batch / 'resume/B1/model.zip') != spec['source_model_sha256'] or sha(folder / 'model.zip') != evaluation['model_sha256']:
        raise ValueError('Registered learned endpoints changed')
    initial = load_ppo(batch / 'resume/B1/model.zip', device='cpu')
    final = load_ppo(folder / 'model.zip', device='cpu')
    if final.num_timesteps - initial.num_timesteps != budget:
        raise ValueError('Actual PPO counter differs')
    adam = adam_step_increments(initial.policy.optimizer.state_dict()['state'], final.policy.optimizer.state_dict()['state'], summary['optimizer_steps'])
    changed = {'actor': 0, 'critic': 0}
    for name, value in initial.policy.state_dict().items():
        if not torch.equal(value, final.policy.state_dict()[name]):
            changed['critic' if 'value_net' in name else 'actor'] += 1
    if not all(changed.values()):
        raise ValueError('Actor and critic must actually change')
    with (batch / 'resume/B1/normalization.pkl').open('rb') as stream:
        a = pickle.load(stream)
    with (folder / 'normalization.pkl').open('rb') as stream:
        b = pickle.load(stream)
    np.testing.assert_array_equal(a.obs_rms.mean, b.obs_rms.mean)
    np.testing.assert_array_equal(a.obs_rms.var, b.obs_rms.var)
    if a.obs_rms.count != b.obs_rms.count or not delivery_audit(batch)['strict_delivery_gate_passed']:
        raise ValueError('Frozen observation statistics or actual command acknowledgements differ')
    groups = {}
    for e in evaluation['episodes']:
        hazard = p['native_validation']['manifest']['cases'][e['case_id']]['hazard']
        groups.setdefault('curve_parking' if hazard is None else hazard['kind'], []).append(e)
    return {'scope': __doc__, 'script_sha256': sha(Path(__file__)), 'batch': batch.name,
        'protocol_sha256': sha(batch / 'protocol.json'), 'initial_model_sha256': spec['source_model_sha256'],
        'final_model_sha256': evaluation['model_sha256'], 'normalization_sha256': evaluation['normalization_sha256'],
        'new_actual_PPO_controls': budget, 'new_actual_Adam_steps': summary['optimizer_steps'],
        'saved_Adam_proof': adam, 'changed_network_tensors': changed, 'observation_RMS_exact': True,
        'all_actual_control_frames_verified': raw['total_applied_controls_verified'], 'all_engine_controls_acknowledged': True,
        'training_parking': training_parking_trace(batch, budget), 'validation_parking': parking_trace(batch, list(range(24))),
        'training_outcomes': raw['training_termination_reasons'], 'validation_successes': sum(e['success'] for e in evaluation['episodes']),
        'validation_cases': 24, 'validation_termination_counts': dict(Counter(e['termination_reason'] for e in evaluation['episodes'])),
        'by_core': {k: {'cases': len(v), 'successes': sum(e['success'] for e in v),
            'termination_counts': dict(Counter(e['termination_reason'] for e in v))} for k, v in groups.items()},
        'lineage_PPO_controls': summary['lineage_control_steps'], 'training_wall_seconds': summary['wall_seconds'],
        'measured_start_actions_verified': 24, 'new_final_conditions': 0,
        'limits': spec['limits'], **({'explicit_recovery': recovery} if recovery else {}), 'research_acceptance': 'UNMET'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.batch)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('new_actual_PPO_controls', 'new_actual_Adam_steps', 'validation_successes', 'by_core')}), flush=True)

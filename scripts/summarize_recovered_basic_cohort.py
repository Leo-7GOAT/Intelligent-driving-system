"""Retain all five basic-course outcomes, including the fifth seed's interrupted recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(original, recovery, recovery_proof):
    original, recovery = Path(original), Path(recovery)
    prepared = read(original / 'prepared_courses.json')['batches']
    completed = read(recovery / 'completed_five_courses.json')
    proof = read(recovery_proof)
    if ([r['seed'] for r in prepared] != [11, 22, 33, 44, 55]
            or [r['seed'] for r in completed['original_four']] != [11, 22, 33, 44]
            or not completed['interruption_disclosed']):
        raise ValueError('Require all five registered seeds; never select only passing gates')
    records, routes, inputs = [], None, []
    for row in prepared:
        seed = row['seed']
        recovered = seed == 55
        path = recovery / 'learning_audit.json' if recovered else original / f'seed{seed}_learning_audit.json'
        result = read(path)
        batch = ROOT / 'runs' / result['batch']
        p = read(batch / 'protocol.json')
        whole_path = batch / 'B1/whole_sequence_inference.json'
        whole = read(whole_path)
        if routes is None:
            routes = p['routes']
        if (p['routes'] != routes or result['seed'] != seed
                or sha(batch / 'protocol.json') != result['protocol_sha256']
                or sha(batch / 'B1/model.zip') != result['final_model_sha256']
                or sha(batch / 'B1/normalization.pkl') != result['final_normalization_sha256']
                or whole['model_sha256'] != result['final_model_sha256']
                or whole['normalization_sha256'] != result['final_normalization_sha256']
                or whole['protocol_sha256'] != result['protocol_sha256']
                or whole['raw_audit_sha256'] != result['raw_audit_sha256']
                or whole['total_bit_exact_actions'] != sum(c['controls'] for c in result['parking']['cases'])
                or sorted(e['case_id'] for e in result['endpoint_case_metrics']) != [0, 1, 2, 3]
                or not result['observation_RMS_mean_variance_count_exact']
                or not result['all_engine_controls_acknowledged']):
            raise ValueError('Completed course must retain the full model/raw/action audit chain')
        if not recovered:
            receipt = next(r for r in completed['original_four'] if r['seed'] == seed)
            if (sha(path) != receipt['learning_audit_sha256'] or result['new_actual_PPO_controls'] != 10240
                    or result['initial_model_sha256'] != row['initial_model_sha256']):
                raise ValueError('Original completed course changed')
            effective, consumed, adam = 10240, 10240, result['new_actual_Adam_steps']
        else:
            prefix = proof['original_prefix_verification']
            if (proof['parent'] != row['batch'] or proof['batch'] != batch.name
                    or not prefix['saved_3840_control_training_fully_inside_verified_prefix']
                    or result['new_actual_PPO_controls'] != 6400 or result['lineage_controls'] != 30720
                    or p['basic_course_recovery']['saved_original_controls'] != 3840
                    or result['initial_model_sha256'] != p['recovery']['B1']['model_sha256']):
                raise ValueError('Fifth-seed recovery lineage differs')
            effective = proof['saved_controls'] if 'saved_controls' in proof else proof['saved_original_controls']
            effective += result['new_actual_PPO_controls']
            consumed = proof['verified_original_consumed_controls_lower_bound'] + result['new_actual_PPO_controls']
            adam = prefix['saved_Adam_increment_proof']['step_increment_each'] + result['new_actual_Adam_steps']
        records.append({'seed': seed, 'original_batch': row['batch'], 'endpoint_batch': batch.name,
            'interrupted_and_recovered': recovered, 'effective_new_PPO_controls': effective,
            'consumed_new_training_controls_lower_bound': consumed, 'effective_new_Adam_steps': adam,
            'endpoint': result['endpoint'], 'model_sha256': result['final_model_sha256'],
            'whole_endpoint_actions_exact': whole['total_bit_exact_actions'],
            'full_training_lineage_controls': result['lineage_controls'],
            'charged_lineage_controls_lower_bound': result['charged_training_controls_lower_bound']})
        inputs.append({'seed': seed, 'learning_audit_sha256': sha(path), 'whole_action_audit_sha256': sha(whole_path)})
    return {'scope': __doc__, 'script_sha256': sha(__file__), 'seeds': records, 'input_audits': inputs,
        'recovery_registration_proof_sha256': sha(recovery_proof), 'independent_warm_started_seeds': 5,
        'unique_known_basic_conditions': 4, 'policy_case_evaluations': 20,
        'endpoint_successes': sum(r['endpoint']['successes'] for r in records),
        'unsafe_endpoint_cases': sum(r['endpoint']['unsafe_cases'] for r in records),
        'qualified_basic_seeds': sum(r['endpoint']['basic_gate_passed'] for r in records),
        'all_five_basic_gates_passed': all(r['endpoint']['basic_gate_passed'] for r in records),
        'mean_seed_success_fraction': statistics.fmean(r['endpoint']['successes']/4 for r in records),
        'effective_new_PPO_controls': sum(r['effective_new_PPO_controls'] for r in records),
        'consumed_new_training_controls_lower_bound': sum(r['consumed_new_training_controls_lower_bound'] for r in records),
        'bridge_verified_new_training_controls_lower_bound': sum(r['consumed_new_training_controls_lower_bound'] for r in records) - proof['original_prefix_verification']['trace_tail_controls_without_readable_bridge_ack'],
        'reported_trace_tail_without_readable_bridge_ack': proof['original_prefix_verification']['trace_tail_controls_without_readable_bridge_ack'],
        'effective_new_Adam_steps': sum(r['effective_new_Adam_steps'] for r in records),
        'new_final_conditions': 0, 'unknown_unflushed_seed55_tail': True,
        'limits': 'Four repeated known TRAIN roads, not20unique conditions or a full-task healthy baseline. Fifth seed restarts a physical episode from its saved RNG/checkpoint and is not an uninterrupted replicate. Discarded work retained. No budget-matched H/B1 superiority or added PPO claim.',
        'research_acceptance': 'UNMET'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', default='runs/B1_shared_parking_five_seed_execution_20261101')
    parser.add_argument('--recovery', default='runs/B1_basic_seed55_recovery_execution_20261102')
    parser.add_argument('--recovery-proof', default='reports/native_B1_seed55_recovery_registration_20261102.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Preserve completed cohort evidence')
    result = summarize(args.original, args.recovery, args.recovery_proof)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('endpoint_successes', 'qualified_basic_seeds', 'effective_new_PPO_controls')}))

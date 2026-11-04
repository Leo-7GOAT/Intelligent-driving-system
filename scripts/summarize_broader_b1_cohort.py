"""Rebuild all registered broader B1 outcomes, with explicit interruption costs."""
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


def summarize(workspace, execution, allow_partial=False):
    workspace, execution = Path(workspace), Path(execution)
    original = workspace / 'runs/B1_additional_broader_cohort_20261102'
    pilot = workspace / 'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO'
    registration = read(execution / 'registration.json')
    if [r['seed'] for r in registration['rows']] != [11, 33, 44, 55]:
        raise ValueError('Require every registered independent broader parent')
    completed = {r['seed']: r for r in read(execution / 'status.json')['completed']}
    records, pending, inputs = [], [], []
    reference = read(pilot / 'protocol.json')
    if len(reference['native_case_manifest']['cases']) != 36 or len(reference['native_validation']['manifest']['cases']) != 24:
        raise ValueError('Require the registered36TRAIN/24knownVAL inventory')
    for seed in (11, 22, 33, 44, 55):
        if seed == 22:
            audit = workspace / 'reports/native_B1_broader_seed22_20261102.json'
            batch = pilot
        else:
            row = next(r for r in registration['rows'] if r['seed'] == seed)
            if seed not in completed:
                if not allow_partial:
                    raise ValueError('Full five-seed outcome incomplete: ' + str(seed))
                pending.append(seed)
                continue
            if completed[seed]['batch'] != row['batch']:
                raise ValueError('Completed batch differs from its registered seed')
            audit = (original if seed in (11, 33) else execution) / f'seed{seed}_learning_audit.json'
            if sha(audit) != completed[seed]['audit_sha256']:
                raise ValueError('Completed learning audit changed')
            batch = workspace / 'runs' / row['batch']
        result, protocol = read(audit), read(batch / 'protocol.json')
        summary = read(batch / 'B1/summary.json')
        for name in ('routes', 'native_case_manifest', 'native_validation', 'calibration_sha256'):
            if protocol[name] != reference[name]:
                raise ValueError('Common physical TRAIN/VAL inventory or calibration differs')
        for name, digest in reference['broader_B1_PPO']['preserved_low_level_source_hashes'].items():
            if protocol['source_hashes'][name] != digest or sha(batch / 'source' / name) != digest:
                raise ValueError('Shared low-level source differs')
        if (protocol['seed'] != seed or sha(batch / 'protocol.json') != result['protocol_sha256']
                or sha(batch / 'B1/model.zip') != result['final_model_sha256']
                or sha(batch / 'B1/normalization.pkl') != result['normalization_sha256']
                or result['lineage_PPO_controls'] != 51200 or summary['lineage_control_steps'] != 51200
                or result['validation_cases'] != 24 or result['measured_start_actions_verified'] != 24
                or not result['all_engine_controls_acknowledged'] or not result['observation_RMS_exact']
                or result['saved_Adam_proof']['parameter_states'] != 35
                or result['saved_Adam_proof']['step_increment_each'] != result['new_actual_Adam_steps']
                or read(batch / 'guard_final.json')['exit_code'] != 0
                or read(batch / 'guard_final.json').get('reason') is not None
                or read(batch / 'summary.json')['status'] != 'complete'
                or read(batch / 'infrastructure/server_stop.json')['status'] != 'owned_empty_server_stopped'):
            raise ValueError('Require complete physical, model, normalization and optimizer evidence')
        controls = result['new_actual_PPO_controls']
        adam = result['new_actual_Adam_steps']
        raw_controls = result['all_actual_control_frames_verified']
        recovered = 'explicit_recovery' in result
        if recovered:
            recovery = result['explicit_recovery']
            parent = workspace / 'runs' / protocol['parent_batch']
            prefix_path = parent / 'verified_saved_prefix.json'
            prefix = read(prefix_path)
            if (seed != 44 or sha(prefix_path) != recovery['saved_prefix_sha256']
                    or sha(parent / 'protocol.json') != recovery['original_protocol_sha256']
                    or prefix['saved_controls'] != recovery['saved_original_controls']
                    or prefix['saved_Adam']['step_increment_each'] != recovery['saved_original_Adam_steps']
                    or prefix['saved_Adam']['parameter_states'] != 35
                    or prefix['verified_controls'] < prefix['saved_controls']
                    or prefix['all_engine_controls_acknowledged'] is not True):
                raise ValueError('Interrupted original prefix lacks the registered saved-state proof')
            controls += prefix['saved_controls']
            adam += prefix['saved_Adam']['step_increment_each']
            raw_controls += prefix['verified_controls']
            inputs.append(dict(seed=seed, original_prefix_sha256=sha(prefix_path)))
        if controls != 20480:
            raise ValueError('Never call a partial training segment a complete broader course')
        terminations = result['validation_termination_counts']
        successes = result['validation_successes']
        unsafe = terminations.get('collision', 0) + terminations.get('out_of_bounds', 0)
        cores = result['by_core']
        if (set(cores) != {'curve_parking', 'static_vehicle', 'crossing_walker'}
                or sum(terminations.values()) != 24 or terminations.get('success', 0) != successes
                or any(c['cases'] != 8 for c in cores.values())
                or sum(c['successes'] for c in cores.values()) != successes):
            raise ValueError('All paired development cases and failures must remain present')
        combined = {}
        for core in cores.values():
            if sum(core['termination_counts'].values()) != 8 or core['termination_counts'].get('success', 0) != core['successes']:
                raise ValueError('Core case outcomes differ')
            for reason, count in core['termination_counts'].items():
                combined[reason] = combined.get(reason, 0) + count
        if combined != terminations:
            raise ValueError('Core failures do not sum to the reported overall outcomes')
        passed = successes / 24 >= .8 and unsafe / 24 <= .05 and all(c['successes'] / 8 >= .7 for c in cores.values())
        records.append(dict(seed=seed, batch=batch.name, interrupted_and_recovered=recovered,
            effective_new_PPO_controls=controls, effective_new_Adam_steps=adam,
            all_bridge_controls_verified=raw_controls, validation_successes=successes,
            validation_cases=24, unsafe_cases=unsafe, termination_counts=terminations, by_core=cores,
            development_screen_passed=passed, model_sha256=result['final_model_sha256'],
            lineage_PPO_controls=51200, inherited_unknown_unflushed_tail=bool(protocol['recovery']['B1'].get('unknown_unflushed_parent_tail', False))))
        inputs.append(dict(seed=seed, learning_audit_sha256=sha(audit), protocol_sha256=sha(batch / 'protocol.json')))
    return dict(scope=__doc__, script_sha256=sha(__file__), registered_seeds=[11, 22, 33, 44, 55],
        complete_five_seed_cohort=not pending, completed_seeds=len(records), pending_seeds=pending,
        seeds=records, input_audits=inputs, unique_known_validation_conditions=24,
        policy_condition_evaluations=24 * len(records), endpoint_successes=sum(r['validation_successes'] for r in records),
        unsafe_cases=sum(r['unsafe_cases'] for r in records),
        mean_completed_seed_success_fraction=statistics.fmean(r['validation_successes'] / 24 for r in records),
        qualified_development_seeds=sum(r['development_screen_passed'] for r in records),
        effective_new_PPO_controls=sum(r['effective_new_PPO_controls'] for r in records),
        effective_new_Adam_steps=sum(r['effective_new_Adam_steps'] for r in records),
        new_final_conditions=0, research_acceptance='UNMET',
        limits='Known24development conditions reused across independent warm starts;not120new final conditions,from-scratch learning,matched H/B1budgets or isolated PPOvalue. Recovered44uses a new measured episode;all interruptions and failures retained. Partial means describe completed seeds only.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', default='runs/B1_broader_disk_recovery_execution_20261102')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(ROOT, args.execution, args.allow_partial)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('complete_five_seed_cohort', 'completed_seeds', 'pending_seeds', 'endpoint_successes')}))

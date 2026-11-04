"""Compare completed constant-rate native B1 PPO probes from identical weights."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from navstack.independent_clip_ppo import load_ppo
from scripts.audit_native_method_endpoint import audit, read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_supervised_ppo import representation_inventory


def analyze(baseline, probe, output):
    output = Path(output)
    if output.exists():raise FileExistsError('Preserve previous comparison')
    protocols, configs, learners, initializers, rows, evidence = [], [], [], [], {}, {}
    torch.set_num_threads(1)
    for label, batch in [('baseline', Path(baseline)), ('probe', Path(probe))]:
        require_finished_parent(batch)
        protocol = read(batch/'protocol.json')
        folder = batch/'B1'
        raw = audit(batch, 'B1')
        if raw != read(folder/'endpoint_raw_audit.json'):raise ValueError('Raw audit differs')
        summary, evaluation = read(folder/'summary.json'), read(folder/'evaluation.json')
        if (protocol['methods'] != ['B1'] or protocol.get('evaluation_only')
                or not protocol.get('freeze_observation_statistics') or protocol.get('own_success_replay')
                or summary['physics_steps'] != 20480 or summary['stopped_reason'] is not None):
            raise ValueError('Require complete equal-budget frozen-statistics PPO without auxiliary replay')
        for name, key in [('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256')]:
            if sha(folder/name) != evaluation[key]:raise ValueError('Evaluated model artifact differs')
        _, episodes, count = representation_inventory(protocol, evaluation)
        if count != 12:raise ValueError('Require all twelve development cases')
        updates = [json.loads(line) for line in (folder/'optimization.jsonl').read_text().splitlines()]
        if ([row['physics_steps'] for row in updates] != list(range(1280, 20481, 1280))
                or updates[-1]['optimizer_steps'] != summary['optimizer_steps']):
            raise ValueError('Actual update/control boundaries differ')
        origin = batch/protocol['recovery']['B1']['checkpoint']
        metadata = read(origin/'metadata.json')
        for name, key in [('model.zip', 'model_sha256'), ('normalization.pkl', 'normalization_sha256'),
                          ('resume_state.pkl', 'resume_state_sha256')]:
            if sha(origin/name) != metadata[key]:raise ValueError('Initializer differs')
        initial = load_ppo(origin/'model.zip', device='cpu')
        if initial.num_timesteps or initial._n_updates or initial.policy.optimizer.state:
            raise ValueError('Require a zero-PPO initialization')
        with (origin/'normalization.pkl').open('rb') as stream:normalizer = pickle.load(stream)
        with (folder/'normalization.pkl').open('rb') as stream:final_normalizer = pickle.load(stream)
        for key in ('mean', 'var', 'count'):
            np.testing.assert_array_equal(getattr(normalizer.obs_rms, key), getattr(final_normalizer.obs_rms, key))
        with zipfile.ZipFile(folder/'model.zip') as archive:learner = json.loads(archive.read('data'))
        if learner['num_timesteps'] != 20480:raise ValueError('Saved learner counter differs')
        families = {}
        for index, case in enumerate(protocol['native_case_manifest']['cases']):
            kind = case.get('hazard', {}).get('kind') if case.get('hazard') else None
            families[index] = {None: 'curve_parking', 'static_vehicle': 'static', 'crossing_walker': 'dynamic'}[kind]
        evidence[label] = {'batch': batch.name, 'successes': sum(row['success'] for row in episodes),
            'termination_counts': dict(Counter(row['termination_reason'] for row in episodes)),
            'unsafe_episodes': sum(row['termination_reason'] in ('collision', 'out_of_bounds') for row in episodes),
            'core_successes': {family: sum(row['success'] for row in episodes if families[row['case_id']] == family)
                               for family in ('curve_parking', 'static', 'dynamic')},
            'training_controls': summary['physics_steps'], 'optimizer_steps': summary['optimizer_steps'],
            'verified_controls': raw['total_applied_controls_verified'], 'learning_rate': learner['learning_rate'],
            'charged_input_controls_lower_bound': protocol['recovery']['B1']['consumed_controls_lower_bound'],
            'model_sha256': evaluation['model_sha256'], 'raw_audit_sha256': sha(folder/'endpoint_raw_audit.json'),
            'normalization_RMS_frozen_exact': True}
        initializers.append((initial, origin))
        protocols.append(protocol);configs.append(read(folder/'config.json')['env']);learners.append(learner)
        rows[label] = {row['case_id']: row for row in episodes}
    for key in ('seed', 'routes', 'native_case_manifest', 'road_geometry_version', 'calibration_sha256',
                'PPO_reward_units', 'approach_initialization'):
        if protocols[0][key] != protocols[1][key]:raise ValueError('Matched protocol differs: '+key)
    for name, value in protocols[0]['source_hashes'].items():
        if name.startswith('navstack/') or name == 'scripts/native_road_learning.py':
            if protocols[1]['source_hashes'].get(name) != value:raise ValueError('Physical/worker source differs')
    if configs[0] != configs[1]:raise ValueError('Environment differs')
    for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef',
                'max_grad_norm', 'normalize_advantage', 'target_kl'):
        if learners[0][key] != learners[1][key]:raise ValueError('Another PPO parameter differs: '+key)
    for name, parameter in initializers[0][0].policy.state_dict().items():
        if not torch.equal(parameter, initializers[1][0].policy.state_dict()[name]):raise ValueError('Initial weights differ')
    for name in ('normalization.pkl', 'resume_state.pkl'):
        if (initializers[0][1]/name).read_bytes() != (initializers[1][1]/name).read_bytes():
            raise ValueError('Initial normalization/RNG bytes differ')
    if set(rows['baseline']) != set(rows['probe']):raise ValueError('Case set differs')
    wins = [i for i in sorted(rows['probe']) if rows['probe'][i]['success'] and not rows['baseline'][i]['success']]
    losses = [i for i in sorted(rows['baseline']) if rows['baseline'][i]['success'] and not rows['probe'][i]['success']]
    report = {'scope': __doc__, 'analyzer_sha256': sha(Path(__file__)), 'seed': protocols[0]['seed'],
        'cases': 12, 'arms': evidence, 'initial_policy_weights_exact': True, 'initial_normalization_RNG_bytes_exact': True,
        'physical_and_worker_source_exact': True, 'probe_only_success_cases': wins, 'baseline_only_success_cases': losses,
        'probe_minus_baseline_percentage_points': 100*(len(wins)-len(losses))/12,
        'probe_safety_target_met': evidence['probe']['unsafe_episodes']/12 <= .05,
        'limits': 'One independent training seed on correlated known development conditions;actual Adam counts differ under KL stopping. New native trajectories differ slightly. No final-test or independent-seed significance claim.',
        'new_audit_controls': 0, 'new_audit_optimizer_steps': 0, 'research_acceptance': 'UNMET'}
    with output.open('x', encoding='utf-8') as stream:json.dump(report, stream, indent=2)
    print(json.dumps(report));return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'probe', 'output'):parser.add_argument('--'+name, required=True)
    args = parser.parse_args();analyze(args.baseline, args.probe, args.output)

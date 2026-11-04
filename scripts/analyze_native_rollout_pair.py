"""Audit 32/256 native PPO forks without treating correlated probes as seeds."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_native_gradient_pair import read, sha, initial_cases
from scripts.audit_native_method_endpoint import audit

FIXED_PARAMETERS = ('batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'learning_rate',
    'ent_coef', 'vf_coef', 'max_grad_norm', 'normalize_advantage', 'target_kl',
    'use_sde', 'sde_sample_freq', 'clip_range', 'clip_range_vf', 'n_envs')


def learner_data(path):
    # Read JSON scalar/schedule representations; do not unpickle model code.
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read('data'))


def check_schedule(parent, saved, rows, summary, rollout_actions):
    """Check actual collection/Adam boundaries against the saved SB3 learner."""
    if any(saved[key] != parent[key] for key in FIXED_PARAMETERS):
        raise ValueError('A learner parameter other than rollout length changed')
    if (parent['n_steps'] != 32 or saved['n_steps'] != rollout_actions
            or saved['batch_size'] != 32 or saved['n_epochs'] != 2 or saved['n_envs'] != 1
            or parent.get('gradient_clip_mode', 'global') != 'global'
            or saved.get('gradient_clip_mode', 'global') != 'global'
            or saved['target_kl'] is not None):
        raise ValueError('Learner does not match the registered 32/256 global schedule')
    for key, value in [('rollout_actions', rollout_actions), ('batch_size', 32), ('ppo_epochs', 2)]:
        if summary[key] != value:
            raise ValueError('Summary learner schedule differs from saved model')
    if not rows:
        raise ValueError('Missing actual optimization boundaries')
    steps_per_phase = 2 * rollout_actions // 32
    previous_controls = 0
    for phase, row in enumerate(rows, 1):
        if (row['actions'] != phase * rollout_actions or row['optimization_phases'] != phase
                or row['optimizer_steps'] != phase * steps_per_phase
                or row['metrics']['train/n_updates'] != parent['_n_updates'] + phase * 2
                or row['physics_steps'] <= previous_controls):
            raise ValueError('Actual rollout/optimizer boundaries differ from registered schedule')
        previous_controls = row['physics_steps']
    for key in ('actions', 'optimization_phases', 'optimizer_steps', 'physics_steps'):
        if rows[-1][key] != summary[key]:
            raise ValueError('Final actual update differs from endpoint summary')
    if (saved['num_timesteps'] != parent['num_timesteps'] + summary['actions']
            or saved['_n_updates'] != parent['_n_updates'] + 2 * len(rows)):
        raise ValueError('Saved cumulative learner counters differ from actual updates')
    return {'rollout_actions': rollout_actions, 'batch_size': 32, 'epochs': 2,
        'optimization_phases': len(rows), 'optimizer_steps_per_phase': steps_per_phase,
        'decisions': summary['actions'], 'optimizer_steps': summary['optimizer_steps'],
        'fixed_parameters': {key: saved[key] for key in FIXED_PARAMETERS}}


def checked_endpoint(batch, name, protocol, size):
    folder = batch / name
    raw = audit(batch, name)
    if raw != read(folder / 'endpoint_raw_audit.json'):
        raise ValueError('Raw frame audit is absent or stale')
    parent = batch / protocol['recovery'][name]['checkpoint']
    summary, evaluation = read(folder / 'summary.json'), read(folder / 'evaluation.json')
    rows = [json.loads(line) for line in (folder / 'optimization.jsonl').read_text(encoding='utf-8').splitlines()]
    schedule = check_schedule(learner_data(parent / 'model.zip'), learner_data(folder / 'model.zip'),
                              rows, summary, size)
    budget = protocol['method_budgets'][name]
    if summary['stopped_reason'] is not None or summary['physics_steps'] < budget:
        raise ValueError('Endpoint did not complete its registered interaction budget')
    if len(rows) > 1 and rows[-2]['physics_steps'] >= budget:
        raise ValueError('Training continued beyond the first full rollout crossing the budget')
    config = read(folder / 'config.json')['env']
    dt = config['control_dt']
    cases = evaluation['episodes']
    if len(cases) != len(protocol['routes']) or {c['case_id'] for c in cases} != set(range(len(protocol['routes']))):
        raise ValueError('Missing or duplicate frozen training-road outcomes')
    return {'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'), 'learner_schedule': schedule,
        'nominal_new_controls': budget, 'new_training_controls': summary['physics_steps'],
        'rollout_overshoot_controls': summary['physics_steps'] - budget,
        'lineage_controls': raw['lineage_controls'], 'charged_controls_lower_bound': raw['charged_controls_lower_bound'],
        'unknown_unflushed_ancestor_tail': raw['unknown_unflushed_ancestor_tail'],
        'training_world_ticks': raw['training_world_ticks_verified'], 'training_reset_ticks': summary['reset_ticks'],
        'training_simulation_seconds': dt * summary['physics_steps'], 'training_wall_seconds': summary['wall_seconds'],
        'evaluation_controls': raw['evaluation_control_frames_verified'], 'evaluation_world_ticks': evaluation['world_ticks'],
        'training_terminations': raw['training_termination_reasons'], 'cases': cases,
        'optimization_metrics': {key: statistics.median(row['metrics'][key] for row in rows)
            for key in ('train/approx_kl', 'train/entropy_loss', 'train/value_loss')},
        'evaluation_summary': {'cases': len(cases), 'successes': sum(c['success'] for c in cases),
            'termination_counts': dict(Counter(c['termination_reason'] for c in cases)),
            'mean_progress_m': statistics.mean(c['progress'] for c in cases),
            'mean_intervention_rate': statistics.mean(c['intervention_rate'] for c in cases)}}


def compare(batch, method):
    batch = Path(batch)
    protocol = read(batch / 'protocol.json')
    if not protocol.get('rollout_length_experiment') or protocol.get('gradient_clipping_experiment'):
        raise ValueError('Not a single-factor registered rollout experiment')
    names = [method + '_rollout' + str(size) for size in (32, 256)]
    jobs = {job['name']: job for job in protocol['jobs']}
    if len(jobs) != len(protocol['jobs']) or list(jobs) != protocol['methods']:
        raise ValueError('Invalid registered job inventory')
    for name, size in zip(names, (32, 256)):
        if jobs.get(name) != {'name': name, 'method': method, 'gradient_clip_mode': 'global', 'rollout_actions': size}:
            raise ValueError('Wrong paired rollout intervention roles')
    if protocol['method_budgets'][names[0]] != protocol['method_budgets'][names[1]]:
        raise ValueError('Unequal registered interaction budgets')
    inherited = []
    for name in names:
        checkpoint = batch / protocol['recovery'][name]['checkpoint']
        inherited.append({p.name: sha(p) for p in checkpoint.iterdir() if p.is_file()})
    if not inherited[0] or inherited[0] != inherited[1]:
        raise ValueError('Pair does not start from identical saved learning state')
    for key in ('lineage_controls', 'consumed_controls_lower_bound', 'retained_optimizer_steps', 'unknown_unflushed_parent_tail'):
        if protocol['recovery'][names[0]][key] != protocol['recovery'][names[1]][key]:
            raise ValueError('Unequal inherited accounting')
    if sha(batch / names[0] / 'config.json') != sha(batch / names[1] / 'config.json'):
        raise ValueError('Pair environment/configuration differs')
    arms = {name: checked_endpoint(batch, name, protocol, size) for name, size in zip(names, (32, 256))}
    tasks = [initial_cases(batch / name) for name in names]
    identities = [{case_id: {k: v for k, v in row.items() if k != 'initial'}
        for case_id, row in cases.items()} for cases in tasks]
    if not identities[0] or identities[0] != identities[1]:
        raise ValueError('Paired probes differ in measured road/task identity')
    by_case = [{c['case_id']: c for c in arms[name]['cases']} for name in names]
    if any(set(cases) != set(tasks[0]) for cases in by_case):
        raise ValueError('Incomplete paired evaluation inventory')
    paired = []
    for case_id in sorted(tasks[0]):
        short, long = [cases[case_id] for cases in by_case]
        start_a, start_b = [cases[case_id]['initial'] for cases in tasks]
        paired.append({'case_id': case_id, 'road_sha256': tasks[0][case_id]['road_sha256'],
            'measured_start_position_difference_m': math.hypot(start_b['x']-start_a['x'], start_b['y']-start_a['y']),
            'measured_start_yaw_difference_rad': (start_b['yaw']-start_a['yaw']+math.pi) % (2*math.pi)-math.pi,
            'measured_start_speed_difference_mps': start_b['v']-start_a['v'],
            'rollout32_success': short['success'], 'rollout256_success': long['success'],
            'rollout32_termination': short['termination_reason'], 'rollout256_termination': long['termination_reason'],
            'progress_difference_m': long['progress']-short['progress'],
            'intervention_rate_difference': long['intervention_rate']-short['intervention_rate']})
    return {'method': method, 'same_parent_checkpoint_hashes': inherited[0], 'arms': arms, 'paired_cases': paired,
        'mean_progress_difference_m': statistics.mean(c['progress_difference_m'] for c in paired),
        'success_difference_count': sum(int(c['rollout256_success'])-int(c['rollout32_success']) for c in paired),
        'actual_training_control_difference': arms[names[1]]['new_training_controls']-arms[names[0]]['new_training_controls']}


def analyze(batch, methods, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous rollout comparison')
    protocol = read(batch / 'protocol.json')
    for name, digest in protocol['source_hashes'].items():
        if sha(batch / 'source' / name) != digest:
            raise ValueError('Frozen experiment source changed')
    result = {'batch': batch.name, 'seed': protocol['seed'], 'protocol_sha256': sha(batch / 'protocol.json'),
        'scope': 'Same-parent 32/256 rollout development comparison on training roads; correlated forks, no independent-seed interval or final-confirmation claim',
        'complete_three_method_comparison': methods == ['H0', 'H', 'B1'],
        'methods': {method: compare(batch, method) for method in methods},
        'new_environment_controls': 0, 'new_optimizer_updates': 0,
        'analysis_source_hashes': {name: sha(ROOT / 'scripts' / name) for name in
            ('analyze_native_rollout_pair.py', 'analyze_native_gradient_pair.py', 'audit_native_method_endpoint.py')}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({method: {name: arm['evaluation_summary'] for name, arm in row['arms'].items()}
        for method, row in result['methods'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--method', choices=['all', 'H0', 'H', 'B1'], default='all')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.batch, ['H0', 'H', 'B1'] if args.method == 'all' else [args.method], args.output)

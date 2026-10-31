"""Compare both completed matched B1 actor-fit-to-PPO development endpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import audit, read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_supervised_ppo import representation_inventory


def analyze(replay, approach, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve the previous full comparison')
    protocols, configs, learners, episodes, evidence = [], [], [], {}, {}
    for arm, batch in [('replay', Path(replay)), ('approach', Path(approach))]:
        require_finished_parent(batch)
        protocol = read(batch/'protocol.json')
        folder = batch/'B1'
        raw = audit(batch, 'B1')
        if raw != read(folder/'endpoint_raw_audit.json'):
            raise ValueError('Stored raw audit differs')
        summary, evaluation = read(folder/'summary.json'), read(folder/'evaluation.json')
        if (protocol['methods'] != ['B1'] or protocol.get('evaluation_only')
                or protocol['approach_initialization']['arm'] != arm
                or summary['physics_steps'] != 20480 or summary['optimizer_steps'] <= 0
                or summary['stopped_reason'] is not None
                or sha(folder/'model.zip') != evaluation['model_sha256']
                or sha(folder/'normalization.pkl') != evaluation['normalization_sha256']):
            raise ValueError('Require both complete registered 20,480-control PPO endpoints')
        for path, key in [(batch/'protocol.json', 'protocol_sha256'),
                          (folder/'config.json', 'config_sha256'),
                          (folder/'evaluation.json', 'evaluation_sha256'),
                          (folder/'evaluation_trace.jsonl.gz', 'evaluation_trace_sha256')]:
            if sha(path) != raw[key]:
                raise ValueError('Endpoint input hash differs')
        _, rows, count = representation_inventory(protocol, evaluation)
        if count != 12:
            raise ValueError('Require every known development condition')
        updates = [json.loads(line) for line in (folder/'optimization.jsonl').read_text().splitlines()]
        if ([row['physics_steps'] for row in updates] != list(range(1280, 20481, 1280))
                or updates[-1]['optimizer_steps'] != summary['optimizer_steps']):
            raise ValueError('Optimizer/physical boundaries differ')
        with zipfile.ZipFile(folder/'model.zip') as archive:
            learner = json.loads(archive.read('data'))
        if learner['num_timesteps'] != 20480:
            raise ValueError('Saved learner counter differs')
        resume = batch/protocol['recovery']['B1']['checkpoint']
        with zipfile.ZipFile(resume/'model.zip') as archive:
            initial = json.loads(archive.read('data'))
        if initial['num_timesteps'] or initial['_n_updates']:
            raise ValueError('Require the registered zero-PPO actor initialization')
        evidence[arm] = {'batch': batch.name, 'successes': sum(row['success'] for row in rows),
            'termination_counts': dict(Counter(row['termination_reason'] for row in rows)),
            'unsafe_episodes': sum(row['termination_reason'] in ('collision', 'out_of_bounds') for row in rows),
            'training_controls': summary['physics_steps'], 'optimizer_steps': summary['optimizer_steps'],
            'raw_controls_verified': raw['total_applied_controls_verified'],
            'charged_input_controls_lower_bound': protocol['recovery']['B1']['consumed_controls_lower_bound'],
            'model_sha256': evaluation['model_sha256'], 'raw_audit_sha256': sha(folder/'endpoint_raw_audit.json'),
            'core_successes': {name: sum(row['success'] for row in rows if row['case_id']//4 == index)
                               for index, name in enumerate(('curve_parking', 'static', 'dynamic'))},
            'logged_KL_mean': sum(row['metrics']['train/approx_kl'] for row in updates)/len(updates),
            'logged_KL_max': max(row['metrics']['train/approx_kl'] for row in updates)}
        protocols.append(protocol)
        configs.append(read(folder/'config.json')['env'])
        learners.append(learner)
        episodes[arm] = {row['case_id']: row for row in rows}
    for key in ('seed', 'native_case_manifest', 'calibration_sha256', 'road_geometry_version',
                'routes', 'source_hashes', 'freeze_observation_statistics', 'PPO_reward_units'):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError('Matched task/runtime differs: '+key)
    if configs[0] != configs[1]:
        raise ValueError('Physical task, direct actions or success gates differ')
    schedule_keys = ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'learning_rate',
                     'ent_coef', 'vf_coef', 'max_grad_norm', 'normalize_advantage', 'target_kl')
    if any(learners[0][key] != learners[1][key] for key in schedule_keys):
        raise ValueError('Registered PPO schedules differ')
    if set(episodes['replay']) != set(episodes['approach']):
        raise ValueError('Case inventory differs')
    wins = [case for case in sorted(episodes['approach'])
            if episodes['approach'][case]['success'] and not episodes['replay'][case]['success']]
    losses = [case for case in sorted(episodes['replay'])
              if episodes['replay'][case]['success'] and not episodes['approach'][case]['success']]
    report = {'scope': __doc__, 'analyzer_sha256': sha(Path(__file__)), 'seed': protocols[0]['seed'],
        'cases': 12, 'arms': evidence, 'schedule': {key: learners[0][key] for key in schedule_keys},
        'approach_only_cases': wins, 'replay_only_cases': losses,
        'approach_minus_replay_percentage_points': 100*(len(wins)-len(losses))/12,
        'both_rejected_as_healthy_full_task_baseline': True,
        'new_audit_controls': 0, 'new_audit_optimizer_steps': 0,
        'limits': 'One seed, twelve correlated known development conditions. Different disclosed actor-fit labels; shared physical/PPO schedule. Actual Adam counts differ under KL stopping. No independent-seed significance, final-test, healthy-baseline or hierarchy-value claim.',
        'research_acceptance': 'UNMET'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('replay', 'approach', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    analyze(args.replay, args.approach, args.output)

"""Compare both complete actor-fit endpoints; no checkpoint or condition selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_supervised_ppo import representation_inventory, basic_motion_training_gate


def analyze(pair, output):
    pair, output = Path(pair), Path(output)
    if output.exists():
        raise FileExistsError('Preserve the prior comparison')
    registration = read(pair/'registration.json')
    arms = read(pair/'complete.json')['arms']
    if [row['arm'] for row in arms] != ['replay', 'approach']:
        raise ValueError('Require both registered actor fits in their original order')
    protocols, configs, rows, evidence = [], [], {}, {}
    for arm in arms:
        batch = ROOT/'runs'/arm['batch']
        require_finished_parent(batch)
        folder = batch/'B1'
        protocol, raw = read(batch/'protocol.json'), read(folder/'endpoint_raw_audit.json')
        evaluation, summary = read(folder/'evaluation.json'), read(folder/'summary.json')
        if (not protocol.get('evaluation_only') or summary['physics_steps'] or summary['optimizer_steps']
                or summary['lineage_control_steps'] or protocol['approach_initialization']['arm'] != arm['arm']
                or sha(folder/'model.zip') != arm['model_sha256']
                or sha(folder/'model.zip') != evaluation['model_sha256']):
            raise ValueError('Endpoint differs from its registered zero-PPO actor fit')
        for path, key in ((batch/'protocol.json', 'protocol_sha256'),
                (folder/'config.json', 'config_sha256'), (folder/'evaluation.json', 'evaluation_sha256'),
                (folder/'evaluation_trace.jsonl.gz', 'evaluation_trace_sha256')):
            if sha(path) != raw[key]:
                raise ValueError('Complete raw-audit input binding differs')
        _, episodes, count = representation_inventory(protocol, evaluation)
        if count != 12:
            raise ValueError('Require all twelve known development conditions')
        try:
            basic = basic_motion_training_gate(protocol, evaluation)
        except ValueError as exc:
            basic = {'passes': False, 'reason': str(exc)}
        else:
            basic['passes'] = True
        rows[arm['arm']] = {r['case_id']: r for r in episodes}
        protocols.append(protocol)
        configs.append(read(folder/'config.json')['env'])
        evidence[arm['arm']] = {'batch': batch.name, 'successes': sum(r['success'] for r in episodes),
            'cases': count, 'termination_counts': dict(Counter(r['termination_reason'] for r in episodes)),
            'raw_controls_verified': raw['total_applied_controls_verified'], 'new_PPO_controls': 0,
            'PPO_updates': 0, 'actor_fit_steps': registration['steps_each'], 'basic_motion_entry': basic,
            'raw_audit_sha256': sha(folder/'endpoint_raw_audit.json'), 'model_sha256': arm['model_sha256'],
            'episodes': [{k: r[k] for k in ('case_id', 'success', 'termination_reason', 'physics_steps', 'progress')}
                         for r in episodes]}
    for key in ('seed', 'native_case_manifest', 'calibration_sha256', 'road_geometry_version', 'routes', 'source_hashes'):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError('Matched comparison differs in '+key)
    if configs[0] != configs[1]:
        raise ValueError('Physical task, action or success gates differ')
    wins = [i for i in range(12) if rows['approach'][i]['success'] and not rows['replay'][i]['success']]
    losses = [i for i in range(12) if rows['replay'][i]['success'] and not rows['approach'][i]['success']]
    report = {'scope': __doc__, 'registration_sha256': sha(pair/'registration.json'),
        'analyzer_sha256': sha(Path(__file__)), 'seed': protocols[0]['seed'], 'arms': evidence,
        'approach_only_cases': wins, 'replay_only_cases': losses,
        'approach_minus_replay_percentage_points': 100*(len(wins)-len(losses))/12,
        'limits': 'One seed,known development conditions,actor-only fits from the same zero-PPO initialization. Additional valid training-state reference labels and fit compute are disclosed;no PPO learning,independent confirmation or healthy full-task qualification implied.',
        'research_acceptance': 'UNMET'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'arms'}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.pair, args.output)

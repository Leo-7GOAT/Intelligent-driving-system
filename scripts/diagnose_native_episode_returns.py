"""Reconstruct completed native episode returns from audited physical ticks.

This describes the unchanged objective and visited trajectories. It is not a
counterfactual reward experiment, a causal explanation, or new driving evidence.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(batch, method, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous episode-return evidence')
    folder = batch / method
    audit = read(folder / 'endpoint_raw_audit.json')
    for name,key in [('config.json','config_sha256'),('summary.json','summary_sha256')]:
        if sha(folder/name) != audit[key]:
            raise ValueError('Raw-audited input changed')
    config = read(folder/'config.json')['env']
    reward = config['reward']
    if reward['heading_potential_weight'] != 0 or reward['progress_weight'] != 0:
        raise ValueError('This diagnostic requires the declared distance-only PBRS objective')
    gamma = reward['gamma_per_second'] ** config['control_dt']
    episodes = {}
    for line in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['episode_id'] in episodes:
            raise ValueError('Duplicate completed episode identity')
        episodes[row['episode_id']] = row
    sums = defaultdict(lambda: {'controls':0,'return':0.,'parts':defaultdict(float),'termination':None})
    total = 0
    for name,digest in sorted(audit['transition_hashes'].items()):
        if sha(folder/name) != digest:
            raise ValueError('Training trace changed after raw audit')
        with gzip.open(folder/name,'rt',encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); summary = sums[row['episode_id']]
                if summary['termination'] is not None:
                    raise ValueError('Controls recorded after terminal episode')
                for tick in row['trace']:
                    if not math.isclose(sum(tick['reward_components'].values()),tick['reward'],abs_tol=1e-9):
                        raise ValueError('Physical reward parts do not sum to recorded reward')
                    discount = gamma ** summary['controls']
                    summary['return'] += discount * tick['reward']
                    for key,value in tick['reward_components'].items():
                        summary['parts'][key] += discount * value
                    summary['controls'] += 1; total += 1
                summary['termination'] = row['termination_reason']
    if total != audit['training_control_frames_verified']:
        raise ValueError('Physical training inventory mismatch')
    if {key for key,row in sums.items() if row['termination']} != set(episodes):
        raise ValueError('Completed episode ledger differs from terminal traces')
    rows = []
    for episode_id,episode in sorted(episodes.items()):
        summary = sums[episode_id]
        scene = episode['initial_scenario']; start,goal = scene['initial'],scene['goal']
        expected = reward['beta'] * min(math.hypot(start['x']-goal[0],start['y']-goal[1])/reward['distance_scale'],2.)
        error = summary['parts']['potential'] - expected
        if (summary['controls'] != episode['physics_steps']
                or summary['termination'] != episode['termination_reason']
                or not math.isclose(summary['return'],episode['discounted_return'],abs_tol=1e-6)
                or abs(error) > 1e-8):
            raise ValueError('Recorded return or terminal-zero potential identity does not reproduce')
        rows.append({'episode_id':episode_id,'road_sha256':scene['road_sha256'],
            'controls':summary['controls'],'termination':summary['termination'],
            'progress_m':episode['progress'],'discounted_return':summary['return'],
            'discounted_parts':dict(summary['parts']),'expected_terminal_potential_sum':expected,
            'potential_identity_error':error})
    groups = []
    for identity in sorted({(r['road_sha256'],r['controls'],r['termination']) for r in rows}):
        group = [r for r in rows if (r['road_sha256'],r['controls'],r['termination']) == identity]
        def span(values):
            return {'min':min(values),'max':max(values),'range':max(values)-min(values)}
        best = max(group,key=lambda r:r['discounted_return'])
        groups.append({'road_sha256':identity[0],'controls':identity[1],'termination':identity[2],
            'episodes':len(group),'progress_m':span([r['progress_m'] for r in group]),
            'return':span([r['discounted_return'] for r in group]),
            'discounted_component_spans':{key:span([r['discounted_parts'][key] for r in group])
                for key in group[0]['discounted_parts']},
            'highest_return_episode_id':best['episode_id'],'highest_return_episode_progress_m':best['progress_m']})
    result={'scope':'Observed completed training episodes under unchanged distance-only terminal-zero PBRS. Correct telescoping does not make shaping a bug or prove the cause of failure. No causal action values, new reward training, simulator controls or policy updates.',
        'batch':batch.name,'method':method,'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'episode_ledger_sha256':sha(folder/'episodes.jsonl'),'script_sha256':sha(Path(__file__)),
        'gamma_per_control':gamma,'completed_episodes':len(rows),'all_training_controls':total,
        'unfinished_episode_controls':sum(row['controls'] for key,row in sums.items() if key not in episodes),
        'max_absolute_potential_identity_error':max((abs(r['potential_identity_error']) for r in rows),default=0.),
        'groups':groups,'episodes':rows,'new_controls':0,'optimizer_updates':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','groups')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',required=True);parser.add_argument('--method',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();analyze(args.batch,args.method,args.output)

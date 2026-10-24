"""Compare all registered clipping arms, or an explicitly partial completed pair.

No independent-seed confidence interval is inferred from four training roads.
The same checkpoint, environment and case identities are required within a pair.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_endpoint(batch, name):
    folder = batch/name
    audit = read(folder/'endpoint_raw_audit.json')
    if audit['method'] != name or audit['protocol_sha256'] != sha(batch/'protocol.json'):
        raise ValueError('Wrong endpoint audit identity')
    for filename, key in (
        ('config.json','config_sha256'), ('summary.json','summary_sha256'),
        ('evaluation.json','evaluation_sha256'), ('optimization.jsonl','optimization_sha256'),
        ('evaluation_trace.jsonl.gz','evaluation_trace_sha256')):
        if sha(folder/filename) != audit[key]:
            raise ValueError('Endpoint changed since raw audit: '+name+'/'+filename)
    for filename, digest in audit['transition_hashes'].items():
        if sha(folder/filename) != digest:
            raise ValueError('Training trace changed')
    if sha(batch/'infrastructure'/audit['bridge_response_stream']) != audit['bridge_response_sha256']:
        raise ValueError('Measured bridge evidence changed')
    summary, evaluation = read(folder/'summary.json'), read(folder/'evaluation.json')
    for filename, key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
        if sha(folder/filename) != evaluation[key]:
            raise ValueError('Evaluation policy bytes changed')
    episodes = evaluation['episodes']
    if len({e['case_id'] for e in episodes}) != len(episodes):
        raise ValueError('Duplicate probe identity')
    rows = [json.loads(line) for line in (folder/'optimization.jsonl').read_text(encoding='utf-8').splitlines()]
    if not rows or any(key not in row for row in rows for key in ('postclip_gradients','metrics')):
        raise ValueError('Missing registered gradient observations')
    stats = {'actor_gradient_median':statistics.median(r['postclip_gradients']['actor_l2_mean'] for r in rows),
        'critic_gradient_median':statistics.median(r['postclip_gradients']['critic_l2_mean'] for r in rows),
        'approx_kl_median':statistics.median(r['metrics']['train/approx_kl'] for r in rows),
        'entropy_loss_median':statistics.median(r['metrics']['train/entropy_loss'] for r in rows)}
    return {'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'new_training_controls':summary['physics_steps'], 'lineage_controls':summary['lineage_control_steps'],
        'charged_controls_lower_bound':summary['charged_control_steps_lower_bound'],
        'optimizer_steps':summary['optimizer_steps'], 'training_world_ticks':summary['world_ticks'],
        'training_wall_seconds':summary['wall_seconds'],
        'training_terminations':audit['training_termination_reasons'],
        'verified_controls':audit['total_applied_controls_verified'],
        'gradient_statistics':stats, 'cases':episodes}


def initial_cases(folder):
    import gzip
    cases = {}
    with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['case_id'] not in cases:
                initial = row['info']['initial_scenario']
                cases[row['case_id']] = {key:initial[key] for key in ('road_sha256','goal','static','dynamic','initial')}
    return cases


def compare(batch, method):
    batch = Path(batch)
    protocol = read(batch/'protocol.json')
    if not protocol.get('gradient_clipping_experiment'):
        raise ValueError('Not a registered clipping experiment')
    names = [method+'_global', method+'_independent']
    jobs = {job['name']:job for job in protocol['jobs']}
    if (any(name not in protocol['methods'] or jobs[name]['method'] != method for name in names)
            or [jobs[name]['gradient_clip_mode'] for name in names] != ['global','independent_actor_critic_v1']):
        raise ValueError('Wrong paired intervention roles')
    if protocol['method_budgets'][names[0]] != protocol['method_budgets'][names[1]]:
        raise ValueError('Unequal registered interaction budgets')
    for key in ('lineage_controls','consumed_controls_lower_bound','retained_optimizer_steps','unknown_unflushed_parent_tail'):
        if protocol['recovery'][names[0]][key] != protocol['recovery'][names[1]][key]:
            raise ValueError('Unequal inherited accounting')
    inherited = []
    for name in names:
        checkpoint = batch/protocol['recovery'][name]['checkpoint']
        inherited.append({p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()})
    if not inherited[0] or inherited[0] != inherited[1]:
        raise ValueError('Pair does not start from identical saved learning state')
    if sha(batch/names[0]/'config.json') != sha(batch/names[1]/'config.json'):
        raise ValueError('Pair environment/configuration differs')
    arms = {name:checked_endpoint(batch,name) for name in names}
    tasks = [initial_cases(batch/name) for name in names]
    identities = [{case_id:{key:value for key,value in row.items() if key!='initial'}
                   for case_id,row in cases.items()} for cases in tasks]
    if not identities[0] or identities[0] != identities[1]:
        raise ValueError('Paired probes differ in measured road/task identity')
    by_case = [{e['case_id']:e for e in arms[name]['cases']} for name in names]
    if any(set(cases) != set(tasks[0]) for cases in by_case):
        raise ValueError('Incomplete paired evaluation inventory')
    paired = []
    for case_id in sorted(tasks[0]):
        global_case, independent_case = [cases[case_id] for cases in by_case]
        start_a, start_b = [cases[case_id]['initial'] for cases in tasks]
        paired.append({'case_id':case_id, 'road_sha256':tasks[0][case_id]['road_sha256'],
            'measured_start_position_difference_m':math.hypot(start_b['x']-start_a['x'],start_b['y']-start_a['y']),
            'measured_start_yaw_difference_rad':(start_b['yaw']-start_a['yaw']+math.pi)%(2*math.pi)-math.pi,
            'measured_start_speed_difference_mps':start_b['v']-start_a['v'],
            'global_success':global_case['success'], 'independent_success':independent_case['success'],
            'global_termination':global_case['termination_reason'],
            'independent_termination':independent_case['termination_reason'],
            'progress_difference_m':independent_case['progress']-global_case['progress'],
            'intervention_rate_difference':independent_case['intervention_rate']-global_case['intervention_rate']})
    for arm in arms.values():
        cases = arm['cases']
        arm['evaluation_summary'] = {'successes':sum(c['success'] for c in cases), 'cases':len(cases),
            'mean_progress_m':statistics.mean(c['progress'] for c in cases),
            'mean_intervention_rate':statistics.mean(c['intervention_rate'] for c in cases),
            'termination_counts':{reason:sum(c['termination_reason']==reason for c in cases)
                for reason in sorted({c['termination_reason'] for c in cases})}}
    return {'method':method, 'same_parent_checkpoint_hashes':inherited[0], 'arms':arms,
        'paired_cases':paired, 'mean_progress_difference_m':statistics.mean(r['progress_difference_m'] for r in paired),
        'success_difference_count':sum(int(r['independent_success'])-int(r['global_success']) for r in paired)}


def analyze(batch, methods, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous comparison')
    protocol = read(batch/'protocol.json')
    for name,digest in protocol['source_hashes'].items():
        if sha(batch/'source'/name) != digest:
            raise ValueError('Frozen experiment source changed')
    result = {'batch':batch.name, 'protocol_sha256':sha(batch/'protocol.json'),
        'scope':'Same-parent clipping development comparison on training roads; correlated forks, no independent-seed inference or policy selection',
        'seed':protocol['seed'],
        'complete_three_method_comparison':methods==['H0','H','B1'], 'methods':{},
        'new_environment_controls':0, 'new_optimizer_updates':0, 'script_sha256':sha(Path(__file__))}
    for method in methods:
        result['methods'][method] = compare(batch,method)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({m:{name:arm['evaluation_summary'] for name,arm in row['arms'].items()}
        for m,row in result['methods'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch',required=True)
    parser.add_argument('--method',choices=['all','H0','H','B1'],default='all',help='A single method is explicitly partial evidence')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    analyze(args.batch,['H0','H','B1'] if args.method=='all' else [args.method],args.output)

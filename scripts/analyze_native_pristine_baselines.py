"""Rebuild trained-H/teacher-only-H/B3 development attribution from raw-bound endpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha
from scripts.analyze_native_feedback_cohort import paired_seed_statistics
from scripts.evaluate_native_rule import require_finished_parent


def complete_cases(episodes):
    cases={e['case_id']:e for e in episodes}
    if len(episodes)!=24 or set(cases)!=set(range(24)):
        raise ValueError('All24known development cases exactly once are required')
    if any(type(e['success']) is not bool or e['success']!=(e['termination_reason']=='success') for e in episodes):
        raise ValueError('Success and terminal outcome disagree')
    return cases


def compare_cases(left,right):
    a,b=complete_cases(left),complete_cases(right)
    common=[i for i in a if a[i]['success'] and b[i]['success']]
    return {'left_successes':sum(e['success'] for e in a.values()),'right_successes':sum(e['success'] for e in b.values()),
        'safe_completion_difference':sum(int(a[i]['success'])-int(b[i]['success']) for i in a)/24,
        'left_unsafe_episodes':sum(e['termination_reason'] in ('collision','out_of_bounds') for e in a.values()),
        'right_unsafe_episodes':sum(e['termination_reason'] in ('collision','out_of_bounds') for e in b.values()),
        'left_termination_counts':dict(Counter(e['termination_reason'] for e in a.values())),
        'right_termination_counts':dict(Counter(e['termination_reason'] for e in b.values())),
        'left_all_outcome_simulation_seconds':sum(e['simulation_seconds'] for e in a.values()),
        'right_all_outcome_simulation_seconds':sum(e['simulation_seconds'] for e in b.values()),
        'left_interventions':sum(e['interventions'] for e in a.values()),'right_interventions':sum(e['interventions'] for e in b.values()),
        'mutually_successful_case_ids':common,
        'mutually_successful_mean_time_difference_seconds':sum(a[i]['simulation_seconds']-b[i]['simulation_seconds'] for i in common)/len(common) if common else None,
        'duration_limit':'All-outcome duration is a cost ledger,not speed superiority;early failures may be short. Travel-time difference uses only explicitly listed mutually successful cases.'}


def rule_compatibility(paired,rule):
    paired,rule=Path(paired),Path(rule);p,q=read(paired/'protocol.json'),read(rule/'protocol.json')
    if (q['config']!=read(paired/'resume/H/config.json')['env'] or q.get('native_validation')!=p['native_validation']
            or q['routes']!=p['routes'] or q['native_case_manifest']!=p['native_case_manifest']
            or q['calibration_sha256']!=p['calibration_sha256'] or q['rule']!='existing_navstack.env.rule_action'):
        raise ValueError('Strong rule uses a different task,information interface or physical calibration')
    inactive={'control_cadence.py','critic_rate_policy.py','fixed_reward_units.py','pursuit_controller.py','reference_manifest.py','terminal_region.py'}
    if any(p.get(k) is not None for k in ('control_cadence','PPO_reward_units','controller','reference_manifest','terminal_region')):
        raise ValueError('Previously inactive diagnostics are activated;re-audit rule comparability')
    common={}
    for path in (paired/'source/navstack').glob('*.py'):
        if path.name in inactive:continue
        other=rule/'source/navstack'/path.name
        if not other.is_file() or sha(path)!=sha(other):raise ValueError('Strong rule execution source differs:'+path.name)
        common['navstack/'+path.name]=sha(path)
    name='scripts/carla_bridge37.py'
    if sha(paired/'source'/name)!=sha(rule/'source'/name):raise ValueError('Rule bridge initialization differs')
    common[name]=sha(paired/'source'/name)
    raw=read(rule/'raw_audit.json')
    for name,key in [('protocol.json','protocol_sha256'),('summary.json','summary_sha256'),('evaluation_trace.jsonl.gz','trace_sha256')]:
        if sha(rule/name)!=raw[key]:raise ValueError('Strong rule audited bytes changed')
    result=read(rule/'summary.json');complete_cases(result['episodes'])
    if result['status']!='complete' or result['training_physics_steps'] or result['optimizer_updates']:
        raise ValueError('Require completed deterministic rule evaluation without training')
    return {'batch':rule.name,'raw_audit_sha256':sha(rule/'raw_audit.json'),'registered_cases':24,
        'raw_controls_verified':raw['evaluation_controls_verified'],'common_active_modules_sha256':common,
        'excluded_inactive_diagnostic_modules':sorted(inactive),
        'scope':'Reuse the completed strong deterministic rule on the exact same24known conditions and actual bridge;not five independently trained rule policies'}


def endpoint(batch):
    batch=Path(batch);require_finished_parent(batch);folder=batch/'H';raw=read(folder/'endpoint_raw_audit.json')
    for path,key in [(batch/'protocol.json','protocol_sha256')]+[(folder/n,k) for n,k in (
            ('config.json','config_sha256'),('summary.json','summary_sha256'),('evaluation.json','evaluation_sha256'),
            ('evaluation_trace.jsonl.gz','evaluation_trace_sha256'))]:
        if sha(path)!=raw[key]:raise ValueError('Endpoint raw binding changed')
    evaluation=read(folder/'evaluation.json');complete_cases(evaluation['episodes'])
    if evaluation.get('partition')!='validation':raise ValueError('Wrong development partition')
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
        if sha(folder/name)!=evaluation[key]:raise ValueError('Evaluated artifact changed')
    return evaluation,read(folder/'summary.json'),raw


def analyze(root,rows,rule,output):
    root,output=Path(root).resolve(),Path(output)
    if output.exists():raise FileExistsError('Preserve the earlier analysis')
    if [r['seed'] for r in rows]!=[11,22,33,44,55]:raise ValueError('All five matched seeds are required')
    rule=root/'runs'/rule;proof=rule_compatibility(root/'runs'/rows[0]['paired_training_batch'],rule)
    rule_episodes=read(rule/'summary.json')['episodes'];results=[];common_source=None
    for row in rows:
        parent=root/'runs'/row['paired_training_batch'];baseline=root/'runs'/row['batch']
        p,q=read(parent/'protocol.json'),read(baseline/'protocol.json')
        for key in ('source_hashes','routes','native_case_manifest','native_validation','seed','calibration_sha256'):
            if p[key]!=q[key]:raise ValueError('PPO and teacher-only control differ:'+key)
        if common_source is None:common_source=p['source_hashes']
        if p['source_hashes']!=common_source:raise ValueError('Execution source differs across paired seeds')
        for name,digest in p['source_hashes'].items():
            if sha(parent/'source'/name)!=digest or sha(baseline/'source'/name)!=digest:
                raise ValueError('Paired frozen source changed')
        if read(parent/'H/config.json')['env']!=read(baseline/'H/config.json')['env']:
            raise ValueError('Executed physical/task configurations differ')
        if (p['seed']!=row['seed'] or sha(parent/'resume/H/model.zip')!=sha(baseline/'resume/H/model.zip')
                or sha(parent/'resume/H/normalization.pkl')!=sha(baseline/'resume/H/normalization.pkl')):
            raise ValueError('Matched original initialization differs')
        trained,ts,ta=endpoint(parent);pristine,ps,pa=endpoint(baseline)
        if (ts['physics_steps']<20480 or ts['optimizer_steps']<=0 or ps['physics_steps'] or ps['optimizer_steps']
                or pristine['model_sha256']!=row['model_sha256']
                or pristine['normalization_sha256']!=row['normalization_sha256']):
            raise ValueError('Actual PPO and zero-PPO control lineage is not established')
        results.append({'seed':row['seed'],'paired_training_batch':parent.name,'baseline_batch':baseline.name,
            'PPO_minus_teacher_only':compare_cases(trained['episodes'],pristine['episodes']),
            'PPO_minus_strong_rule':compare_cases(trained['episodes'],rule_episodes),
            'PPO_training_controls':ts['physics_steps'],'PPO_Adam_steps':ts['optimizer_steps'],
            'teacher_only_new_controls':0,'teacher_only_new_updates':0,
            'trained_raw_audit_sha256':sha(parent/'H/endpoint_raw_audit.json'),'baseline_raw_audit_sha256':sha(baseline/'H/endpoint_raw_audit.json'),
            'trained_model_sha256':trained['model_sha256'],'baseline_model_sha256':pristine['model_sha256'],
            'new_baseline_evaluation_controls':pa['evaluation_control_frames_verified']})
    report={'scope':__doc__+' All24conditions were known development validation;this is not final confirmation.',
        'seeds':[r['seed'] for r in rows],'unique_conditions':24,'policy_seed_case_evaluations_each_arm':120,
        'independently_PPO_trained_policies':5,'teacher_only_PPO_updates':0,'strong_rule':proof,'results':results,
        'statistics':{name:paired_seed_statistics([r[name]['safe_completion_difference'] for r in results])
            for name in ('PPO_minus_teacher_only','PPO_minus_strong_rule')},
        'new_controls_for_analysis':0,'optimizer_updates_for_analysis':0,'research_acceptance':'UNMET',
        'limits':'Teacher-only control isolates added PPO after a common supervised prior on this development distribution. Strong-rule comparison is distinct. Two exploratory comparisons,no confirmatory significance or general RL-necessity claim. Repeated seeds do not turn24conditions into120unique final cases.'}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();execution=Path(args.execution).resolve();plan=read(execution/'plan.json')
    rows=read(execution.parent/plan['preparation']/'status.json')['batches']
    analyze(execution.parent.parent,rows,plan['strong_rule_batch'],args.output)

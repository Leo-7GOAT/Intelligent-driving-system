"""Rebuild all five completed basic-course outcomes without selecting seeds or cases."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_native_method_endpoint import read,sha
from scripts.prepare_native_full_task_after_course import require_all_courses,require_completed_courses_for_extension
from scripts.run_native_basic_motion_ppo_cohort import basic_qualification


def analyze(execution,output):
    execution,output=Path(execution).resolve(),Path(output)
    if output.exists():raise FileExistsError('Preserve the previous cohort result')
    state=read(execution/'status.json')
    if (execution/'failure.json').exists():raise ValueError('Cohort infrastructure failure remains unresolved')
    if state['phase']=='ALL_FIVE_BASIC_PPO_COURSES_QUALIFIED_FOR_FULL_TASK_TRAINING':require_all_courses(state)
    else:require_completed_courses_for_extension(state)
    root=execution.parent.parent;results=[];environment=None;manifest=None
    for row in state['completed']:
        batch=root/'runs'/row['batch'];folder=batch/'B1'
        protocol=read(batch/'protocol.json');audit=read(folder/'endpoint_raw_audit.json')
        if sha(batch/'protocol.json')!=audit['protocol_sha256']:raise ValueError('Audited protocol changed')
        for name,key in (('config.json','config_sha256'),('evaluation.json','evaluation_sha256'),
                          ('summary.json','summary_sha256'),('evaluation_trace.jsonl.gz','evaluation_trace_sha256'),
                          ('optimization.jsonl','optimization_sha256')):
            if sha(folder/name)!=audit[key]:raise ValueError('Audited endpoint input changed:'+name)
        evaluation=read(folder/'evaluation.json');summary=read(folder/'summary.json');cfg=read(folder/'config.json')
        for name,key in (('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')):
            if sha(folder/name)!=evaluation[key]:raise ValueError('Evaluated artifact changed')
        if (protocol['seed']!=row['seed'] or cfg['seed']!=row['seed'] or protocol['methods']!=['B1']
                or protocol['method_budgets']!={'B1':10240} or summary['physics_steps']!=10240
                or summary['optimizer_steps']<=0 or summary['stopped_reason'] is not None
                or audit['training_control_frames_verified']!=10240 or summary['reload_exact'] is not True):
            raise ValueError('Every seed requires its complete audited actual PPOcourse')
        if environment is None:
            environment=cfg['env'];manifest=protocol['native_case_manifest']
        if cfg['env']!=environment or protocol['native_case_manifest']!=manifest or len(manifest['cases'])!=4:
            raise ValueError('Known basic conditions and resolved physical environments must match')
        episodes=evaluation['episodes'];qualified=basic_qualification(episodes)
        if qualified is not row['basic_qualified_after_learning']:raise ValueError('Recorded gate differs from measured outcomes')
        results.append({'seed':row['seed'],'batch':batch.name,'successes':sum(e['success'] for e in episodes),
            'unsafe_episodes':sum(e['termination_reason'] in ('collision','out_of_bounds') for e in episodes),
            'qualified':qualified,'termination_counts':dict(Counter(e['termination_reason'] for e in episodes)),
            'new_training_controls':summary['physics_steps'],'lineage_controls':summary['lineage_control_steps'],
            'actual_new_Adam_steps':summary['optimizer_steps'],'total_raw_controls_verified':audit['total_applied_controls_verified'],
            'model_sha256':evaluation['model_sha256'],'normalization_sha256':evaluation['normalization_sha256'],
            'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
            'episodes':[{k:e[k] for k in ('case_id','success','termination_reason','physics_steps','interventions','simulation_seconds')} for e in episodes]})
    result={'scope':__doc__+' Same four known TRAINING roads repeated across five trained policies;not20unique conditions or a healthy full-task baseline.',
        'execution':execution.name,'execution_status_sha256':sha(execution/'status.json'),'source_script_sha256':sha(__file__),
        'unique_known_condition_count':4,'trained_policy_seeds':[r['seed'] for r in results],
        'successes':sum(r['successes'] for r in results),'evaluated_episodes':20,
        'unsafe_episodes':sum(r['unsafe_episodes'] for r in results),'qualified_seeds':sum(r['qualified'] for r in results),
        'total_new_training_controls':sum(r['new_training_controls'] for r in results),
        'actual_new_Adam_steps':sum(r['actual_new_Adam_steps'] for r in results),
        'raw_controls_verified':sum(r['total_raw_controls_verified'] for r in results),
        'seeds':results,'new_controls_for_analysis':0,'optimizer_updates_for_analysis':0,'research_acceptance':'UNMET'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='seeds'}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();analyze(args.execution,args.output)

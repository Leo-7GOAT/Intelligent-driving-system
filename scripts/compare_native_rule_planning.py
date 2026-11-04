"""Compare two whole-audited B3 runs differing only in queue guidance."""
import argparse
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from scripts.audit_native_rule import audit,read,sha
from scripts.evaluate_native_rule import require_finished_parent


CONTROL_FILES=('navstack/env.py','navstack/subgoals.py','navstack/vehicle.py',
    'navstack/trajectory.py','navstack/mpc.py','navstack/carla_actuation.py',
    'scripts/carla_bridge37.py','navstack/world.py','navstack/road_grid.py',
    'navstack/carla_bridge.py','navstack/road_candidate_policy.py')


def compare(baseline,candidate,output):
    baseline,candidate,output=Path(baseline),Path(candidate),Path(output)
    if output.exists():raise FileExistsError('Preserve comparison evidence')
    protocols=[];summaries=[];streams=[]
    for batch in (baseline,candidate):
        require_finished_parent(batch)
        if audit(batch)!=read(batch/'raw_audit.json'):raise ValueError('Whole-run raw audit absent or stale')
        protocols.append(read(batch/'protocol.json'));summaries.append(read(batch/'summary.json'))
        with gzip.open(batch/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            streams.append([json.loads(line) for line in stream])
    old,new=protocols;configs=[deepcopy(p['config']) for p in protocols]
    if configs[0]['planner'].pop('heuristic_mode','euclidean')!='euclidean' or configs[1]['planner'].pop('heuristic_mode',None)!='dubins_v1':
        raise ValueError('Expected explicit Euclidean versus Dubins guidance')
    if configs[0]!=configs[1] or old['routes']!=new['routes']:
        raise ValueError('Other task/planner/vehicle configuration differs')
    if old['calibration_sha256']!=new['calibration_sha256'] or old['rule_speed_policy']!=new['rule_speed_policy']:
        raise ValueError('Calibration or speed selector differs')
    if any(old['source_hashes'][name]!=new['source_hashes'][name] for name in CONTROL_FILES):
        raise ValueError('Rule, candidate generation or physical control source differs')
    cases=[]
    for case_id in range(len(old['routes'])):
        initial=[next(r['info']['initial_scenario'] for r in rows if r['case_id']==case_id) for rows in streams]
        if any({k:v for k,v in x.items() if k!='initial'}!={k:v for k,v in initial[0].items() if k!='initial'} for x in initial[1:]):
            raise ValueError('Registered road or task differs')
        deltas={k:initial[1]['initial'][k]-value for k,value in initial[0]['initial'].items()}
        deltas['yaw']=math.atan2(math.sin(deltas['yaw']),math.cos(deltas['yaw']))
        row={'case_id':case_id,'measured_start_deltas':deltas}
        for label,summary,rows in zip(('euclidean','dubins_v1'),summaries,streams):
            metrics=next(r for r in summary['episodes'] if r['case_id']==case_id)
            row[label]={k:metrics[k] for k in ('success','termination_reason','physics_steps','progress',
                'interventions','plan_failures','mpc_failures','speed_errors_rms','jerk_rms','full_decision_times')}
            selected=[r for r in rows if r['case_id']==case_id]
            row[label]['wait_decisions']=sum(r['policy_action']==0 for r in selected)
            row[label]['rule_candidate_search_nodes']=sum((r['info'].get('rule_candidate_compute') or {}).get('search_nodes',0) for r in selected)
        cases.append(row)
    report={'scope':'All four paired known training roads, one rule execution per variant; descriptive planner engineering comparison, no PPO value, independent-training interval or realtime claim',
        'baseline':baseline.name,'candidate':candidate.name,'raw_audit_sha256':{b.name:sha(b/'raw_audit.json') for b in (baseline,candidate)},
        'script_sha256':sha(__file__),'same_registered_tasks':True,'unchanged_control_sources':list(CONTROL_FILES),
        'start_deltas_reported_without_case_exclusion':True,'rule_nodes_exclude_any_additional_environment_search_and_are_not_total_planner_nodes':True,
        'new_controls':0,'optimizer_updates':0,'cases':cases}
    write_json(output,report)
    print(json.dumps({'cases':len(cases),'successes':{label:sum(r[label]['success'] for r in cases) for label in ('euclidean','dubins_v1')}}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();compare(a.baseline,a.candidate,a.output)

"""Whole-audited scalar/batch road-SAT B3 contrast from identical frozen code.

No simulator connection. Execution differences and failed cases are reported,
never filtered to make a compute intervention look equivalent or successful.
"""
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
from scripts.compare_native_rule_planning import CONTROL_FILES
from scripts.evaluate_native_rule import require_finished_parent


def validate_pair(old,new):
    configs=[deepcopy(p['config']) for p in (old,new)]
    backends=[c.pop('road_collision_backend','scalar') for c in configs]
    if backends!=['scalar','batch_sat_v1']:
        raise ValueError('Expected scalar versus batch_sat_v1')
    if configs[0]!=configs[1] or old['routes']!=new['routes'] or len(old['routes'])!=4:
        raise ValueError('Require the same four tasks and every other environment setting')
    for key in ('calibration_sha256','rule_speed_policy'):
        if old[key]!=new[key]:raise ValueError('Calibration or rule speed selector differs')
    required=set(CONTROL_FILES)|{'navstack/planning.py','navstack/road_grid_batch.py',
                                 'navstack/config.py','scripts/evaluate_native_rule.py'}
    sources=[p['source_hashes'] for p in (old,new)]
    if (any(not required.issubset(s) for s in sources) or sources[0]!=sources[1]):
        raise ValueError('Require identical complete frozen source inventories and hashes')


def execution_deltas(left,right):
    ticks=[[t for row in rows for t in row['info']['trace']] for rows in (left,right)]
    paired=list(zip(*ticks))
    state_keys=('x','y','yaw','v','steering','acceleration','time')
    command_keys=('acceleration','steering')
    def maximum(section,keys):
        result={k:None for k in keys}
        for a,b in paired:
            for k in keys:
                value=b[section][k]-a[section][k]
                if section=='state' and k=='yaw':value=math.atan2(math.sin(value),math.cos(value))
                if not math.isfinite(value):raise ValueError('Nonfinite executed-state/control difference')
                result[k]=max(result[k] or 0.,abs(value))
        return result
    def meaningful_tick(t):
        # Frame/episode/request IDs and solver/wall timings are identities or
        # measurements, not differences in physical behavior. Keep actual plant
        # controls, actuator history, rewards, interventions and tracking values.
        evidence=t['backend_evidence']
        return {**{k:t[k] for k in ('state','command','requested_command','reward',
                    'reward_components','physical','tracking','intervention','controller_status')},
                **{k:evidence[k] for k in ('vehicle_control','speed_setpoint','actuation_mode','actuation_state') if k in evidence}}
    def plan_metadata(row):
        plan=row['info']['plan']
        return None if plan is None else {k:v for k,v in plan.items() if k!='planning_time'}
    actions=[[int(r['policy_action']) for r in rows] for rows in (left,right)]
    return dict(alignment='Control index since each measured reset; differing lengths and unmatched tails retained',
        actions=dict(scalar=actions[0],batch_sat_v1=actions[1]),
        same_action_sequence=actions[0]==actions[1],
        same_plan_metadata=[plan_metadata(r) for r in left]==[plan_metadata(r) for r in right],
        plan_metadata_does_not_include_unrecorded_full_paths=True,
        compared_controls=len(paired),unpaired_scalar_controls=len(ticks[0])-len(paired),
        unpaired_batch_controls=len(ticks[1])-len(paired),
        maximum_absolute_state_deltas=maximum('state',state_keys),
        maximum_absolute_command_deltas=maximum('command',command_keys),
        exact_observed_execution=(bool(paired) and len(ticks[0])==len(ticks[1]) and
            all(meaningful_tick(a)==meaningful_tick(b) for a,b in paired)))


def compare(baseline,candidate,output):
    baseline,candidate,output=Path(baseline),Path(candidate),Path(output)
    if output.exists():raise FileExistsError('Preserve comparison evidence')
    protocols=[];summaries=[];streams=[]
    for batch in (baseline,candidate):
        require_finished_parent(batch)
        if audit(batch)!=read(batch/'raw_audit.json'):
            raise ValueError('Whole-run raw audit is absent or stale')
        protocols.append(read(batch/'protocol.json'));summaries.append(read(batch/'summary.json'))
        with gzip.open(batch/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            streams.append([json.loads(line) for line in stream])
    validate_pair(*protocols)
    cases=[]
    for case_id in range(4):
        rows=[[r for r in stream if r['case_id']==case_id] for stream in streams]
        initial=[r[0]['info']['initial_scenario'] for r in rows]
        if {k:v for k,v in initial[0].items() if k!='initial'}!={k:v for k,v in initial[1].items() if k!='initial'}:
            raise ValueError('Actual task, road, obstacles or initial actuator evidence differ')
        deltas={k:initial[1]['initial'][k]-v for k,v in initial[0]['initial'].items()}
        deltas['yaw']=math.atan2(math.sin(deltas['yaw']),math.cos(deltas['yaw']))
        metrics={label:next(e for e in summary['episodes'] if e['case_id']==case_id)
                 for label,summary in zip(('scalar','batch_sat_v1'),summaries)}
        cases.append(dict(case_id=case_id,measured_start_deltas=deltas,
                          metrics=metrics,execution=execution_deltas(*rows)))
    result=dict(scope='All four known development roads, one actual rule run per backend, identical frozen code; descriptive compute/behavior contrast, no PPO contribution, independent-seed interval or realtime claim',
        baseline=baseline.name,candidate=candidate.name,
        raw_audit_sha256={b.name:sha(b/'raw_audit.json') for b in (baseline,candidate)},
        script_sha256=sha(__file__),identical_frozen_source_hashes=protocols[0]['source_hashes'],
        all_cases_and_failures_retained=True,new_controls=0,optimizer_updates=0,cases=cases)
    write_json(output,result)
    print(json.dumps(dict(cases=4,successes={label:sum(c['metrics'][label]['success'] for c in cases)
        for label in ('scalar','batch_sat_v1')},exact_observed_executions=sum(c['execution']['exact_observed_execution'] for c in cases))))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True);parser.add_argument('--candidate',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    compare(args.baseline,args.candidate,args.output)

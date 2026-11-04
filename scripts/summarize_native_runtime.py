"""Aggregate every audited evaluation frame,including failures and interventions."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.metrics import timing_summary
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent


def summarize_rows(rows,episodes,control_dt,decision_period,reported_vehicle_controls=None):
    groups=defaultdict(list)
    for row in rows:groups[row['case_id']].append(row)
    expected={row['case_id']:row for row in episodes}
    if set(groups)!=set(expected) or len(expected)!=len(episodes):
        raise ValueError('Require every registered episode exactly once')
    for case_id,selected in groups.items():
        episode=expected[case_id]
        if len(selected)!=episode['actions'] or sum(len(r['info']['trace']) for r in selected)!=episode['physics_steps']:
            raise ValueError('Decision/control inventory differs from complete endpoint')
    decisions=[row for selected in groups.values() for row in selected]
    ticks=[tick for row in decisions for tick in row['info']['trace']]
    if not ticks:raise ValueError('Nonempty measured evaluation required')
    plans=[row['info']['plan'] for row in decisions if row['info']['plan'] is not None
           and row['info']['plan']['status'] not in ('reference_reused','terminal_brake')]
    values={'environment_decision':[r['info']['environment_decision_seconds'] for r in decisions],
        'control_cycle':[t['control_cycle_seconds'] for t in ticks],
        'planner_search':[p['planning_time'] for p in plans],
        'execution_feedback':[r['info']['execution_feedback_seconds'] for r in decisions
                              if 'execution_feedback_seconds' in r['info']]}
    if any(not math.isfinite(v) or v<0 for series in values.values() for v in series):
        raise ValueError('Finite nonnegative execution timings required')
    modes=Counter(r['info']['mode'] for r in decisions)
    modes_controls=Counter()
    for row in decisions:modes_controls[row['info']['mode']]+=len(row['info']['trace'])
    actions=Counter(str(r['policy_action']) for r in decisions if type(r['policy_action']) is int)
    successes=[r for r in episodes if r['success']]
    controls=[];recovered=0
    for tick in ticks:
        evidence=tick['backend_evidence'];control=evidence.get('vehicle_control')
        if control is None and reported_vehicle_controls is not None:
            control=reported_vehicle_controls.get((evidence.get('episode_id'),evidence.get('frame')))
            recovered+=control is not None
        if control is not None:controls.append(control)
    counts={'cases':len(episodes),'successes':len(successes),
        'unsafe_episodes':sum(r['termination_reason'] in ('collision','out_of_bounds') for r in episodes),
        'decisions':len(decisions),'controls':len(ticks)}
    return {'counts':counts,'termination_counts':dict(Counter(r['termination_reason'] for r in episodes)),
        'latencies':{key:timing_summary(series,control_dt if key=='control_cycle' else decision_period)
                     for key,series in values.items()},
        'mode_decisions':dict(modes),'mode_controls':dict(modes_controls),'categorical_action_decisions':dict(actions),
        'controller_status_controls':dict(Counter(t['controller_status'] for t in ticks)),
        'planner_status_queries':dict(Counter(p['status'] for p in plans)),
        'expanded_nodes':sum(p.get('expanded_nodes',0) for p in plans),
        'intervention_controls':sum(t['intervention'] for t in ticks),
        'shared_shield_controls':sum(t['shared_shield_intervention'] for t in ticks),
        'stationary_controls_abs_v_below_0_05':sum(abs(t['state']['v'])<.05 for t in ticks),
        'negative_acceleration_request_controls':sum(t['requested_command']['acceleration']<0 for t in ticks),
        'actual_brake_controls_above_1e_6':sum(c['brake']>1e-6 for c in controls),
        'vehicle_control_coverage_controls':len(controls),'vehicle_control_missing_controls':len(ticks)-len(controls),
        'vehicle_controls_recovered_from_raw_same_frame':recovered,
        'success_only_travel_seconds':[r['simulation_seconds'] for r in successes],
        'all_episode_durations_seconds':[r['simulation_seconds'] for r in episodes],
        'per_case':[{'case_id':i,'success':e['success'],'termination_reason':e['termination_reason'],
            'controls':e['physics_steps'],'decisions':e['actions'],'simulation_seconds':e['simulation_seconds'],
            'progress_metres':e['progress'],'interventions':e['interventions'],
            'plan_failures':e['plan_failures'],'mpc_failures':e['mpc_failures']} for i,e in sorted(expected.items())]}


def analyze(batch,method,output):
    batch,output=Path(batch),Path(output)
    if output.exists():raise FileExistsError('Preserve prior runtime evidence')
    require_finished_parent(batch)
    folder=batch/method;raw=read(folder/'endpoint_raw_audit.json')
    for path,key in ((batch/'protocol.json','protocol_sha256'),(folder/'config.json','config_sha256'),
            (folder/'evaluation.json','evaluation_sha256'),(folder/'evaluation_trace.jsonl.gz','evaluation_trace_sha256')):
        if sha(path)!=raw[key]:raise ValueError('Timing input differs from complete raw audit')
    config=read(folder/'config.json')['env'];evaluation=read(folder/'evaluation.json')
    with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        rows=[json.loads(line) for line in stream]
    response_path=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(response_path)!=raw['bridge_response_sha256']:raise ValueError('Original native response stream differs')
    controls={}
    with gzip.open(response_path,'rt',encoding='utf-8') as stream:
        for line in stream:
            response=json.loads(line).get('response',{})
            if 'vehicle_control' not in response:continue
            key=(response.get('episode_id'),response.get('frame'));control=response['vehicle_control']
            if key in controls and controls[key]!=control:raise ValueError('Conflicting same-frame native control reports')
            controls[key]=control
    dt=config['control_dt'];period=dt*config['flat_hold'] if config['method']=='B1' else config['decision_dt']
    result=summarize_rows(rows,evaluation['episodes'],dt,period,controls)
    if result['counts']['controls']!=raw['evaluation_control_frames_verified']:
        raise ValueError('Not all audited evaluation controls included')
    if result['vehicle_control_missing_controls']:raise ValueError('Actual control report missing after same-frame raw join')
    report={'scope':__doc__,'batch':batch.name,'method':method,'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'analyzer_sha256':sha(Path(__file__)),'results':result,
        'timing_limits':'Environment decision timer excludes policy prediction/normalization and runner serialization. Control cycle excludes planning and the later candidate-feedback observation. These synchronized execution timings are not full-runner latency or a hard realtime certificate.',
        'mechanism_limits':'Parking brake,policy-selected waiting,dynamic braking,planner-failure fallback and shared shield are reported separately. Stationary time alone does not classify stagnation. No controller superiority or learned-feedback benefit inferred.',
        'new_controls':0,'optimizer_updates':0,'research_acceptance':'UNMET'}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'method':method,**result['counts'],'latencies':result['latencies'],
        'mode_controls':result['mode_controls'],'intervention_controls':result['intervention_controls']}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('batch','method','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();analyze(args.batch,args.method,args.output)

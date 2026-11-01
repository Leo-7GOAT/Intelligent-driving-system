"""Verify common parking assistance against every measured control and unchanged goal gate."""
import argparse
from dataclasses import asdict
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.vehicle import Command,VehicleState,wrap
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_native_control_delivery import control_errors
from scripts.evaluate_native_rule import require_finished_parent


def audit_arm(batch,method):
    batch=Path(batch).resolve();require_finished_parent(batch)
    rule=method=='B3';folder=batch if rule else batch/method;p=read(batch/'protocol.json')
    raw=read(folder/('raw_audit.json' if rule else 'endpoint_raw_audit.json'))
    config=EnvConfig.from_dict(p['config'] if rule else read(folder/'config.json')['env'])
    if not config.goal_parking_assist:raise ValueError('Parking assistance is not declared')
    helper_path=batch/'source/navstack/parking_assist.py'
    if sha(helper_path)!=p['source_hashes']['navstack/parking_assist.py']:raise ValueError('Frozen assist source changed')
    spec=importlib.util.spec_from_file_location('navstack._frozen_parking_assist',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    trace=folder/'evaluation_trace.jsonl.gz'
    if sha(trace)!=raw['trace_sha256' if rule else 'evaluation_trace_sha256']:raise ValueError('Actual trajectory changed')
    with gzip.open(trace,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    summary=read(batch/'summary.json') if rule else read(folder/'evaluation.json')
    episodes=summary['episodes'];cases=[];assistance=0;controls=0;blocked_parked_frames=0
    for case in sorted({r['case_id'] for r in rows}):
        selected=[r for r in rows if r['case_id']==case]
        initial=selected[0]['info']['initial_scenario'];goal=initial['goal'];state=VehicleState(**initial['initial'])
        distance=0.;ever=False;changed=0;count=0
        for row in selected:
            for tick in row['info']['trace']:
                nominal=Command(**tick['unassisted_command'])
                ever|=tick['unassisted_controller_status'] in ('direct','solved')
                command,intervened,eligible=helper.goal_parking_command(state,goal,nominal,config,ever and distance>.1)
                if (asdict(command)!=tick['requested_command'] or intervened!=tick['goal_parking_intervention']
                        or eligible!=tick['goal_parking_eligible']):raise ValueError('Applied goal assistance differs from the exact frozen rule')
                if tick['requested_command']['steering']!=nominal.steering:raise ValueError('Parking aid changes steering')
                if intervened and (tick['controller_status']!='goal_parking_assist' or not tick['intervention']):raise ValueError('Unreported parking takeover')
                after=VehicleState(**tick['state']);distance+=math.hypot(after.x-state.x,after.y-state.y);state=after
                changed+=int(intervened);count+=1
                parked=(math.hypot(state.x-goal[0],state.y-goal[1])<=config.goal_tolerance
                    and abs(wrap(state.yaw-goal[2]))<=config.goal_yaw_tolerance and abs(state.v)<=config.goal_speed)
                blocked_parked_frames+=int(parked and tick['intervention'])
        metric=next(e for e in episodes if e['case_id']==case)
        if metric['goal_parking_controls']!=changed or metric['physics_steps']!=count:raise ValueError('Assistance/physical frame counts differ')
        final=selected[-1]['info']['trace'][-1]
        if metric['success'] and (final['intervention'] or not (math.hypot(state.x-goal[0],state.y-goal[1])<=config.goal_tolerance
                and abs(wrap(state.yaw-goal[2]))<=config.goal_yaw_tolerance and abs(state.v)<=config.goal_speed)):
            raise ValueError('Success must satisfy the original physical goal conditions on a nonintervened terminal frame')
        cases.append({'case_id':case,'termination':metric['termination_reason'],'controls':count,'parking_overrides':changed,
            'final_frame_intervened':final['intervention'],'final_speed':state.v})
        assistance+=changed;controls+=count
    if sorted(r['case_id'] for r in cases)!=[0,1,2,3]:raise ValueError('All4known TRAINroads required')
    streams=list((batch/'infrastructure').glob('*_responses.jsonl.gz'))
    if len(streams)!=1 or sha(streams[0])!=raw['bridge_response_sha256']:raise ValueError('Complete measured control stream differs')
    engine=0
    with gzip.open(streams[0],'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line).get('response',{})
            if 'vehicle_control' not in r:continue
            if (control_errors(r['vehicle_control'],r['applied_vehicle_control'])
                    or r.get('control_delivery')!='batch_sync_v1' or r.get('applied_control_matches_request') is not True):
                raise ValueError('Actual engine control was not confirmed for this step')
            engine+=1
    if engine!=controls or controls!=raw['evaluation_controls_verified' if rule else 'evaluation_control_frames_verified']:
        raise ValueError('Measured physical frame inventory differs')
    success=sum(e['success'] for e in episodes);unsafe=sum(e['termination_reason'] in ('collision','out_of_bounds') for e in episodes)
    return {'method':method,'batch':batch.name,'version':helper.VERSION,'helper_sha256':sha(helper_path),
        'calibration_sha256':sha(batch/'calibration/calibration.json'),'successes':success,'cases':4,'unsafe_cases':unsafe,
        'controls':controls,'parking_overrides':assistance,'parking_override_fraction':assistance/controls,
        'case_outcomes':cases,'parked_intervened_frames':blocked_parked_frames,
        'original_goal_gate_and_nonintervened_terminal_frames_verified':True,'all_engine_controls_acknowledged':True,
        'raw_audit_sha256':sha(folder/('raw_audit.json' if rule else 'endpoint_raw_audit.json')),
        'protocol_sha256':sha(batch/'protocol.json'),'evaluation_trace_sha256':sha(trace),
        'new_PPO_updates':0,'new_supervised_updates':0,
        'configuration':{k:config.to_dict()[k] for k in ('goal_tolerance','goal_speed','goal_yaw_tolerance','parking','goal_parking_assist','control_dt','vehicle')}}


def analyze(restored_batches=None):
    paths=[('B1','20261101T104233573082Z_native_shared_parking_v2_B1'),('H','20261101T104236093121Z_native_shared_parking_v2_H'),('B3','20261101T104917Z_native_rule')]
    arms=[audit_arm(restored_batches[method] if restored_batches else ROOT/'runs'/name,method) for method,name in paths]
    for arm in arms[1:]:
        for key in ('helper_sha256','calibration_sha256','configuration'):
            if arm[key]!=arms[0][key]:raise ValueError('Shared parking capability differs by method:'+key)
    defect=audit_arm(restored_batches['v1_defect'] if restored_batches else ROOT/'runs/20261101T103609019454Z_native_shared_parking_B1','B1')
    return {'scope':__doc__,'script_sha256':sha(Path(__file__)),'arms':arms,'same_physical_parking_capability':True,
        'v1_defect_retained':defect,'v1_reason':'Braking continued after already parked,so the unchanged nonintervened-terminal success gate stayed false. V2 releases at the existing goal_speed;the success gate is not changed.',
        'unassisted_reference':{'batch':'20261101T100704463570Z_native_reflection_feedback_probe','B1successes':3,'bounds':1},
        'new_controls':sum(r['controls'] for r in arms)+defect['controls'],'new_PPO_updates':0,
        'limits':'Single known development seed and4known TRAINroads;frozen deployment transfers,not trained-under-wrapper policies or an isolated PPObenefit. Halso transfers from legacy delivery to sync with identical gains. Five healthy independent policies,full-task tests and120NEWfinalconditions remain unmet.',
        'research_acceptance':'UNMET'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();result=analyze()
    with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps([{k:r[k] for k in ('method','successes','controls','parking_overrides')} for r in result['arms']]),flush=True)

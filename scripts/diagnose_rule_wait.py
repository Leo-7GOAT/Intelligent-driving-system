"""Selected-state production queries after a stopped rule run, including interruptions.

An interrupted gzip prefix is never a completed episode or full-run raw audit.
Only the explicitly matched selected control frames support these offline queries.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import write_json
from navstack.planning import HybridAStar
from navstack.road_grid import RoadGrid
from navstack.world import World
from scripts.audit_native_action_availability import state_references,verify_source
from scripts.audit_native_method_endpoint import read,read_prefix,sha,_verify_ticks,verify_initial_actuator


def first_wait_state(rows,case_id,actual,acknowledgements,actuator_observation='off'):
    decisions=[r for r in rows if r['case_id']==case_id]
    first=next((i for i,r in enumerate(decisions) if r['policy_action']==0),None)
    if first is None:raise ValueError('No recorded first wait in this case')
    selected=decisions[:first+1]
    used,previous,episodes=set(),{},set()
    for row in selected:
        _verify_ticks(row['info']['trace'],actual,acknowledgements,{},used,previous,episodes,actuator_observation)
    if len(episodes)!=1:raise ValueError('Selected decisions cross bridge episodes')
    episode=next(iter(episodes));initial=decisions[0]['info']['initial_scenario']
    if any(r['info']['initial_scenario']!=initial for r in decisions):
        raise ValueError('Case initial task changed')
    begin=next((r for r in acknowledgements.values() if r.get('episode_id')==episode and 'reset_ticks' in r),None)
    if begin is None or begin.get('state')!=initial['initial']:
        raise ValueError('Initial task does not match measured begin state')
    if selected[0]['info']['trace'][0]['backend_frame']!=begin['frame']+1:
        raise ValueError('First control does not follow the measured reset frame')
    if actuator_observation!='off':verify_initial_actuator(initial,episode,acknowledgements)
    state=decisions[first-1]['info']['trace'][-1]['state'] if first else initial['initial']
    return {'decision':first,'state':state,'episode_id':episode,'initial':initial,
        'predecision_frame':decisions[first-1]['info']['trace'][-1]['backend_frame'] if first else begin['frame'],
        'selected_control_frames_verified':len(used),
        'measured_prefix_path':[initial['initial']]+[t['state'] for row in selected for t in row['info']['trace']],
        'preceding_decisions':decisions[max(0,first-3):first]}


def analyze(batch,case_id,output):
    batch,output=Path(batch),Path(output)
    if output.exists():raise FileExistsError('Preserve diagnostic evidence')
    final=read(batch/'guard_final.json')
    if not (batch/'server_cleanup.json').is_file() or read(batch/'server_cleanup.json').get('returncode')!=0:
        raise ValueError('Wait for owned cleanup before offline planning')
    if final.get('exit_code') is None:raise ValueError('Worker is not terminal')
    summary_complete=(batch/'summary.json').is_file() and read(batch/'summary.json').get('status')=='complete'
    if not summary_complete and final.get('reason')!='wall_deadline':
        raise ValueError('Unsupported incomplete lifecycle; inspect infrastructure first')
    protocol=read(batch/'protocol.json')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/'):
            verify_source(ROOT/name,batch/'source'/name,digest)
    config=EnvConfig.from_dict(protocol['config'])
    paths=list((batch/'infrastructure').glob('*_responses.jsonl.gz'))
    if len(paths)!=1:raise ValueError('Expected one bridge stream')
    responses,response_error=read_prefix(paths[0]);actual,acknowledgements={},{}
    for wrapper in responses:
        response=wrapper.get('response',{});episode=response.get('episode_id');request=response.get('request_id')
        if episode is not None and request is not None:
            key=(episode,request)
            if key in acknowledgements and acknowledgements[key]!=response:raise ValueError('Conflicting acknowledgement')
            acknowledgements[key]=response
        if 'vehicle_control' in response:
            key=(episode,response['frame'])
            if key in actual:raise ValueError('Duplicate measured control frame')
            actual[key]=response
    rows,trace_error=read_prefix(batch/'evaluation_trace.jsonl.gz')
    raw_audit=None
    if summary_complete and final.get('exit_code')==0 and final.get('reason') is None:
        from scripts.audit_native_rule import audit
        raw_audit=audit(batch)
    chosen=first_wait_state(rows,case_id,actual,acknowledgements,config.actuator_observation)
    scene_path=batch/f'scene_{case_id:04d}.json';scene_record=read(scene_path);scene=scene_record['scene']
    if (scene_record['initial']!=chosen['initial'] or scene['road_grid']['sha256']!=chosen['initial']['road_sha256']
            or scene['native_layout']['geometry_id']!=protocol['routes'][case_id]['geometry_id']):
        raise ValueError('Scene differs from registered task')
    begin=next(r for r in acknowledgements.values() if r.get('episode_id')==chosen['episode_id'] and 'reset_ticks' in r)
    if begin.get('scene')!=scene:
        raise ValueError('Saved scene differs from measured begin acknowledgement')
    # Use the unchanged decoder, planner and time parameterization; the holder
    # plant is never stepped and alternatives are not execution claims.
    print(json.dumps({'phase':'all_candidate_references','case_id':case_id,'first_wait':chosen['decision']}),flush=True)
    references=state_references(config,scene,chosen['state'])
    print(json.dumps({'phase':'candidate_queries_complete','reference_actions':references['reference_actions'],
        'mode_counts':references['mode_counts']}),flush=True)
    world=World(scene['width'],scene['height'],scene['static'],road=RoadGrid(scene['road_grid']))
    goal_queries=[]
    for multiplier in (1,4):
        planner_config=deepcopy(config.planner);planner_config.max_nodes*=multiplier
        print(json.dumps({'phase':'goal_query','node_budget':planner_config.max_nodes}),flush=True)
        result=HybridAStar(config.vehicle,planner_config).plan(
            tuple(chosen['state'][k] for k in ('x','y','yaw')),scene['goal'],world)
        goal_queries.append({'node_budget':planner_config.max_nodes,'status':result.status,
            'expanded_nodes':result.expanded_nodes,'failure_reason':result.failure_reason,
            'path':result.path.tolist(),'cost':result.cost})
        print(json.dumps({k:v for k,v in goal_queries[-1].items() if k!='path'}),flush=True)
    tracking=[]
    for row in chosen.pop('preceding_decisions'):
        ticks=row['info']['trace']
        tracking.append({'action':row['policy_action'],'mode':row['info']['mode'],
            'last_state':ticks[-1]['state'],'rule_candidate_compute':row['info'].get('rule_candidate_compute'),
            'maximum_lateral_error_m':max((t['tracking']['lateral'] or 0 for t in ticks),default=0),
            'maximum_heading_error_rad':max((abs(t['tracking']['heading'] or 0) for t in ticks),default=0),
            'controller_status_counts':dict(Counter(t['controller_status'] for t in ticks))})
    result={'scope':'Offline production queries at the measured state before first wait; only selected frames verified, not a full-run audit or executed alternative. Search failure is not global infeasibility.',
        'batch':batch.name,'case_id':case_id,'summary_reports_complete':summary_complete,
        'run_status':'complete_raw_audited' if raw_audit else 'interrupted_prefix_only',
        'full_run_raw_audit':raw_audit,
        'guard_reason':final.get('reason'),'trace_prefix_error':trace_error,'response_prefix_error':response_error,
        'protocol_sha256':sha(batch/'protocol.json'),'trace_sha256':sha(batch/'evaluation_trace.jsonl.gz'),
        'response_sha256':sha(paths[0]),'scene_sha256':sha(scene_path),'script_sha256':sha(__file__),
        'new_controls':0,'optimizer_updates':0,'selection':chosen,
        'preceding_tracking':tracking,'candidate_references':references,'direct_goal_queries':goal_queries}
    write_json(output,result)
    print(json.dumps({'case_id':case_id,'first_wait':chosen['decision'],'mode_counts':references['mode_counts'],
        'reference_actions':references['reference_actions'],'goal_queries':[{k:v for k,v in r.items() if k!='path'} for r in goal_queries]}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--case',type=int,default=3)
    p.add_argument('--output',required=True);a=p.parse_args();analyze(a.batch,a.case,a.output)

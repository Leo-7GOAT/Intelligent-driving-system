"""Compare frozen planner variants on one recorded initial state; no plant stepping."""
import argparse
from dataclasses import replace,asdict
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.planning import HybridAStar,PlanResult
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,read_prefix,sha
from scripts.audit_native_policy_sequence import observation_env


def probe(batch,case_id,output,speed_ceilings=()):
    batch,output=Path(batch).resolve(),Path(output)
    if output.exists():raise FileExistsError('Preserve every diagnostic attempt')
    rows,error=read_prefix(batch/'H/evaluation_trace.jsonl.gz')
    if error:raise ValueError('Require closed trace')
    selected=[r for r in rows if r['case_id']==case_id]
    if not selected or 'episode_metrics' not in selected[-1]['info']:raise ValueError('Require completed case')
    row=selected[0];episode=row['info']['trace'][0]['backend_evidence']['episode_id'];begins=[]
    for stream in (batch/'infrastructure').glob('*_responses.jsonl.gz'):
        wrappers,error=read_prefix(stream)
        if error:raise ValueError('Require closed bridge evidence')
        begins.extend(w['response'] for w in wrappers if w.get('response',{}).get('episode_id')==episode and 'scene' in w['response'])
    if len(begins)!=1:raise ValueError('Require unique measured reset')
    begin=begins[0];cfg=EnvConfig.from_dict(read(batch/'H/config.json')['env'])
    env=observation_env(cfg,begin['scene'],row['info']['initial_scenario'],begin=begin)
    if env.world.dynamic:raise ValueError('This bounded probe only supports static initial states')
    env.world.road_collision_backend=cfg.road_collision_backend
    env.reference=None;env._reference_selection=None;env._rule_plan_cache=None
    env.episode=dict(plan_times=[],plan_failures=0);captured=[]
    class Capture:
        def plan(self,start,goal,world):
            captured.append((start,goal))
            return PlanResult('no_path',np.empty((0,3)),None,0,0.,'diagnostic_capture')
    env.planner=Capture()
    try:
        _,detail=env._prepare_reference(row['policy_action'])
        if len(captured)!=1:raise ValueError('Expected exactly one planner call')
        start,goal=captured[0];config=replace(cfg.planner,max_nodes=32000,max_seconds=None);results=[]
        for label,analytic,transition in [('legacy_analytic',True,False),('constant_steer_no_analytic',False,False),('measured_slew_no_analytic',False,True)]:
            planner=HybridAStar(cfg.vehicle,replace(config,analytic_expansion=analytic))
            options=dict(initial_steering=env.state.steering,reference_speed=max(.1,env.state.v,env.decode_action(row['policy_action'])[1])) if transition else {}
            result=planner.plan(start,goal,env.world,**options)
            results.append(dict(variant=label,status=result.status,failure_reason=result.failure_reason,
                nodes=result.expanded_nodes,seconds=result.planning_time,path=result.path.tolist()))
        if len(speed_ceilings)>2 or any(not 0<speed<=1.5 for speed in speed_ceilings):raise ValueError('At most two bounded counterfactual speed ceilings')
        for speed in speed_ceilings:
            planner=HybridAStar(cfg.vehicle,replace(config,analytic_expansion=False))
            result=planner.plan(start,goal,env.world,initial_steering=env.state.steering,reference_speed=speed)
            results.append(dict(variant='counterfactual_slew_speed_ceiling',speed_ceiling=speed,
                status=result.status,failure_reason=result.failure_reason,nodes=result.expanded_nodes,
                seconds=result.planning_time,path=result.path.tolist()))
        atomic_json(output,dict(batch=batch.name,case_id=case_id,decision=0,action=row['policy_action'],
            trace_sha256=sha(batch/'H/evaluation_trace.jsonl.gz'),script_sha256=sha(Path(__file__)),
            start=start,goal=goal,measured_state=asdict(env.state),terminal_goal_prefix=detail.get('terminal_goal_prefix'),
            planner_budget=asdict(config),results=results,new_native_controls=0,
            scope='Read-only geometric root-cause probe. Legacy paths do not certify steering-slew feasibility, native-road acceptance, task success, or learning benefit.'))
        print([{k:r[k] for k in ('variant','status','nodes','seconds')} for r in results])
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--case-id',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--speed-ceilings',type=float,nargs='*',default=[])
    a=p.parse_args();probe(a.batch,a.case_id,a.output,a.speed_ceilings)

"""Offline model probe aligning search checks with the nine-offset reference tube."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.planning import HybridAStar,PlanResult
from navstack.trajectory import parameterize
from navstack.steering_retiming import retime
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,read_prefix,sha
from scripts.audit_native_policy_sequence import observation_env


def probe(batch,case_id,output,exact_counterexamples=False):
    batch,output=Path(batch),Path(output)
    if output.exists():raise FileExistsError('Preserve every model probe')
    rows,error=read_prefix(batch/'H/evaluation_trace.jsonl.gz')
    if error:raise ValueError('Require closed measured trace')
    rows=[r for r in rows if r['case_id']==case_id]
    if not rows or 'episode_metrics' not in rows[-1]['info']:raise ValueError('Require completed case')
    row=rows[0];episode=row['info']['trace'][0]['backend_evidence']['episode_id'];begins=[]
    for path in (batch/'infrastructure').glob('*_responses.jsonl.gz'):
        wrappers,error=read_prefix(path)
        if error:raise ValueError('Require closed bridge')
        begins.extend(w['response'] for w in wrappers if 'scene' in w.get('response',{}) and w['response']['episode_id']==episode)
    if len(begins)!=1:raise ValueError('Require one measured reset')
    begin=begins[0];cfg=EnvConfig.from_dict(read(batch/'H/config.json')['env'])
    env=observation_env(cfg,begin['scene'],row['info']['initial_scenario'],begin=begin)
    env.world.road_collision_backend=cfg.road_collision_backend
    env.reference=None;env._reference_selection=None;env._rule_plan_cache=None
    env.episode=dict(plan_times=[],plan_failures=0);captured=[]
    class Capture:
        def plan(self,start,goal,world):
            captured.append((start,goal))
            return PlanResult('no_path',np.empty((0,3)),None,0,0.,'capture')
    env.planner=Capture()
    try:
        _,detail=env._prepare_reference(row['policy_action'])
        if len(captured)!=1 or env.world.dynamic:raise ValueError('Initial static planner probe only')
        base_free=env.world.free_many;calls=[0]
        def tube_free(poses,vehicle,margin=0.):
            poses=np.asarray(poses,dtype=float);flat=poses.reshape(-1,3);valid=np.ones(len(flat),dtype=bool)
            for lat in (-.05,0.,.05):
                for turn in (-.05,0.,.05):
                    shifted=flat.copy();shifted[:,0]-=lat*np.sin(flat[:,2]);shifted[:,1]+=lat*np.cos(flat[:,2]);shifted[:,2]+=turn
                    valid &= base_free(shifted,vehicle,margin);calls[0]+=len(flat)
            return valid.reshape(poses.shape[:-1])
        points=()
        if exact_counterexamples:
            from navstack.reference_point_constraints import allowed,local_points
            evidence=row['info']['plan']['road_refinement']['probes']
            rejected=[point for reply in evidence for point in reply['reference_road_probe']['rejected_world']]
            if not rejected:raise ValueError('Require actual rejected points')
            raw=read(batch/'H/endpoint_raw_audit.json')
            if raw['evaluation_trace_sha256']!=sha(batch/'H/evaluation_trace.jsonl.gz'):
                raise ValueError('Require raw-verified measured trace')
            points=local_points(rejected,begin['scene']['native_layout'])
            from navstack.reference_road_refinement import planning_view
            cells=set(map(tuple,row['info']['plan']['road_refinement']['cells']))
            env.world=planning_view(env.world,cells,points);calls[0]=None
        else:env.world.free_many=tube_free
        planner=HybridAStar(cfg.vehicle,replace(cfg.planner,max_nodes=32000,max_seconds=None,analytic_expansion=False))
        result=planner.plan(*captured[0],env.world,initial_steering=env.state.steering,reference_speed=.25)
        record=dict(batch=batch.name,case_id=case_id,action=row['policy_action'],status=result.status,
            failure_reason=result.failure_reason,nodes=result.expanded_nodes,seconds=result.planning_time,
            model_footprint_checks=calls[0],path=result.path.tolist(),trace_sha256=sha(batch/'H/evaluation_trace.jsonl.gz'),
            exact_counterexamples=exact_counterexamples,local_points=points,
            constraint_source_hashes={name:sha(ROOT/name) for name in ['navstack/reference_point_constraints.py',
                'navstack/world.py','navstack/reference_road_refinement.py','navstack/planner_memo.py']},
            script_sha256=sha(Path(__file__)),layout=begin['scene']['native_layout'],new_native_controls=0,
            limits='Conservative raster model with nine-offset body checks;actual native road acceptance and tracking not established. Initial/goal nominal validity and task acceptance tolerances unchanged.')
        if result.status=='success':
            ref=retime(parameterize(result.path,env.decode_action(row['policy_action'])[1],env.state.v,cfg.vehicle,
                env.state.time,bool(detail.get('terminal_goal_prefix'))),cfg.vehicle,env.state.steering)
            record.update(retimed_seconds=float(ref.t[-1]),poses=np.c_[ref.x,ref.y,ref.yaw].tolist())
        atomic_json(output,record);print({k:v for k,v in record.items() if k not in ['path','poses','layout']})
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--case-id',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--exact-counterexamples',action='store_true')
    a=p.parse_args();probe(a.batch,a.case_id,a.output,a.exact_counterexamples)

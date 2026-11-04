"""Compare pose-only spatial planners on every static/curve known24 native start.

No policy inference,trajectory retiming,MPC,live client or new driving occurs.
This component benchmark does not certify the complete B2 native system.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(batch,output):
    batch,output=Path(batch).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every benchmark attempt')
    candidate=ROOT/'navstack/planning.py';candidate_bytes=candidate.read_bytes()
    candidate_digest=hashlib.sha256(candidate_bytes).hexdigest()
    script_digest=sha(Path(__file__))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    protocol=read(batch/'protocol.json');source=batch/'source'
    for name,digest in protocol['source_hashes'].items():
        if sha(source/name)!=digest:raise ValueError('Frozen source changed: '+name)
    if ((batch/'failure.json').exists() or read(batch/'summary.json')['status']!='complete'
            or read(batch/'guard_final.json')['exit_code']!=0
            or read(batch/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped'):
        raise ValueError('Require a closed native batch and owned-server cleanup')
    raw=read(batch/'H/endpoint_raw_audit.json')
    for relative,key in (('protocol.json','protocol_sha256'),('H/config.json','config_sha256'),
                         ('H/evaluation_trace.jsonl.gz','evaluation_trace_sha256')):
        if sha(batch/relative)!=raw[key]:raise ValueError('Native fixture bytes changed')
    bridge=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(bridge)!=raw['bridge_response_sha256']:raise ValueError('Native reset stream changed')
    first={}
    with gzip.open(batch/'H/evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);first.setdefault(row['case_id'],row['info']['initial_scenario'])
    begins={}
    with gzip.open(bridge,'rt',encoding='utf-8') as stream:
        for line in stream:
            response=json.loads(line).get('response',{})
            if 'scene' in response:
                case=response['scene']['condition']
                if case in begins:raise ValueError('Unexpected repeated evaluation reset')
                begins[case]=response
    if set(first)!=set(range(24)) or set(begins)!=set(first):
        raise ValueError('Require the complete original known24 inventory')
    # The only source overlay is the explicit B2 node cap. All other planner
    # functions and imports must be the original executed version.
    allowed={'grid_search','_grid_search','GridAStarPlanner'}
    def unchanged(payload):
        return [ast.dump(n,include_attributes=False) for n in ast.parse(payload).body
                if getattr(n,'name',None) not in allowed]
    if unchanged(candidate_bytes)!=unchanged((source/'navstack/planning.py').read_bytes()):
        raise ValueError('Overlay changes more than the registered grid search/budget adapter')
    sys.path.insert(0,str(source))
    import numpy as np
    from navstack.config import EnvConfig
    from navstack.planning import HybridAStar
    from navstack.road_grid import RoadGrid
    from navstack.world import World
    spec=importlib.util.spec_from_file_location('navstack._bounded_grid_component',candidate)
    overlay=importlib.util.module_from_spec(spec);sys.modules[spec.name]=overlay
    exec(compile(candidate_bytes,str(candidate),'exec'),overlay.__dict__)
    cfg=EnvConfig.from_dict(read(batch/'H/config.json')['env'])
    for module in ('navstack.planning','navstack.road_grid','navstack.world'):
        if not Path(sys.modules[module].__file__).resolve().is_relative_to(source):
            raise ValueError('Common spatial modules must load from the frozen native source')
    if protocol['reference_steering_transition']['max_nodes']!=32000:
        raise ValueError('Require the original registered32000node per-search ceiling')
    common=deepcopy(cfg.planner);common.max_nodes=32000;common.max_seconds=None
    output.mkdir(parents=True);rows=[]
    for case,initial in sorted(first.items()):
        snapshot=read(batch/'H'/f'scene_{case+1:04d}.json');scene=snapshot['scene'];begin=begins[case]
        if (snapshot['initial']!=initial or scene!=begin['scene']
                or initial['initial']!=begin['state'] or initial['dynamic']!=begin['dynamic']):
            raise ValueError('Saved initial state/geometry differs from its measured reset')
        if initial['dynamic']:continue
        road=RoadGrid(scene['road_grid'])
        if road.sha256!=initial['road_sha256'] or scene['goal']!=initial['goal'] or scene['static']!=initial['static']:
            raise ValueError('Spatial fixture geometry differs')
        state=initial['initial'];start=np.array([state['x'],state['y'],state['yaw']]);goal=np.array(initial['goal'])
        kind=protocol['native_validation']['manifest']['cases'][case].get('hazard') or {}
        family='static_obstacle' if kind.get('kind')=='static_vehicle' else 'curve_parking'
        row=dict(case_id=case,family=family,scene_sha256=sha(batch/'H'/f'scene_{case+1:04d}.json'),
                 measured_start=state,registered_goal=initial['goal'],road_sha256=road.sha256,planners={})
        arms=[('HybridAStar',HybridAStar),('GridAStar_body_spline_v1',overlay.GridAStarPlanner)]
        if case%2:arms.reverse()
        for label,cls in arms:
            params=deepcopy(common)
            if label.startswith('Grid'):params.grid_postprocess='body_spline_v1'
            world=World(scene['width'],scene['height'],scene['static'],road=road,
                        road_collision_backend=cfg.road_collision_backend)
            begin_time=time.perf_counter();result=cls(cfg.vehicle,params).plan(start,goal,world)
            elapsed=time.perf_counter()-begin_time
            if result.expanded_nodes>32000:raise ValueError('Node budget exceeded')
            detail=dict(status=result.status,failure_reason=result.failure_reason,expanded_nodes=result.expanded_nodes,
                        algorithm_cost=result.cost,wall_seconds=elapsed,path_samples=len(result.path),
                        planner_config=asdict(params),path_sha256=hashlib.sha256(result.path.tobytes()).hexdigest())
            if result.status=='success':
                path=result.path;ds=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)
                curvature=np.abs(np.diff(np.unwrap(path[:,2])))/np.maximum(ds,1e-12)
                detail.update(sampled_full_body_free=bool(world.free_many(path,cfg.vehicle).all()),
                    path_length_m=float(ds.sum()),maximum_sampled_curvature=float(curvature.max(initial=0)),
                    terminal_position_error_m=float(np.linalg.norm(path[-1,:2]-goal[:2])),
                    terminal_yaw_error_rad=float(abs((path[-1,2]-goal[2]+np.pi)%(2*np.pi)-np.pi)))
                if not detail['sampled_full_body_free']:raise ValueError('Returned path fails common full-body check')
            row['planners'][label]=detail
            with (output/'partial_records.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(case_id=case,planner=label,result=detail))+'\n')
        rows.append(row);print(json.dumps(dict(case_id=case,statuses={k:v['status'] for k,v in row['planners'].items()})),flush=True)
    if len(rows)!=16 or Counter(r['family'] for r in rows)!=dict(static_obstacle=8,curve_parking=8):
        raise ValueError('Retain all original eight static and eight curve fixtures')
    result=dict(scope=__doc__,research_acceptance='UNMET',batch=batch.name,source_protocol_sha256=sha(batch/'protocol.json'),
        source_trace_sha256=sha(batch/'H/evaluation_trace.jsonl.gz'),source_bridge_sha256=sha(bridge),
        frozen_planning_sha256=sha(source/'navstack/planning.py'),bounded_grid_overlay_sha256=candidate_digest,
        benchmark_script_sha256=script_digest,repository_commit=commit,repository_has_local_changes=dirty,
        vehicle_config=asdict(cfg.vehicle),fixed_max_nodes_per_search=32000,max_seconds=None,
        pose_only_search=True,initial_speed_and_steering_not_used_for_search=True,
        all_cases=rows,status_counts={name:dict(Counter(r['planners'][name]['status'] for r in rows)) for name,_ in arms},
        new_controls=0,new_optimizer_updates=0,
        limits='All16static/curve known-development starts to their registered terminal goals,not PPO-selected subgoals or full native reference refinement.Pose-only search excludes steering slew,time parameterization,MPC and new execution.Grid uses bounded max_nfev150 spline fitting plus continuous spatial certificate;node counts exclude raster construction and postprocessing,whose cost remains in total wall time.Hybrid uses the saved base analytic/heuristic configuration,not the native steering-transition adapter.Algorithm costs have different definitions;compare physical path lengths separately.One alternating-order pass on a shared workstation is exploratory,not real-time or independent-seed inference.Native B2and other ablations remain pending.')
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(fixtures=len(rows),status_counts=result['status_counts'],new_controls=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();compare(args.batch,args.output)

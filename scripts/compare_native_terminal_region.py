"""Offline exact-pose/unchanged-task-region comparison on all registered queries."""
import argparse
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path.insert(0,str(ROOT))
import torch
from navstack.config import EnvConfig
from navstack.mpc import MPC
from navstack.planning import analytic_connection
from navstack.road_grid import RoadGrid
from navstack.trajectory import parameterize,dynamic_safe
from navstack.vehicle import VehicleState,Command
from navstack.world import World,MovingObstacle
from scripts.audit_native_method_endpoint import read,sha
module_path=Path(__file__).resolve().parents[1]/'navstack/terminal_region.py'
spec=importlib.util.spec_from_file_location('terminal_region_prototype',module_path)
region=importlib.util.module_from_spec(spec);spec.loader.exec_module(region)


def compare(batch,queries,output):
    batch,queries,output=map(Path,(batch,queries,output))
    if output.exists():raise FileExistsError('Preserve prior region diagnosis')
    started=time.monotonic();raw=read(batch/'B1/endpoint_raw_audit.json');manifest=read(queries/'manifest.json')
    if (sha(batch/'B1/endpoint_raw_audit.json')!=manifest['raw_audit_sha256']
            or sha(queries/'queries.jsonl')!=manifest['query_log_sha256']):raise ValueError('Query source differs')
    protocol=read(batch/'protocol.json')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:raise ValueError('Use exact frozen runtime')
    records=[json.loads(line) for line in (queries/'queries.jsonl').read_text(encoding='utf-8').splitlines()]
    wanted={(r['episode_id'],r['frame_before']) for r in records};responses={};begins={}
    path=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(path)!=raw['bridge_response_sha256']:raise ValueError('Raw bridge bytes differ')
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            value=json.loads(line)['response']
            if 'scene' in value:begins[value['episode_id']]=value
            if (value.get('episode_id'),value.get('frame')) in wanted:responses[value['episode_id'],value['frame']]=value
    if set(responses)!=wanted:raise ValueError('Measured query snapshot missing')
    cfg=EnvConfig.from_dict(read(batch/'B1/config.json')['env']);torch.set_num_threads(1);results=[]
    for row in records:
        if time.monotonic()-started>600:raise TimeoutError('Offline goal-region budget expired')
        before=time.monotonic();response=responses[row['episode_id'],row['frame_before']];scene=begins[row['episode_id']]['scene']
        if response['state']!=row['state']:raise ValueError('Query state differs from exact native snapshot')
        state=VehicleState(**row['state']);world=World(scene['width'],scene['height'],static=scene['static'],
            road=RoadGrid(scene['road_grid']),road_collision_backend=cfg.road_collision_backend)
        world.dynamic=[MovingObstacle(**r) for r in response.get('dynamic',[])];goal=scene['goal']
        connection=region.connect_region((state.x,state.y,state.yaw),goal,world,cfg.vehicle,
            cfg.goal_tolerance,cfg.goal_yaw_tolerance,analytic_connection)
        result={'condition':row['condition'],'training_episode':row['training_episode'],'control_index':row['control_index'],
            'state':row['state'],'goal':goal,'old_oracle_mode':row.get('oracle_mode'),
            'exact_goal_full_body_free':world.free(goal,cfg.vehicle),'region_connection_found':connection is not None}
        if connection is not None:
            target=connection['target'];distance=math.dist(target[:2],goal[:2]);angle=abs((target[2]-goal[2]+math.pi)%(2*math.pi)-math.pi)
            if distance>cfg.goal_tolerance or angle>cfg.goal_yaw_tolerance:raise ValueError('Candidate escapes original task gates')
            result.update(target=target,position_offset_metres=distance,yaw_offset_rad=angle,
                exact_goal_connection=connection['exact_goal'],analytic_queries=connection['queries'],
                path_points=len(connection['path']),path_length=sum(math.dist(a[:2],b[:2]) for a,b in zip(connection['path'],connection['path'][1:])))
            try:
                reference=parameterize(connection['path'],.75,state.v,cfg.vehicle,start_time=state.time,stop=True)
                result['CV_dynamic_safe']=dynamic_safe(reference,world,cfg.vehicle,cfg.prediction_seconds,cfg.control_dt)
                control=MPC(cfg.vehicle,cfg.control_dt,cfg.mpc_horizon).step(state,reference,Command(**response['command']))
                result.update(controller_status=control.status,controller_command={'acceleration':control.command.acceleration,'steering':control.command.steering})
            except ValueError as exc:result['trajectory_rejection']=str(exc)
        result['seconds']=time.monotonic()-before;results.append(result)
        print(json.dumps({k:result.get(k) for k in ('condition','control_index','old_oracle_mode','exact_goal_full_body_free','region_connection_found','controller_command','seconds')}),flush=True)
    report={'scope':'Offline TRAINING-state goal-pose contract diagnostic;original position/yaw/speed success thresholds retained. No tracking,driving or improvement claim.',
        'batch':batch.name,'query_manifest_sha256':sha(queries/'manifest.json'),'script_sha256':sha(Path(__file__)),
        'region_source_sha256':sha(module_path),'cases':results,'new_physical_controls':0,'optimizer_updates':0,
        'production_behavior_changed':False,'wall_seconds':time.monotonic()-started}
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('batch','queries','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();compare(a.batch,a.queries,a.output)

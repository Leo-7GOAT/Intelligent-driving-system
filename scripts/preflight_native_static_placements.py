"""Model-only construction checks for a new, explicitly separate TRAIN version.

All registered proposals are retained. No CARLA state is stepped, no policy is
queried, and no historical outcome is changed. Native measurement is still needed.
"""
import argparse
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import EnvConfig
from navstack.native_hazards import resolve_native_hazard,projected_box_strips
from navstack.planning import HybridAStar
from navstack.research import atomic_json
from navstack.road_grid import RoadGrid
from navstack.trajectory import parameterize
from navstack.world import World
from scripts.audit_native_method_endpoint import read,read_prefix,sha


def transformed_body(vertices,old_pose,new_pose):
    angle=new_pose[2]-old_pose[2];c,s=math.cos(angle),math.sin(angle)
    return [[new_pose[0]+c*(x-old_pose[0])-s*(y-old_pose[1]),
             new_pose[1]+s*(x-old_pose[0])+c*(y-old_pose[1])] for x,y in vertices]


def check(scene,initial,config,spec):
    old=scene['native_hazard'];new=resolve_native_hazard(scene['native_layout'],spec)
    vertices=transformed_body(old['static_body_vertices_local_xy'],old['spawn_pose'],new['spawn_pose'])
    static=projected_box_strips(vertices);road=RoadGrid(scene['road_grid'])
    supported=not any(road.intersects(np.array([[a,b],[c,b],[c,d],[a,d]])) for a,b,c,d in static)
    world=World(scene['width'],scene['height'],static,road=road)
    world.road_collision_backend=config.road_collision_backend
    start=tuple(initial[k] for k in ('x','y','yaw'));goal=tuple(scene['goal'])
    result=dict(hazard_fully_road_supported=bool(supported),start_free=bool(world.free(start,config.vehicle)),
                goal_free=bool(world.free(goal,config.vehicle)),eligible_model_proposal=False)
    if not all(result[k] for k in ('hazard_fully_road_supported','start_free','goal_free')):
        return result,None
    plan=HybridAStar(config.vehicle,config.planner).plan(start,goal,world)
    result.update(status=plan.status,nodes=plan.expanded_nodes,reason=plan.failure_reason)
    if plan.status!='success':return result,None
    result['path_samples_free']=all(world.free(p,config.vehicle) for p in plan.path)
    try:
        reference=parameterize(plan.path,1.5,initial['v'],config.vehicle,initial['time'],True)
        result['timed_samples_free']=all(world.free((x,y,yaw),config.vehicle)
            for x,y,yaw in zip(reference.x,reference.y,reference.yaw))
        result['eligible_model_proposal']=result['path_samples_free'] and result['timed_samples_free']
    except ValueError as exc:result['trajectory_rejection']=str(exc)
    return result,plan.path


def run(batch,output):
    batch,output=Path(batch).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every proposal attempt')
    protocol=read(batch/'protocol.json');raw=read(batch/'H/endpoint_raw_audit.json')
    if raw['protocol_sha256']!=sha(batch/'protocol.json'):raise ValueError('Raw protocol binding changed')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:raise ValueError('Frozen model source differs')
    trace=batch/'H/evaluation_trace.jsonl.gz';stream=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(trace)!=raw['evaluation_trace_sha256'] or sha(stream)!=raw['bridge_response_sha256']:
        raise ValueError('Measured inputs changed')
    rows,error=read_prefix(trace)
    if error:raise ValueError(error)
    bridge,error=read_prefix(stream)
    if error:raise ValueError(error)
    begins={x['response']['episode_id']:x['response'] for x in bridge if 'scene' in x.get('response',{})}
    first={}
    for row in rows:first.setdefault(row['case_id'],row)
    if set(first)!=set(range(12)):raise ValueError('Require the complete known static inventory')
    config=EnvConfig.from_dict(read(batch/'H/config.json')['env'])
    if config.planner.max_seconds is not None:raise ValueError('Fixed node budget required')
    proposals=[dict(version='native_hazard_v1',kind='static_vehicle',static_geometry='projected_box_strips12_v1',
                    distance_m=distance,lateral_m=lateral) for distance in (10.,13.,16.) for lateral in (-1.5,0.,1.5)]
    output.mkdir(parents=True)
    atomic_json(output/'registration.json',dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json'),
        script_sha256=sha(Path(__file__)),proposals=proposals,cases=list(range(12)),
        planner_node_budget=config.planner.max_nodes,maximum_searches=108,
        selection='Retain all proposals;eligibility independent of policy actions or scores',
        scope='Known TRAIN geometry proposals only;new version requires fresh native measurements'))
    results=[]
    for case,row in sorted(first.items()):
        begin=begins[row['info']['trace'][0]['backend_evidence']['episode_id']]
        if begin['dynamic']:raise ValueError('Static scenes only')
        for index,spec in enumerate(proposals):
            try:
                result,path=check(begin['scene'],begin['state'],config,spec)
            except ValueError as exc:result,path=dict(rejection=str(exc),eligible_model_proposal=False),None
            result.update(case_id=case,proposal_index=index,spec=spec)
            if path is not None:
                target=output/f'case{case}_proposal{index}_path.npy';np.save(target,path);result['path_sha256']=sha(target)
            results.append(result);atomic_json(output/'progress.json',dict(results=results))
        print(dict(case_id=case,eligible=sum(r['eligible_model_proposal'] for r in results if r['case_id']==case)),flush=True)
    atomic_json(output/'complete.json',dict(results=results,new_controls=0,new_updates=0,new_final_conditions=0,
        limits='Transformed old measured boxes are proposals,not new native measurements. Model eligibility does not prove tracking or continuous safety. Original inventories and scores unchanged;do not retrospectively filter them or claim PPO benefit.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.batch,args.output)

from pathlib import Path
import sys,math,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.planning import HybridAStar
from navstack.road_grid import RoadGrid
from navstack.world import World
from navstack.native_hazards import resolve_native_hazard,projected_box_strips
from navstack.disc_connectivity import necessary_connectivity
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha

b=ROOT/'runs/20261102T101413887816Z_native_diverse_TRAIN36_teacher'
out=ROOT/'runs/static_geometry_model_preflight_20261102'
if out.exists():raise FileExistsError('Preserve all offline attempts')
out.mkdir();cfg=EnvConfig.from_dict(read(b/'source_config.json')['env']);cfg.planner.max_seconds=None
atomic_json(out/'registration.json',dict(source_batch=b.name,source_config_sha256=sha(b/'source_config.json'),
 cases=list(range(12,24)),planner_nodes=cfg.planner.max_nodes,script_sha256=sha(Path(__file__)),
 repair_candidates=[dict(distance_m=13.,lateral_m=x) for x in (1.5,-1.5)]+[dict(distance_m=x,lateral_m=0.) for x in (10.,12.,14.,16.,18.,8.)],
 scope='Read-only geometry/planning preflight of known TRAIN cases. No policy inference,physics execution,learning or final conditions. Failed old version preserved. Repairs are geometric proposals requiring future native measurements.'))
results=[]
def check(case,scene,initial,static,label):
 world=World(scene['width'],scene['height'],static,road=RoadGrid(scene['road_grid']));world.road_collision_backend=cfg.road_collision_backend
 start=tuple(initial[k] for k in ('x','y','yaw'));goal=scene['goal']
 connection=necessary_connectivity(world.road,static,start,goal,cfg.vehicle,cfg.goal_tolerance)
 plan=HybridAStar(cfg.vehicle,cfg.planner).plan(start,goal,world)
 result=dict(case_id=case,label=label,connectivity=connection,status=plan.status,nodes=plan.expanded_nodes,seconds=plan.planning_time,path_poses=len(plan.path))
 if plan.status=='success':
  if not all(world.free(pose,cfg.vehicle) for pose in plan.path):raise ValueError('Certificate path collides')
  import numpy as np
  np.save(out/(label+'_case'+str(case)+'_path.npy'),plan.path)
 results.append(result);atomic_json(out/'heartbeat.json',dict(results=results,epoch=time.time()));print({k:result[k] for k in ('case_id','label','status','nodes','seconds')},flush=True)
 return result
for case in range(12,24):
 record=read(b/('scene_%04d.json'%case));scene=record['scene'];check(case,scene,record['initial']['initial'],scene['static'],'original')
record=read(b/'scene_0014.json');scene=record['scene'];old=scene['native_hazard'];ox,oy,oa=old['spawn_pose']
for index,candidate in enumerate(read(out/'registration.json')['repair_candidates']):
 spec=dict(version='native_hazard_v1',kind='static_vehicle',static_geometry='projected_box_strips12_v1',**candidate)
 try:new=resolve_native_hazard(scene['native_layout'],spec)
 except ValueError as error:
  results.append(dict(case_id=14,label='repair'+str(index),rejected=str(error)));continue
 nx,ny,na=new['spawn_pose'];angle=na-oa;c,s=math.cos(angle),math.sin(angle)
 vertices=[[nx+c*(x-ox)-s*(y-oy),ny+s*(x-ox)+c*(y-oy)] for x,y in old['static_body_vertices_local_xy']]
 result=check(14,scene,record['initial']['initial'],projected_box_strips(vertices),'repair'+str(index));result['proposed_hazard']=spec
 # Keep every registered candidate;no policy/outcome-dependent early selection.
atomic_json(out/'complete.json',dict(results=results,new_controls=0,new_updates=0,new_final_conditions=0,
 scope='Known TRAIN geometric proposals only;transformed old measured body corners are not new CARLAmeasurements. No current inventory or running training was changed.'))

"""Opt-in planning-only constraints from frozen-v4 lane predicate queries.

Python3.7 imports the pure query helpers. Physical worlds and task scoring are
never edited; only a private planning view may gain occupied cells.
"""
import copy
import hashlib
import json
import math

VERSION='sampled_native_reference_refinement_v1'


def settings(protocol):
    value=protocol.get('reference_road_refinement')
    if value is None:return None
    if (not isinstance(value,dict) or type(value.get('max_attempts')) is not int
            or value!={'version':VERSION,'max_attempts':3} or not protocol.get('evaluation_only')
            or protocol.get('methods') not in (['H'],['H0'])):
        raise ValueError('Refinement prototype requires explicit frozen H/H0 evaluation registration')
    return value


def pose_digest(poses):
    return hashlib.sha256(json.dumps(poses,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def footprint_points(pose,vehicle,layout):
    x,y,yaw=pose;co,si=math.cos(yaw),math.sin(yaw)
    angle=math.radians(layout['angle']);a,b=math.cos(angle),math.sin(angle)
    ox,oy=layout['origin'];z=layout['spawn_z']-.6
    return [[ox+(x+f*co-l*si)*a+(y+f*si+l*co)*b,
             oy+(x+f*co-l*si)*b-(y+f*si+l*co)*a,z]
        for f in (-vehicle['rear'],0.,vehicle['front']*.5,vehicle['front'])
        for l in (-vehicle['width']*.5,0.,vehicle['width']*.5)]


def query_points(poses,vehicle,layout,contains,location):
    if (not isinstance(poses,list) or not 1<=len(poses)<=1024
            or any(not isinstance(p,list) or len(p)!=3 or not all(type(v) in (int,float) and math.isfinite(v) for v in p) for p in poses)):
        raise ValueError('Bounded finite reference poses required')
    bad=set();samples=0
    for x,y,yaw in poses:
        for lat in (-.05,0.,.05):
            for turn in (-.05,0.,.05):
                shifted=[x-lat*math.sin(yaw),y+lat*math.cos(yaw),yaw+turn]
                for xyz in footprint_points(shifted,vehicle,layout):
                    point=location(x=xyz[0],y=xyz[1],z=xyz[2]);samples+=1
                    if not contains(point):bad.add((point.x,point.y,point.z))
    ordered=sorted(bad)
    return dict(version=VERSION,pose_sha256=pose_digest(poses),sample_count=samples,
                rejected_world=[list(p) for p in ordered[:8192]],truncated=len(ordered)>8192)


def planning_view(world,cells):
    view=copy.copy(world);view.road=copy.deepcopy(world.road)
    for row,column in cells:
        if type(row) is not int or type(column) is not int or not 0<=row<view.road.size or not 0<=column<view.road.size:
            raise ValueError('Invalid planning-only road cell')
        view.road.collision_drivable[row,column]=False
    return view


def rejected_cells(points,layout,road):
    angle=math.radians(layout['angle']);c,s=math.cos(angle),math.sin(angle);ox,oy=layout['origin'];cells=set()
    for point in points:
        if len(point)!=3 or not all(math.isfinite(v) for v in point):raise ValueError('Invalid query counterexample')
        dx,dy=point[0]-ox,point[1]-oy;x,y=dx*c+dy*s,dx*s-dy*c
        column,row=math.floor(x/road.dx),math.floor(y/road.dy)
        if 0<=row<road.size and 0<=column<road.size:cells.add((row,column))
    return cells


def prepare_reference(env,action,prepare,validate,options):
    owner=env.world
    if getattr(env,'_refinement_world_owner',None) is not owner:
        env._refinement_world_owner=owner;env._refinement_cells=set();env._validated_road_reference=None
    probes=[];attempts=[]
    for _ in range(options['max_attempts']):
        reuse=(env.reference is not None and env.reference is env._validated_road_reference
               and int(action)!=0 and env.reusable_reference(*env.decode_action(action)))
        env.world=owner if reuse else planning_view(owner,env._refinement_cells)
        try:mode,detail=prepare(action)
        finally:env.world=owner
        detail=dict(status='wait',expanded_nodes=0,planning_time=0.) if detail is None else dict(detail)
        attempts.append({k:detail.get(k) for k in ('status','expanded_nodes','planning_time')})
        if mode!='reference':break
        if env.reference is env._validated_road_reference:break
        poses=[[float(x),float(y),float(yaw)] for x,y,yaw in zip(env.reference.x,env.reference.y,env.reference.yaw)]
        reply=validate(poses);probes.append(reply)
        result=reply['reference_road_probe']
        if result['version']!=VERSION or result['pose_sha256']!=pose_digest(poses):raise ValueError('Unbound road query reply')
        if not result['rejected_world'] and not result['truncated']:
            env._validated_road_reference=env.reference;break
        env._refinement_cells.update(rejected_cells(result['rejected_world'],env.scene_evidence['native_layout'],owner.road))
        env.reference=None;env._reference_selection=None;env._rule_plan_cache=None
    else:
        mode='plan_failed';detail.update(status='trajectory_failed',failure_reason='native_reference_road_refinement_exhausted')
        env.episode['plan_failures']+=1
    detail['road_refinement']=dict(version=VERSION,cells=sorted([list(c) for c in env._refinement_cells]),
        attempts=attempts,probes=probes,rejected_references=sum(bool(p['reference_road_probe']['rejected_world']) for p in probes),
        total_expanded_nodes=sum(a.get('expanded_nodes') or 0 for a in attempts),
        query_samples=sum(p['reference_road_probe'].get('sample_count',0) for p in probes))
    return mode,detail

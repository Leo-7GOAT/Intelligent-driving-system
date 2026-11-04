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
    from .fixed_lowlevel_training import registered
    allowed=protocol.get('methods') in (['H'],['H0'])
    if not allowed:
        from .shared_rule_registration import registered as rule_registered
        allowed=rule_registered(protocol)
    if (not isinstance(value,dict) or type(value.get('max_attempts')) is not int
            or value!={'version':VERSION,'max_attempts':3} or not (protocol.get('evaluation_only') is True or registered(protocol))
            or not allowed):
        raise ValueError('Refinement prototype requires explicit frozen H/H0 evaluation registration')
    from .reference_compute import enabled
    result=dict(value,stop_repeated_references=True) if enabled(protocol) else value
    from .reference_point_constraints import settings as point_settings
    return dict(result,point_constraints=True) if point_settings(protocol) else result


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


def planning_view(world,cells,points=None):
    view=copy.copy(world);view.road=copy.deepcopy(world.road)
    for row,column in cells:
        if type(row) is not int or type(column) is not int or not 0<=row<view.road.size or not 0<=column<view.road.size:
            raise ValueError('Invalid planning-only road cell')
        view.road.collision_drivable[row,column]=False
    if points is not None:
        from .reference_point_constraints import VERSION as POINT_VERSION,signature
        view._reference_point_constraints=tuple(sorted(points));view._reference_point_constraint_version=POINT_VERSION
        signature(view)
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
        env._refinement_world_owner=owner;env._refinement_cells=set();env._refinement_points=set();env._validated_road_reference=None
    probes=[];attempts=[];rejected_hashes=set();shortcut=None;point_budget_exceeded=False
    exact_shortcut=(options.get('stop_repeated_references',False)
                    and env.config.planner.max_seconds is None)
    for _ in range(options['max_attempts']):
        reuse=(env.reference is not None and env.reference is env._validated_road_reference
               and int(action)!=0 and env.reusable_reference(*env.decode_action(action)))
        env.world=owner if reuse else planning_view(owner,env._refinement_cells,
            env._refinement_points if options.get('point_constraints') else None)
        try:mode,detail=prepare(action)
        finally:env.world=owner
        detail=dict(status='wait',expanded_nodes=0,planning_time=0.) if detail is None else dict(detail)
        attempts.append({k:detail.get(k) for k in ('status','expanded_nodes','planning_time')})
        if detail.get('joint_speed_search') is not None:
            attempts[-1]['joint_speed_search']=detail['joint_speed_search']
        if mode!='reference':break
        if env.reference is env._validated_road_reference:break
        poses=[[float(x),float(y),float(yaw)] for x,y,yaw in zip(env.reference.x,env.reference.y,env.reference.yaw)]
        if exact_shortcut and pose_digest(poses) in rejected_hashes:
            env.reference=None;env._reference_selection=None;env._rule_plan_cache=None
            mode='plan_failed';detail.update(status='trajectory_failed',failure_reason='native_reference_road_refinement_stalled')
            env.episode['plan_failures']+=1;shortcut='identical_reference_already_rejected_in_this_frame';break
        reply=validate(poses);probes.append(reply)
        result=reply['reference_road_probe']
        if result['version']!=VERSION or result['pose_sha256']!=pose_digest(poses):raise ValueError('Unbound road query reply')
        if not result['rejected_world'] and not result['truncated']:
            env._validated_road_reference=env.reference;break
        rejected_hashes.add(result['pose_sha256'])
        added=rejected_cells(result['rejected_world'],env.scene_evidence['native_layout'],owner.road)-env._refinement_cells
        geometry_changed=any(owner.road.collision_drivable[row,column] for row,column in added)
        env._refinement_cells.update(added)
        env.reference=None;env._reference_selection=None;env._rule_plan_cache=None
        if options.get('point_constraints'):
            from .reference_point_constraints import local_points,MAX_POINTS
            points=set(local_points(result['rejected_world'],env.scene_evidence['native_layout']))
            proposed=env._refinement_points|points
            if len(proposed)>MAX_POINTS:
                point_budget_exceeded=True;shortcut='exact_point_constraint_budget_exhausted'
                mode='plan_failed';detail.update(status='trajectory_failed',failure_reason=shortcut)
                env.episode['plan_failures']+=1;break
            geometry_changed=geometry_changed or bool(points-env._refinement_points)
            env._refinement_points=proposed
        if exact_shortcut and not geometry_changed:
            mode='plan_failed';detail.update(status='trajectory_failed',failure_reason='native_reference_road_refinement_stalled')
            env.episode['plan_failures']+=1;shortcut='rejection_adds_no_new_planning_constraint';break
    else:
        mode='plan_failed';detail.update(status='trajectory_failed',failure_reason='native_reference_road_refinement_exhausted')
        env.episode['plan_failures']+=1
    detail['road_refinement']=dict(version=VERSION,cells=sorted([list(c) for c in env._refinement_cells]),
        attempts=attempts,probes=probes,rejected_references=sum(bool(p['reference_road_probe']['rejected_world']) for p in probes),
        total_expanded_nodes=sum(a['joint_speed_search']['total_expanded_nodes'] if a.get('joint_speed_search') is not None
            else a.get('expanded_nodes') or 0 for a in attempts),
        query_samples=sum(p['reference_road_probe'].get('sample_count',0) for p in probes))
    if options.get('stop_repeated_references',False):
        from .reference_compute import VERSION as COMPUTE_VERSION
        detail['road_refinement'].update(compute_profile=COMPUTE_VERSION,shortcut=shortcut)
    if options.get('point_constraints'):
        from .reference_point_constraints import VERSION as POINT_VERSION
        detail['road_refinement']['point_constraints']=dict(version=POINT_VERSION,
            points=sorted([list(p) for p in env._refinement_points]),budget_exceeded=point_budget_exceeded)
    return mode,detail

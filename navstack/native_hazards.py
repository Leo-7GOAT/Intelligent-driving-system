"""Versioned route-relative actor specifications; compatible with Python3.7.

This resolves spawn geometry only. CARLA owns all actor motion and measured
states; the policy receives measured positions/velocities, not future scripts.
"""
import math


def resolve_native_hazard(layout,spec):
    allowed={'version','kind','distance_m','lateral_m','speed_m_s','start_delay_s'}
    if not isinstance(spec,dict) or set(spec)-allowed or spec.get('version')!='native_hazard_v1':
        raise ValueError('Invalid native hazard schema/version')
    kind=spec.get('kind')
    if kind not in ('static_vehicle','crossing_walker'):raise ValueError('Unknown native hazard kind')
    values={key:spec.get(key,default) for key,default in
            [('distance_m',10.),('lateral_m',0. if kind=='static_vehicle' else -3.5),('speed_m_s',.8),('start_delay_s',2.5)]}
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values.values()):
        raise ValueError('Native hazard geometry must be finite')
    path=layout['path'];segments=[math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:])]
    distance=values['distance_m'];length=sum(segments)
    if distance<8. or length-distance<6.:raise ValueError('Hazard requires start/goal clearance on a longer route')
    if kind=='static_vehicle':
        if abs(values['lateral_m'])>1.5 or 'speed_m_s' in spec or 'start_delay_s' in spec:
            raise ValueError('Invalid static hazard placement/options')
    elif not 2.5<=abs(values['lateral_m'])<=5. or not .2<=values['speed_m_s']<=1.5 or not 0<=values['start_delay_s']<=15.:
        raise ValueError('Invalid crossing geometry/speed/delay')
    remaining=distance
    for index,segment in enumerate(segments):
        if segment<=0:continue
        if remaining<=segment:
            a,b=path[index],path[index+1];fraction=remaining/segment
            yaw=a[2]+fraction*((b[2]-a[2]+math.pi)%(2*math.pi)-math.pi)
            x=a[0]+fraction*(b[0]-a[0])-values['lateral_m']*math.sin(yaw)
            y=a[1]+fraction*(b[1]-a[1])+values['lateral_m']*math.cos(yaw)
            break
        remaining-=segment
    else:raise ValueError('Hazard distance exceeds route')
    if not .5<x<layout['width']-.5 or not .5<y<layout['height']-.5:raise ValueError('Hazard spawn outside registered extent')
    direction=-1. if values['lateral_m']>0 else 1.
    return {'version':'native_hazard_v1','kind':kind,'distance_m':distance,'lateral_m':values['lateral_m'],
        'spawn_pose':[x,y,yaw],'coordinate_scope':'route point is actor origin, not ego rear axle',
        'task_family':'static_obstacle' if kind=='static_vehicle' else 'dynamic_interaction',
        'walker_direction_local':[-direction*math.sin(yaw),direction*math.cos(yaw)],
        'walker_speed_m_s':values['speed_m_s'] if kind=='crossing_walker' else 0.,
        'start_delay_s':values['start_delay_s'] if kind=='crossing_walker' else 0.}


def verify_native_hazard_scene(layout,spec,observed,static,dynamic):
    expected=resolve_native_hazard(layout,spec)
    for key,value in expected.items():
        actual=observed.get(key)
        if isinstance(value,list):
            if not isinstance(actual,list) or len(actual)!=len(value) or any(abs(a-b)>1e-7 for a,b in zip(actual,value)):
                raise ValueError('Native actor spawn geometry differs from registration')
        elif actual!=value:raise ValueError('Native hazard metadata differs from registration')
    if type(observed.get('actor_id')) is not int or observed['actor_id']<=0:raise ValueError('Native hazard lacks actual actor identity')
    if expected['kind']=='static_vehicle':
        if len(static)!=1 or dynamic or not observed.get('actor_type','').startswith('vehicle.'):
            raise ValueError('Native static actor inventory differs')
        box=static[0];point=[(box[0]+box[2])/2,(box[1]+box[3])/2]
    else:
        if static or len(dynamic)!=1 or not observed.get('actor_type','').startswith('walker.'):
            raise ValueError('Native walker inventory differs')
        point=[dynamic[0]['x'],dynamic[0]['y']]
    if math.hypot(point[0]-expected['spawn_pose'][0],point[1]-expected['spawn_pose'][1])>.75:
        raise ValueError('Measured actor is not at its registered initial location')


def verify_native_hazard_begin(begin,spec):
    scene=begin.get('scene',{})
    if (spec is None)!=(scene.get('native_hazard') is None):
        raise ValueError('Measured native hazard presence differs from registered task')
    if spec is None:return
    observed=scene['native_hazard']
    verify_native_hazard_scene(scene['native_layout'],spec,observed,scene['static'],begin['dynamic'])
    measured=begin.get('native_dynamic_actors',[])
    expected=[] if spec['kind']=='static_vehicle' else [{'actor_id':observed['actor_id'],
        'actor_type':observed['actor_type'],'state':begin['dynamic'][0]}]
    if measured!=expected:raise ValueError('Initial native actor identity/state differs from measured dynamic observation')

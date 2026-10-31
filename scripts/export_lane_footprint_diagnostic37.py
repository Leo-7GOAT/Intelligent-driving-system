"""Read a local CARLA map only; no Client, server, actors, controls or ticks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import carla
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.lane_projection import DrivingSurfaceIndex


def run(diagnosis,map_file,output,spacing=.1,wall_seconds=60):
    diagnosis,map_file,output=map(Path,(diagnosis,map_file,output))
    if output.exists() or not .025<=spacing<=.2 or not 30<=wall_seconds<=180:
        raise ValueError('Require fresh output and bounded diagnostic settings')
    started=time.monotonic();deadline=started+wall_seconds
    data=json.loads(diagnosis.read_text(encoding='utf-8-sig'))
    road=carla.Map(map_file.stem,map_file.read_text(encoding='utf-8'))
    map_hash=hashlib.sha256(road.to_opendrive().encode()).hexdigest()
    if data['OpenDRIVE_sha256']!=map_hash:
        raise ValueError('Local OpenDRIVE differs from the actual recorded map')
    index=DrivingSurfaceIndex(road,carla.LaneType.Driving)
    coarse=[w for bucket in index.buckets.values() for _,w in bucket]
    cases=[];fine_queries=0;membership_queries=0
    def guard():
        if time.monotonic()>deadline or fine_queries>30000:raise TimeoutError('Offline map diagnostic budget exceeded')
    for case in data['cases']:
        guard();body=case['world_body_corners'];plane=case['plane_z']
        padding=index.max_half_width+2.
        xmin=min(p[0] for p in body)-padding;xmax=max(p[0] for p in body)+padding
        ymin=min(p[1] for p in body)-padding;ymax=max(p[1] for p in body)+padding
        groups={}
        for waypoint in coarse:
            point=waypoint.transform.location
            if xmin<=point.x<=xmax and ymin<=point.y<=ymax and abs(point.z-plane)<1.5:
                key=(waypoint.road_id,waypoint.section_id,waypoint.lane_id)
                groups.setdefault(key,[]).append(float(waypoint.s))
        ribbons=[]
        for key,positions in sorted(groups.items()):
            low=max(0.,min(positions)-1.1);high=max(positions)+1.1
            samples=[]
            for step in range(int(math.ceil((high-low)/spacing))+1):
                if step%128==0:guard()
                coordinate=min(high,low+step*spacing)
                waypoint=road.get_waypoint_xodr(key[0],key[2],coordinate);fine_queries+=1
                if (waypoint is None or waypoint.section_id!=key[1] or waypoint.lane_type!=carla.LaneType.Driving
                        or abs(waypoint.transform.location.z-plane)>=1.5):
                    samples.append(None);continue
                point=waypoint.transform.location;angle=math.radians(waypoint.transform.rotation.yaw);half=waypoint.lane_width/2
                normal=(-math.sin(angle)*half,math.cos(angle)*half)
                samples.append({'s':float(waypoint.s),'center':[point.x,point.y,point.z],
                    'left':[point.x+normal[0],point.y+normal[1]],'right':[point.x-normal[0],point.y-normal[1]],'width':waypoint.lane_width})
            for first,second in zip(samples,samples[1:]):
                if first is not None and second is not None and 0<second['s']-first['s']<=spacing*1.01:
                    ribbons.append({'road_id':key[0],'section_id':key[1],'lane_id':key[2],
                        's':[first['s'],second['s']],'polygon':[first['left'],first['right'],second['right'],second['left']]})
        # Dense independent point membership is additional evidence, not a
        # substitute for the full polygon-union check performed separately.
        length=math.hypot(body[1][0]-body[0][0],body[1][1]-body[0][1])
        width=math.hypot(body[3][0]-body[0][0],body[3][1]-body[0][1])
        nx=int(math.ceil(length/.1));ny=int(math.ceil(width/.1));misses=[];count=0
        for i in range(nx+1):
            guard()
            for j in range(ny+1):
                u,v=float(i)/nx,float(j)/ny
                x=body[0][0]+u*(body[1][0]-body[0][0])+v*(body[3][0]-body[0][0])
                y=body[0][1]+u*(body[1][1]-body[0][1])+v*(body[3][1]-body[0][1])
                count+=1;membership_queries+=1
                if not index.contains(carla.Location(x=x,y=y,z=plane)):misses.append([x,y])
        cases.append({'case_id':case['case_id'],'world_body_corners':body,'ribbons':ribbons,
            'dense_samples':count,'dense_misses':misses,'raster_intersection':case['raster_intersection'],
            'task_box_violation':case['task_box_violation']})
    result={'scope':'Offline local-map lane ribbons and dense footprint membership;not rendered-mesh proof or a changed evaluation. No server connection,spawn,control or tick.',
        'diagnosis_sha256':hashlib.sha256(diagnosis.read_bytes()).hexdigest(),'OpenDRIVE_sha256':map_hash,
        'ribbon_spacing_m':spacing,'fine_waypoint_queries':fine_queries,'dense_membership_queries':membership_queries,
        'width_dilation_m':0.,'cases':cases,'wall_seconds':time.monotonic()-started,
        'new_physical_controls':0,'optimizer_updates':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps([{'case':c['case_id'],'ribbons':len(c['ribbons']),'dense_samples':c['dense_samples'],'dense_misses':len(c['dense_misses'])} for c in cases]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('diagnosis','map-file','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--spacing',type=float,default=.1);p.add_argument('--wall-seconds',type=int,default=60)
    a=p.parse_args();run(a.diagnosis,a.map_file,a.output,a.spacing,a.wall_seconds)

"""Read-only shared-grid and outside-boundary controls for refined lane unions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import carla
from navstack.lane_projection import local_driving_union,DrivingSurfaceIndex
from navstack.native_route import sample_road_grid


def audit(source,output,indexed=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.time();client=carla.Client("127.0.0.1",2000);client.set_timeout(15)
    if client.get_server_version()!="0.9.15" or client.get_client_version()!="0.9.15":raise ValueError("version")
    road=client.get_world().get_map();rows=[];controls=[]
    index=DrivingSurfaceIndex(road,carla.LaneType.Driving) if indexed else None
    for file in sorted((Path(source)/"B1").glob("scene_*.json")):
        data=json.loads(file.read_text(encoding="utf-8"));scene=data['scene'];layout=scene['native_layout']
        angle=math.radians(layout['angle']);co,si=math.cos(angle),math.sin(angle);old=scene['road_grid']
        cache={}
        def query(x,y):
            if time.time()-start>180:raise RuntimeError("read-only180s budget")
            loc=carla.Location(x=layout['origin'][0]+x*co+y*si,y=layout['origin'][1]+x*si-y*co,z=layout['spawn_z']-.6)
            if index is not None:return index.contains(loc)
            raw=road.get_waypoint(loc,project_to_road=False,lane_type=carla.LaneType.Driving)
            if raw is not None and abs(raw.transform.location.z-loc.z)<1.5:return True
            return local_driving_union(road,loc,carla.LaneType.Driving)['inside']
        grid=sample_road_grid(layout,query,version='carla_driving_indexed_union_64_v3' if indexed else 'carla_driving_refined_union_64_v2')
        recovered=sum(a==0 and b==1 for ra,rb in zip(old['drivable'],grid['drivable']) for a,b in zip(ra,rb))
        lost=sum(a==1 and b==0 for ra,rb in zip(old['drivable'],grid['drivable']) for a,b in zip(ra,rb))
        # Save as diagnostic alternative, never label a changed grid as v1 data.
        (output/file.name).write_text(json.dumps(dict(scene=scene,alternative_grid=grid,source_sha256=hashlib.sha256(file.read_bytes()).hexdigest())),encoding='utf-8')
        rows.append(dict(scene=file.name,recovered=recovered,lost=lost))
        for position_index in (3,8,13,18,21):
            x,y,yaw=layout['path'][position_index];loc=carla.Location(x=layout['origin'][0]+x*co+y*si,y=layout['origin'][1]+x*si-y*co,z=layout['spawn_z']-.6)
            union=local_driving_union(road,loc,carla.LaneType.Driving)
            first=road.get_waypoint(loc,project_to_road=True,lane_type=carla.LaneType.Driving)
            if first.is_junction:continue
            a=math.radians(first.transform.rotation.yaw);normal=(-math.sin(a),math.cos(a))
            center=first.transform.location
            lanes=[]
            for candidate in union['candidates']:
                w=road.get_waypoint_xodr(candidate['road_id'],candidate['lane_id'],first.s)
                p=w.transform.location;offset=(p.x-center.x)*normal[0]+(p.y-center.y)*normal[1]
                lanes.append((offset-w.lane_width/2,offset+w.lane_width/2))
            lower,upper=min(p[0] for p in lanes),max(p[1] for p in lanes)
            for offset,expected,label in ((lower-.05,False,'left_out'),(lower+.05,True,'left_in'),(upper-.05,True,'right_in'),(upper+.05,False,'right_out')):
                point=carla.Location(x=center.x+normal[0]*offset,y=center.y+normal[1]*offset,z=center.z)
                result=index.inspect(point) if index is not None else local_driving_union(road,point,carla.LaneType.Driving)
                controls.append(dict(scene=file.name,index=position_index,label=label,expected=expected,inside=result['inside'],point=[point.x,point.y,point.z],candidates=result['candidates']))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(scope='Read-only alternative-grid/negative-control audit; no policy or terminal gate changed',rows=rows,controls=controls,
        controls_correct=sum(r['inside']==r['expected'] for r in controls),seconds=time.time()-start,
        map_sha256=hashlib.sha256(road.to_opendrive().encode()).hexdigest())
    (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--indexed',action='store_true')
    a=p.parse_args();audit(a.source,a.output,a.indexed)

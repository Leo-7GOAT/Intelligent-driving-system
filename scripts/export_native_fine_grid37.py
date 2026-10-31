"""Offline local-map v4 raster diagnostic; never connects to a CARLA server."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import carla
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.lane_projection import DrivingSurfaceIndex
from navstack.native_route import sample_road_grid, REFINED_GRID_VERSION


def run(diagnosis,map_file,output):
    data=json.loads(Path(diagnosis).read_text(encoding='utf-8-sig'))
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve prior geometry diagnostic')
    road=carla.Map(Path(map_file).stem,Path(map_file).read_text(encoding='utf-8'))
    if hashlib.sha256(road.to_opendrive().encode()).hexdigest()!=data['OpenDRIVE_sha256']:
        raise ValueError('Local map differs from recorded OpenDRIVE')
    started=time.monotonic();index=DrivingSurfaceIndex(road,carla.LaneType.Driving)
    cache={};cases=[]
    for row in data['cases']:
        layout=row['native_layout'];key=json.dumps(layout,sort_keys=True)
        if key not in cache:
            begin=time.monotonic();angle=math.radians(layout['angle']);co,si=math.cos(angle),math.sin(angle)
            def query(x,y):
                if time.monotonic()-started>240:raise TimeoutError('Offline fine-grid budget exceeded')
                return index.contains(carla.Location(x=layout['origin'][0]+x*co+y*si,
                    y=layout['origin'][1]+x*si-y*co,z=layout['spawn_z']-.6))
            grid=sample_road_grid(layout,query,REFINED_GRID_VERSION)
            cache[key]=(grid,time.monotonic()-begin)
        grid,seconds=cache[key]
        cases.append({'case_id':row['case_id'],'grid':grid,'construction_seconds':seconds,
                      'local_body_corners':row['local_body_corners'],
                      'old_raster_intersection':row['raster_intersection'],
                      'task_box_violation':row['task_box_violation']})
        print(json.dumps({'case':row['case_id'],'construction_seconds':seconds}),flush=True)
    result={'scope':'Offline alternative geometry;original outcomes unchanged;zero controls and updates',
        'OpenDRIVE_sha256':data['OpenDRIVE_sha256'],
        'diagnosis_sha256':hashlib.sha256(Path(diagnosis).read_bytes()).hexdigest(),
        'cases':cases,'new_physical_controls':0,'optimizer_updates':0,'wall_seconds':time.monotonic()-started}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('diagnosis','map-file','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.diagnosis,a.map_file,a.output)

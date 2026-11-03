"""Offline installed-SDK map predicate diagnostic. No Client, world or actor operations."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import carla


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(input_path,xodr_path,lane_module):
    data=json.loads(Path(input_path).read_text(encoding='utf-8'))
    content=Path(xodr_path).read_bytes().decode('utf-8-sig')
    if hashlib.sha256(content.encode('utf-8')).hexdigest()!=data['opendrive_sha256']:
        raise ValueError('Installed map differs from recorded native OpenDRIVE')
    if digest(lane_module)!=data['lane_projection_sha256']:raise ValueError('Use the executed lane predicate source')
    spec=importlib.util.spec_from_file_location('recorded_lane_projection',str(lane_module))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    road_map=carla.Map(data['map_name'],content)
    index=module.DrivingSurfaceIndex(road_map,carla.LaneType.Driving)
    def inspect(xyz,details=True):
        point=carla.Location(x=xyz[0],y=xyz[1],z=xyz[2])
        raw=road_map.get_waypoint(point,project_to_road=False,lane_type=carla.LaneType.Driving)
        refined=index.inspect(point)
        result={'world':[point.x,point.y,point.z],'raw_inside':raw is not None,
                'indexed_inside':index.contains(point),'refined_inside':refined['inside']}
        if details:result['candidates']=refined['candidates']
        return result
    angle=math.radians(data['layout']['angle']);co,si=math.cos(angle),math.sin(angle)
    origin=data['layout']['origin'];z=data['layout']['spawn_z']-.6
    def world(x,y):return [origin[0]+x*co+y*si,origin[1]+x*si-y*co,z]
    x0,y0,x1,y1=data['raster_cell_bounds']
    five=[world(x0,y0),world(x1,y0),world(x0,y1),world(x1,y1),world((x0+x1)/2,(y0+y1)/2)]
    cell_samples=[inspect(point) for point in five]
    measured=inspect(data['recorded_offroad_point_world'])
    point_at_grid_height=list(data['recorded_offroad_point_world']);point_at_grid_height[2]=z
    at_grid_height=inspect(point_at_grid_height)
    dense=[];deadline=time.monotonic()+120
    for iy in range(17):
        for ix in range(17):
            if time.monotonic()>deadline:raise TimeoutError('Finite offline map sampling budget exhausted')
            dense.append(inspect(world(x0+(x1-x0)*ix/16,y0+(y1-y0)*iy/16),False))
    return {'scope':__doc__,'script_sha256':digest(__file__),'input_sha256':digest(input_path),
        'opendrive_sha256':data['opendrive_sha256'],'lane_projection_sha256':digest(lane_module),
        'runtime_python':sys.version.split()[0],'recorded_input':data,
        'measured_point_query':measured,'same_xy_at_raster_height_query':at_grid_height,
        'raster_five_sample_queries':cell_samples,'dense_cell_sample_count':len(dense),
        'dense_non_driving_count':sum(not r['indexed_inside'] for r in dense),'dense_cell_queries':dense,
        'new_controls':0,'new_world_ticks':0,'new_updates':0,
        'limits':'Offline OpenDRIVE lane-membership approximation only;not rendered-mesh or new CARLA-driving evidence. No thresholds or failure outcomes are modified.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True)
    p.add_argument('--xodr',required=True);p.add_argument('--lane-module',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Preserve original offline map evidence')
    result=run(a.input,a.xodr,a.lane_module)
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'measured_indexed_inside':result['measured_point_query']['indexed_inside'],
        'grid_height_inside':result['same_xy_at_raster_height_query']['indexed_inside'],
        'raster_sample_inside':[r['indexed_inside'] for r in result['raster_five_sample_queries']],
        'dense_non_driving_count':result['dense_non_driving_count']}))

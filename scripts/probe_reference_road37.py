"""Nominal reference footprint vs frozen native lane predicate; no Client or ticks."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.reference_road_refinement import footprint_points


def run(input_path,xodr,module_path,output):
    import carla
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve nominal query attempts')
    data=json.loads(Path(input_path).read_text(encoding='utf-8'))
    content=Path(xodr).read_bytes().decode('utf-8-sig')
    if hashlib.sha256(content.encode('utf-8')).hexdigest()!=data['opendrive_sha256'] or digest(module_path)!=data['lane_module_sha256']:
        raise ValueError('Require recorded map and lane implementation')
    spec=importlib.util.spec_from_file_location('frozen_lane_predicate',str(module_path));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    road_map=carla.Map(data['map_name'],content);index=module.DrivingSurfaceIndex(road_map,carla.LaneType.Driving)
    calibration=data['known_non_driving_point_world']
    if index.contains(carla.Location(x=calibration[0],y=calibration[1],z=calibration[2])):
        raise ValueError('Known boundary calibration no longer reproduces')
    results=[];deadline=time.monotonic()+300
    for item in data['references']:
        checked=misses=0;examples=[]
        for pose_index,pose in enumerate(item['poses']):
            if time.monotonic()>deadline:raise TimeoutError('Finite300second nominal-query budget exhausted;partial preserved')
            for xyz in footprint_points(pose,data['vehicle'],item['layout']):
                checked+=1
                point=carla.Location(x=xyz[0],y=xyz[1],z=xyz[2])
                if not index.contains(point):
                    misses+=1
                    if len(examples)<20:examples.append(dict(pose_index=pose_index,
                        world=[point.x,point.y,point.z],requested_world=xyz))
        results.append(dict(key=item['key'],points_checked=checked,non_driving_points=misses,examples=examples))
        output.with_suffix('.partial.json').write_text(json.dumps(dict(results=results)),encoding='utf-8')
        print(dict(key=item['key'],non_driving_points=misses),flush=True)
    report=dict(scope=__doc__,input_sha256=digest(input_path),script_sha256=digest(__file__),results=results,
                new_controls=0,new_world_ticks=0,new_updates=0,
                limits='Nominal sampled model poses at grid height,not actual roll/pitch/tracking or continuous safety. Does not alter road predicates,scenarios,models or recorded outcomes.')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('input','xodr','lane-module','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.input,a.xodr,a.lane_module,a.output)

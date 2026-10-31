"""Separate recorded native samples, raster footprint and task-box violations."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import EnvConfig
from navstack.road_grid import RoadGrid
from navstack.world import body_corners


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(batch,method,output):
    batch,output=Path(batch),Path(output);folder=batch/method
    if output.exists():raise FileExistsError('Preserve prior bounds diagnosis')
    raw=read(folder/'endpoint_raw_audit.json');trace=folder/'evaluation_trace.jsonl.gz'
    if sha(trace)!=raw['evaluation_trace_sha256']:raise ValueError('Evaluation trace differs from raw frame audit')
    if sha(batch/'protocol.json')!=raw['protocol_sha256'] or sha(folder/'config.json')!=raw['config_sha256']:
        raise ValueError('Registered map/configuration changed')
    map_hashes={r['opendrive_sha256'] for r in read(batch/'protocol.json')['routes']}
    if len(map_hashes)!=1:raise ValueError('Require one actual recorded map')
    scenes={read(p)['scene']['condition']:read(p)['scene'] for p in folder.glob('scene_*.json')}
    config=EnvConfig.from_dict(read(folder/'config.json')['env']);cases=[]
    with gzip.open(trace,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['info']['termination_reason']!='out_of_bounds':continue
            tick=row['info']['trace'][-1];state=tick['state'];scene=scenes[row['case_id']]
            body=body_corners((state['x'],state['y'],state['yaw']),config.vehicle)
            layout=scene['native_layout'];angle=np.deg2rad(layout['angle']);co,si=np.cos(angle),np.sin(angle)
            origin=layout['origin'];world_body=np.c_[origin[0]+co*body[:,0]+si*body[:,1],origin[1]+si*body[:,0]-co*body[:,1]]
            cases.append({'case_id':row['case_id'],'frame':tick['backend_frame'],'episode_id':tick['backend_evidence']['episode_id'],
                'state':state,'local_body_corners':body.tolist(),'world_body_corners':world_body.tolist(),
                'plane_z':layout['spawn_z']-.6,'native_layout':layout,
                'raster_intersection':RoadGrid(scene['road_grid']).intersects(body),
                'task_box_violation':bool(np.any(body<0) or np.any(body[:,0]>scene['width']) or np.any(body[:,1]>scene['height'])),
                'native_road_probe':tick['backend_evidence']['road_surface_probe'],
                'collisions':tick['backend_evidence']['collisions'],'scene_road_sha256':scene['road_grid']['sha256']})
    result={'scope':'Diagnostic only;recorded scores and safety labels unchanged. A finite native sample does not prove full continuous-body coverage.',
        'batch':batch.name,'method':method,'evaluation_trace_sha256':sha(trace),'OpenDRIVE_sha256':next(iter(map_hashes)),
        'vehicle':asdict(config.vehicle),'cases':cases,
        'new_physical_controls':0,'optimizer_updates':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps([{'case':r['case_id'],'raster_hit':bool(r['raster_intersection']),'task_box':r['task_box_violation'],
        'native_sample_misses':len(r['native_road_probe']['offroad_points_world'])} for r in cases]))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('batch','method','output'):p.add_argument('--'+key,required=True)
    args=p.parse_args();audit(args.batch,args.method,args.output)

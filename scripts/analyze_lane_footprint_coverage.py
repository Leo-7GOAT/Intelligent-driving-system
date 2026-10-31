"""Check complete recorded body polygons against undilated sampled lane ribbons."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def analyze(source,output,dependency_dir=None):
    if dependency_dir is not None:sys.path.insert(0,str(Path(dependency_dir).resolve()))
    import shapely
    from shapely.geometry import Polygon
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError('Preserve prior continuous-body diagnosis')
    data=json.loads(source.read_text(encoding='utf-8'));results=[]
    if data['width_dilation_m']!=0.:raise ValueError('Lane dilation is not permitted in this diagnostic')
    for row in data['cases']:
        polygons=[Polygon(r['polygon']) for r in row['ribbons']]
        if any(not p.is_valid or p.area<=0 for p in polygons):raise ValueError('Invalid native lane ribbon')
        surface=shapely.union_all(polygons);body=Polygon(row['world_body_corners'])
        if not body.is_valid or body.area<=0:raise ValueError('Invalid measured body polygon')
        covered=bool(surface.covers(body))
        results.append({'case_id':row['case_id'],'raster_intersection':row['raster_intersection'],
            'task_box_violation':row['task_box_violation'],'dense_samples':row['dense_samples'],
            'dense_misses':len(row['dense_misses']),'complete_body_covered_by_lane_union':covered,
            'body_area_m2':body.area,'body_area_outside_lane_union_m2':body.difference(surface).area,
            'body_to_union_boundary_clearance_m':body.distance(surface.boundary) if covered else 0.,
            'lane_ribbons':len(polygons)})
    result={'scope':'Full polygon coverage under0.1m sampled CARLA lane-width ribbons,not rendered asphalt mesh/swept-volume certification. Original scores and gates unchanged.',
        'surface_export_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'OpenDRIVE_sha256':data['OpenDRIVE_sha256'],'ribbon_spacing_m':data['ribbon_spacing_m'],
        'shapely_version':shapely.__version__,'GEOS_version':shapely.geos_version_string,
        'width_dilation_m':0.,'cases':results,'new_physical_controls':0,'optimizer_updates':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(results));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--dependency-dir')
    a=p.parse_args();analyze(a.source,a.output,a.dependency_dir)

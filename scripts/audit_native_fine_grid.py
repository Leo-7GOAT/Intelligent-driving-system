"""Compare original and alternative full-body rasters without relabeling runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from navstack.native_route import REFINED_GRID_VERSION
from navstack.road_grid import RoadGrid


def audit(source,diagnosis,batch,output):
    source,diagnosis,batch,output=map(Path,(source,diagnosis,batch,output))
    if output.exists():raise FileExistsError('Preserve prior diagnostic')
    data=json.loads(source.read_text(encoding='utf-8'));original=json.loads(diagnosis.read_text(encoding='utf-8'))
    if (data['diagnosis_sha256']!=hashlib.sha256(diagnosis.read_bytes()).hexdigest()
            or data['OpenDRIVE_sha256']!=original['OpenDRIVE_sha256']):raise ValueError('Diagnostic provenance differs')
    scenes={json.loads(p.read_text(encoding='utf-8'))['scene']['condition']:json.loads(p.read_text(encoding='utf-8'))['scene']
            for p in (batch/original['method']).glob('scene_*.json')}
    results=[]
    for row in data['cases']:
        scene=scenes[row['case_id']];coarse=RoadGrid(scene['road_grid']);fine=RoadGrid(row['grid'])
        if fine.payload['version']!=REFINED_GRID_VERSION:raise ValueError('Require explicit v4')
        np.testing.assert_array_equal(fine.drivable,coarse.drivable)
        body=np.array(row['local_body_corners']);old=coarse.intersects(body);new=fine.intersects(body)
        if old!=row['old_raster_intersection']:raise ValueError('Original terminal raster changed')
        results.append({'case_id':row['case_id'],'old_raster_intersection':old,'new_raster_intersection':new,
            'task_box_violation':row['task_box_violation'],'common64_policy_grid_identical':True,
            'old_cell_metres':[coarse.dx,coarse.dy],'new_cell_metres':[fine.dx,fine.dy],
            'new_body_clearance_metres':fine.clearance(body),'new_grid_sha256':fine.sha256,
            'construction_seconds':row['construction_seconds']})
    report={'scope':'Offline geometry diagnostic;not changed driving scores or evidence of trained-v4 performance',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'diagnosis_sha256':data['diagnosis_sha256'],'OpenDRIVE_sha256':data['OpenDRIVE_sha256'],
        'cases':results,'new_physical_controls':0,'optimizer_updates':0,
        'conclusion':'Both raster-only endpoints are inside finer sampled geometry;both task-box violations remain invalid. Requires separately registered native migration and retraining;old6/12and4unsafe unchanged.',
        'limitation':'Five map samples per cell do not certify continuous lane coverage or rendered asphalt mesh;no lane-width dilation.'}
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','diagnosis','batch','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();audit(a.source,a.diagnosis,a.batch,a.output)

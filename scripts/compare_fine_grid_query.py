"""One fixed recorded query under two geometry versions; no model or physics."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import EnvConfig
from navstack.planning import HybridAStar
from navstack.road_grid import RoadGrid
from navstack.world import World


def run(fixture,fine_export,output):
    fixture,fine_export,output=map(Path,(fixture,fine_export,output))
    if output.exists():raise FileExistsError('Preserve prior query comparison')
    case=json.loads(fixture.read_text(encoding='utf-8'))
    export=json.loads(fine_export.read_text(encoding='utf-8'))
    config=EnvConfig.from_dict(case['config'])
    if config.planner.max_seconds is not None:raise ValueError('Require fixed node-budget search')
    original=case['world']['road_grid']
    alternatives=[r['grid'] for r in export['cases'] if r['grid']['geometry_id']==original['geometry_id']]
    if not alternatives or any(a!=alternatives[0] for a in alternatives):raise ValueError('Missing/ambiguous matching geometry')
    alternative=alternatives[0]
    if (original['width'],original['height'])!=(alternative['width'],alternative['height']):raise ValueError('Task extent differs')
    np.testing.assert_array_equal(original['drivable'],alternative['drivable'])
    rows=[]
    for payload in (original,alternative):
        world=World(case['world']['width'],case['world']['height'],static=case['world']['static'],
            road=RoadGrid(payload),road_collision_backend=config.road_collision_backend)
        start=tuple(case['state'][k] for k in ('x','y','yaw'));begin=time.perf_counter()
        plan=HybridAStar(config.vehicle,deepcopy(config.planner)).plan(start,case['goal'],world)
        rows.append({'version':payload['version'],'grid_sha256':payload['sha256'],
            'status':plan.status,'reason':plan.failure_reason,'expanded_nodes':plan.expanded_nodes,
            'cost':plan.cost if plan.cost is None or np.isfinite(plan.cost) else None,
            'path_points':len(plan.path),'path_sha256':hashlib.sha256(plan.path.tobytes()).hexdigest(),
            'wall_seconds':time.perf_counter()-begin})
    result={'scope':'Offline one saved-query geometry comparison;not new driving,success-rate or real-time proof',
        'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'fine_export_sha256':hashlib.sha256(fine_export.read_bytes()).hexdigest(),
        'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/compare_fine_grid_query.py','navstack/config.py','navstack/native_route.py',
             'navstack/planning.py','navstack/world.py','navstack/road_grid.py','navstack/road_grid_batch.py')},
        'planner':case['config']['planner'],'queries':rows,'new_physical_controls':0,'optimizer_updates':0}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('fixture','fine-export','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.fixture,a.fine_export,a.output)

"""Read registered fixed trajectories with exact file/geometry/road binding."""
import hashlib,json
from pathlib import Path
import numpy as np
from .trajectory import Trajectory

def load_fixed_reference(directory,case_id,geometry_id,road_sha256=None,goal=None):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    rows=manifest['references']
    if len(rows)!=4 or {r['case_id'] for r in rows}!=set(range(4)):raise ValueError('Require exactly four registered fixed references')
    record=next(r for r in rows if r['case_id']==case_id)
    if record['file']!=f'case_{case_id:02d}.json' or record['geometry_id']!=geometry_id:raise ValueError('Fixed reference case/file/geometry differs')
    path=directory/record['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Fixed reference file changed')
    payload=json.loads(path.read_text(encoding='utf-8'))
    if payload['geometry_id']!=geometry_id or payload['road_sha256']!=record['road_sha256'] or road_sha256 is not None and payload['road_sha256']!=road_sha256:
        raise ValueError('Fixed reference road/geometry differs from measured scene')
    if goal is not None and payload['goal']!=list(goal):raise ValueError('Fixed reference original global goal differs')
    values=payload['reference'];arrays={k:np.asarray(values[k],dtype=float) for k in ('x','y','yaw','s','kappa','v','a','t')}
    if (len(arrays['x'])<3 or any(a.ndim!=1 or a.shape!=arrays['x'].shape or not np.isfinite(a).all() for a in arrays.values())
            or not np.all(np.diff(arrays['s'])>0) or not np.all(np.diff(arrays['t'])>0)
            or values['start_time']!=0 or abs(arrays['v'][0])>1e-4 or abs(arrays['v'][-1])>1e-12):
        raise ValueError('Require finite continuous fixed rest-to-stop reference')
    for value in arrays.values():value.setflags(write=False)
    return Trajectory(**arrays,start_time=0.),payload,record

"""Retime an already recorded geometric counterfactual; no simulation client."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.trajectory import parameterize
from navstack.steering_retiming import retime,VERSION
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha


def probe(proof,batch,output):
    proof,batch,output=Path(proof),Path(batch),Path(output)
    if output.exists():raise FileExistsError('Preserve retiming attempts')
    p=read(proof);cfg=EnvConfig.from_dict(read(batch/'H/config.json')['env'])
    if p['trace_sha256']!=sha(batch/'H/evaluation_trace.jsonl.gz'):raise ValueError('Measured input trace changed')
    results=[]
    for item in p['results']:
        if item['variant']!='counterfactual_slew_speed_ceiling' or item['status']!='success':continue
        target=1.5  # Registered speed of this specific recorded action, checked below.
        if p['case_id']!=1 or p['action']!=26 or p['terminal_goal_prefix'] is not True:
            raise ValueError('This bounded probe supports only registered case1 action26')
        path=np.array(item['path']);state=p['measured_state']
        nominal=parameterize(path,target,state['v'],cfg.vehicle,state['time'],True)
        slow=parameterize(path,item['speed_ceiling'],state['v'],cfg.vehicle,state['time'],True)
        reference=retime(nominal,cfg.vehicle,state['steering'])
        k=reference.kappa;near=np.where(k[:-1]*k[1:]<=0,0,np.minimum(abs(k[:-1]),abs(k[1:])))
        rates=cfg.vehicle.wheelbase*abs(np.diff(k))/np.diff(reference.t)/(1+(cfg.vehicle.wheelbase*near)**2)
        results.append(dict(search_speed_ceiling=item['speed_ceiling'],requested_speed=target,
            uniform_slow_seconds=float(slow.t[-1]),unconstrained_seconds=float(nominal.t[-1]),retimed_seconds=float(reference.t[-1]),
            sampled_max_steering_rate=float(rates.max()),max_acceleration=float(reference.a.max()),
            max_deceler=float(-reference.a.min()),maximum_speed=float(reference.v.max()),
            initial_speed=float(reference.v[0]),final_speed=float(reference.v[-1]),
            geometry_unchanged=all(np.array_equal(getattr(reference,n),getattr(nominal,n)) for n in ['x','y','yaw','s']),
            kappa=reference.kappa.tolist(),speed=reference.v.tolist(),times=reference.t.tolist(),
            poses=np.c_[reference.x,reference.y,reference.yaw].tolist()))
    if not results:raise ValueError('No successful lower-speed spatial witness')
    atomic_json(output,dict(scope=__doc__,profile=VERSION,proof_sha256=sha(proof),
        script_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'navstack/steering_retiming.py'),
        case_id=p['case_id'],results=results,new_native_controls=0,
        limits='Sampled model feasibility only. Geometry/native-road and actual closed-loop tracking still require verification.'))
    print([{k:v for k,v in r.items() if k not in ['poses','speed','times','kappa']} for r in results])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('proof','batch','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();probe(a.proof,a.batch,a.output)

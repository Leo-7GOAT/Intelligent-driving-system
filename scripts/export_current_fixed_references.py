"""Generate four current production references from saved measured TRAIN starts.

Fresh interpreter required. No policy prediction,native query or plant step.
The action26global-goal query is an explicit controller-study input,not a claim
that PPO selected it. Both later controllers receive exactly these same arrays.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import sys


def generate(parent,output):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve all reference-generation attempts')
    if any(k=='navstack' or k.startswith('navstack.') for k in sys.modules):
        raise RuntimeError('Fresh interpreter required to select immutable physical source')
    p=json.loads((parent/'protocol.json').read_text());source=parent/'source';sys.path.insert(0,str(source))
    import numpy as np
    from navstack.config import EnvConfig
    from navstack.steering_transition import prepare_reference,settings
    from navstack.reference_option_commitment import configure,settings as option_settings
    from navstack.steering_retiming import retime
    from scripts.audit_native_policy_sequence import observation_env
    import navstack.planning
    if not Path(navstack.planning.__file__).resolve().is_relative_to(source):raise ValueError('Planner source differs')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for name,digest in p['source_hashes'].items():
        if sha(source/name)!=digest:raise ValueError('Frozen reference source changed:'+name)
    routes=p['routes'][:4];geometries={route['geometry_id']:i for i,route in enumerate(routes)};selected={}
    for path in sorted((parent/'H').glob('scene_*.json')):
        evidence=json.loads(path.read_text());scene,initial=evidence['scene'],evidence['initial'];geometry=scene['native_layout']['geometry_id']
        if geometry in geometries and not initial['static'] and not initial['dynamic']:
            selected.setdefault(geometries[geometry],(path,evidence))
    if set(selected)!=set(range(4)):raise ValueError('All four first registered basic geometries require saved measured starts')
    config=EnvConfig.from_dict(json.loads((parent/'H/config.json').read_text())['env'])
    p.update(evaluation_only=True,controlled_budget_per_method=0,method_budgets={'H':0})
    output.mkdir();rows=[];attempts=[]
    def default(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    for case in range(4):
        path,evidence=selected[case];env=observation_env(config,evidence['scene'],evidence['initial'])
        try:
            if abs(env.state.v)>1e-4 or abs(env.state.time)>1e-6:raise ValueError('Measured rest/reset clock required')
            before=asdict(env.state);configure(env,option_settings(p));env.episode=dict(plan_times=[],plan_failures=0)
            target,speed=env.decode_action(26)
            if np.linalg.norm(np.asarray(target)-np.asarray(env.goal))>1e-6:raise ValueError('Action26doesnot select the registered global goal')
            mode,detail=prepare_reference(env,26,env._prepare_reference,settings(p))
            result=dict(case_id=case,mode=mode,plan=detail,source_scene_sha256=sha(path),state_unchanged=asdict(env.state)==before)
            if not result['state_unchanged']:raise ValueError('Reference query changed measured state')
            if mode!='reference' or env.reference is None:
                result['status']='REFERENCE_UNAVAILABLE';attempts.append(result);continue
            reference=retime(replace(env.reference,v=np.minimum(env.reference.v,1.)),config.vehicle,env.state.steering)
            poses=np.c_[reference.x,reference.y,reference.yaw]
            if not env.world.free_many(poses,config.vehicle).all():raise ValueError('Current fixed reference is not body-free')
            if (reference.v[-1]!=0 or np.linalg.norm(poses[-1,:2]-np.asarray(env.goal[:2]))>.8
                    or abs(np.arctan2(np.sin(poses[-1,2]-env.goal[2]),np.cos(poses[-1,2]-env.goal[2])))>.35):
                raise ValueError('Current fixed reference doesnot end in the task pose region at rest')
            payload=dict(scope='One production global-goal planning query from recorded initial state;not a PPO decision or native execution',
                case_id=case,geometry_id=routes[case]['geometry_id'],road_sha256=evidence['scene']['road_grid']['sha256'],
                goal=list(env.goal),reference=asdict(reference),fixed_query_action=26,policy_called=False,
                speed_cap_m_s=1.,source_scene_sha256=sha(path),source_protocol_sha256=sha(parent/'protocol.json'),
                source_initial_state=before,measured_initial_steering=env.state.steering,plan=detail,
                qualification='Current sampled measured grid and production steering-rate retimer;no new native road query certificate')
            file=output/f'case_{case:02d}.json';file.write_text(json.dumps(payload,default=default,indent=2)+'\n',encoding='utf-8')
            rows.append(dict(case_id=case,file=file.name,sha256=sha(file),geometry_id=payload['geometry_id'],road_sha256=payload['road_sha256']))
            result.update(status='FIXED_REFERENCE_GENERATED',samples=len(reference.x),nominal_seconds=float(reference.t[-1]));attempts.append(result)
        except Exception as error:
            attempts.append(dict(case_id=case,status='INPUT_OR_GENERATION_EXCEPTION',
                error_type=type(error).__name__,error=str(error),source_scene_sha256=sha(path)))
        finally:env.close()
    manifest=dict(scope='Four current fixed rest-to-stop references for controller isolation;zero new controls/updates',
        parent_batch=parent.name,parent_protocol_sha256=sha(parent/'protocol.json'),references=rows,attempts=attempts,
        production_source_hashes={name:p['source_hashes'][name] for name in ['navstack/planning.py','navstack/trajectory.py','navstack/steering_retiming.py','navstack/vehicle.py']},
        export_script_sha256=sha(Path(__file__)),new_physical_controls=0,PPO_updates=0)
    (output/'manifest.json').write_text(json.dumps(manifest,default=default,indent=2)+'\n',encoding='utf-8')
    if len(rows)!=4:raise ValueError('All reference attempts retained;no control study admitted with an incomplete catalog')
    print(json.dumps(dict(status='FOUR_CURRENT_REFERENCES_READY',nominal_seconds=[x['nominal_seconds'] for x in attempts],new_controls=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();generate(args.parent,args.output)

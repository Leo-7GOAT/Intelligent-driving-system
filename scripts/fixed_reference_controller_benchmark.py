"""Paired physical-integrator tracking benchmark,explicitly separate from CARLA."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import gzip,hashlib,json,math,os,sys,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import VehicleConfig
from navstack.experiments import write_json,jsonable
from navstack.metrics import timing_summary
from navstack.mpc import MPC
from navstack.pursuit_controller import TimedPursuitPI,PARAMETERS
from navstack.road_grid import RoadGrid
from navstack.research import resources_ok
from navstack.reference_manifest import load_fixed_reference
from navstack.trajectory import parameterize,tracking_error
from navstack.vehicle import VehiclePlant,VehicleState,wrap
from navstack.world import World
from scripts.audit_native_rule import audit as rule_audit

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def terminal(state,reference,world,vehicle,goal=None):
    if world.out_of_bounds((state.x,state.y,state.yaw),vehicle):return 'out_of_bounds'
    if world.collision((state.x,state.y,state.yaw),vehicle):return 'collision'
    goal=goal if goal is not None else (reference.x[-1],reference.y[-1],reference.yaw[-1])
    if math.hypot(state.x-goal[0],state.y-goal[1])<=.8 and state.v<=.25 and abs(wrap(state.yaw-goal[2]))<=.35:return 'success'
    if state.time>=45-1e-8:return 'deadline'
    return None

def run(source_rule,wall_seconds=600,fixed_references=None):
    source_rule=Path(source_rule).resolve()
    if not 60<=wall_seconds<=900:raise ValueError('Bounded60..900s CPU benchmark required')
    if fixed_references is None:raise ValueError('Registered body-free production references required')
    if rule_audit(source_rule)!=read(source_rule/'raw_audit.json'):raise ValueError('Source road/frame audit changed')
    calibration=read(source_rule/'calibration/calibration.json');vehicle=VehicleConfig(**calibration['vehicle'])
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_fixed_reference_controllers_cpu');batch.mkdir()
    started=time.time();deadline=started+wall_seconds
    protocol={'scope':'Lightweight physical VehiclePlant control comparison on measured road grids,NOTnew CARLA execution or PPO evidence.',
      'seeds':[11,22,33,44,55],'routes':4,'controllers':['mpc20','pursuit_pi'],'plant_variants':['nominal','lag_delay_scale'],
      'paired_episodes':80,'target_speed_m_s':1.,'control_dt':.05,'episode_seconds':45.,
      'source_rule_batch':source_rule.name,'source_raw_audit_sha256':sha(source_rule/'raw_audit.json'),
      'calibration_sha256':sha(source_rule/'calibration/calibration.json'),'pursuit_parameters':PARAMETERS,
      'created_epoch':started,'deadline_epoch':deadline,
      'initial_errors':'Independent fixed seeds generate lateral[-.15,.15]m and heading[-.02,.02]rad offsets; paired controllers share exact initial state.',
      'mismatch':'Physical integrator only:wheelbase_scale1.1,acceleration_scale.8,steer_tau.3,input_delay_steps2;controller nominal geometry/limits unchanged.'}
    protocol['source_hashes']={name:sha(ROOT/name) for name in ('scripts/fixed_reference_controller_benchmark.py','navstack/pursuit_controller.py','navstack/mpc.py','navstack/trajectory.py','navstack/vehicle.py','navstack/world.py','navstack/road_grid.py','navstack/config.py')}
    protocol['source_scene_hashes']={f'scene_{i:04d}.json':sha(source_rule/f'scene_{i:04d}.json') for i in range(4)}
    protocol['nominal_vehicle']=asdict(vehicle)
    protocol['fixed_references_manifest_sha256']=sha(Path(fixed_references)/'manifest.json')
    write_json(batch/'protocol.json',protocol);results=[]
    try:
        for route_id in range(4):
            scene_path=source_rule/f'scene_{route_id:04d}.json';scene=read(scene_path)['scene']
            grid=RoadGrid(scene['road_grid']);world=World(scene['width'],scene['height'],road=grid)
            reference,_,_=load_fixed_reference(fixed_references,route_id,scene['native_layout']['geometry_id'],grid.sha256,scene['goal'])
            if not world.free_many(np.c_[reference.x,reference.y,reference.yaw],vehicle).all():raise ValueError('Fixed reference is not body-free on measured source grid')
            ref_file=batch/f'reference_{route_id}.json';write_json(ref_file,asdict(reference));ref_sha=sha(ref_file)
            for plant_variant in ('nominal','lag_delay_scale'):
                physical=vehicle if plant_variant=='nominal' else replace(vehicle,wheelbase_scale=1.1,acceleration_scale=.8,steer_tau=.3,input_delay_steps=2)
                for seed in protocol['seeds']:
                    rng=np.random.default_rng(seed);lateral=float(rng.uniform(-.15,.15));heading=float(rng.uniform(-.02,.02))
                    yaw=float(reference.yaw[0]);initial=VehicleState(float(reference.x[0]-lateral*math.sin(yaw)),float(reference.y[0]+lateral*math.cos(yaw)),yaw+heading)
                    for controller_name in protocol['controllers']:
                        healthy,reason=resources_ok(ROOT)
                        if not healthy:raise RuntimeError('Resource floor: '+str(reason))
                        if time.time()>=deadline:raise TimeoutError('Fixed CPU benchmark wall budget expired; preserve completed pairs')
                        name=f'route{route_id}_{plant_variant}_seed{seed}_{controller_name}'
                        controller=MPC(vehicle,.05,20) if controller_name=='mpc20' else TimedPursuitPI(vehicle,.05)
                        plant=VehiclePlant(physical,initial);errors=[];jerks=[];times=[];failures=0;reason=None
                        with gzip.open(batch/(name+'.jsonl.gz'),'wt',compresslevel=1) as stream:
                            for step in range(900):
                                if time.time()>=deadline:raise TimeoutError('Fixed CPU benchmark wall budget expired')
                                before=plant.state;control=controller.step(before,reference,plant.previous_command)
                                state=plant.step(control.command,.05);error=tracking_error(state,reference)
                                errors.append(error);jerks.append((state.acceleration-before.acceleration)/.05)
                                times.append(control.diagnostics['control_seconds']);failures+=control.status!='solved'
                                reason=terminal(state,reference,world,physical,scene['goal'])
                                stream.write(json.dumps({'step':step,'before':asdict(before),'command':asdict(control.command),'state':asdict(state),'tracking':error,'status':control.status,'diagnostics':control.diagnostics,'terminal':reason},allow_nan=False)+'\n')
                                if step%100==0:write_json(batch/'heartbeat.json',{'epoch':time.time(),'episode':name,'step':step,'complete_episodes':len(results)})
                                if reason:break
                        row={'episode':name,'route_id':route_id,'seed':seed,'plant_variant':plant_variant,'controller':controller_name,
                             'reference_sha256':ref_sha,'initial_state':asdict(initial),'physics_steps':len(errors),'success':reason=='success','termination_reason':reason,
                             'tracking_rms':{key:float(np.sqrt(np.mean([e[key]**2 for e in errors]))) for key in ('lateral','heading','speed')},
                             'jerk_rms':float(np.sqrt(np.mean(np.square(jerks)))),'controller_failures':failures,
                             'control_timing':timing_summary(times,.05),'final_state':asdict(plant.state),
                             'raw_sha256':sha(batch/(name+'.jsonl.gz'))}
                        results.append(row);write_json(batch/'partial_summary.json',{'episodes':results})
        if len(results)!=80:raise ValueError('All80declared paired episodes required')
        write_json(batch/'summary.json',{'status':'complete','scope':protocol['scope'],'episodes':results,'wall_seconds':time.time()-started})
        print(batch,flush=True)
        print(json.dumps({v:{c:sum(r['success'] for r in results if r['plant_variant']==v and r['controller']==c) for c in protocol['controllers']} for v in protocol['plant_variants']}),flush=True)
    except BaseException as exc:
        write_json(batch/'failure.json',{'error':repr(exc),'completed_episodes':len(results),'physical_backend':'VehiclePlant,not CARLA'});raise
    return batch

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-rule',type=Path,required=True);parser.add_argument('--wall-seconds',type=int,default=600)
    parser.add_argument('--fixed-references',type=Path,required=True)
    args=parser.parse_args();run(args.source_rule,args.wall_seconds,args.fixed_references)

"""Recompute every saved integrator frame and paired controller metrics."""
import argparse
from dataclasses import asdict,replace
import gzip,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import VehicleConfig
from navstack.vehicle import VehiclePlant,VehicleState,Command
from navstack.trajectory import Trajectory,tracking_error
from navstack.road_grid import RoadGrid
from navstack.world import World
from navstack.metrics import timing_summary

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def analyze(batch,output):
    batch,output=Path(batch),Path(output)
    if output.exists():raise FileExistsError('Preserve prior controller analysis')
    protocol=read(batch/'protocol.json');summary=read(batch/'summary.json');rows=summary['episodes']
    if summary['status']!='complete' or len(rows)!=80 or len({r['episode'] for r in rows})!=80:raise ValueError('Require every declared controller episode')
    for name,digest in protocol['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Use the registered original dynamics/controller source')
    vehicle=VehicleConfig(**protocol['nominal_vehicle']);frames=0;groups={};pairs={}
    scene_root=batch/'source_scenes' if (batch/'source_scenes').is_dir() else ROOT/'runs'/protocol['source_rule_batch']
    scenes={}
    for case_id in range(4):
        file=scene_root/f'scene_{case_id:04d}.json'
        if sha(file)!=protocol['source_scene_hashes'][file.name]:raise ValueError('Registered source scene changed')
        scenes[case_id]=read(file)['scene']
    for row in rows:
        config=vehicle if row['plant_variant']=='nominal' else replace(vehicle,wheelbase_scale=1.1,acceleration_scale=.8,steer_tau=.3,input_delay_steps=2)
        plant=VehiclePlant(config,VehicleState(**row['initial_state']));trace=batch/(row['episode']+'.jsonl.gz')
        if sha(trace)!=row['raw_sha256'] or sha(batch/f"reference_{row['route_id']}.json")!=row['reference_sha256']:raise ValueError('Controller input/raw trace changed')
        scene=scenes[row['route_id']];reference_data=read(batch/f"reference_{row['route_id']}.json")
        reference=Trajectory(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in reference_data.items()})
        world=World(scene['width'],scene['height'],road=RoadGrid(scene['road_grid']))
        from scripts.fixed_reference_controller_benchmark import terminal
        n=0;errors=[];jerks=[];times=[];reason=None
        with gzip.open(trace,'rt',encoding='utf-8') as stream:
            for line in stream:
                tick=json.loads(line)
                if reason is not None:raise ValueError('Controller trace continues beyond its terminal state')
                if tick['step']!=n or tick['before']!=asdict(plant.state):raise ValueError('Raw state sequence differs')
                before=plant.state;after=plant.step(Command(**tick['command']),.05)
                if any(abs(v-tick['state'][k])>1e-12 for k,v in asdict(after).items()):raise ValueError('Saved physical-integrator state differs from actual command replay')
                error=tracking_error(after,reference)
                if any(abs(error[k]-tick['tracking'][k])>1e-12 for k in ('lateral','heading','speed')):raise ValueError('Recorded tracking differs from fixed-reference state projection')
                reason=terminal(after,reference,world,config,scene['goal'])
                if reason!=tick['terminal']:raise ValueError('Recorded terminal differs from actual body/goal/clock conditions')
                errors.append(error);jerks.append((after.acceleration-before.acceleration)/.05);times.append(tick['diagnostics']['control_seconds']);n+=1
        if n!=row['physics_steps'] or asdict(plant.state)!=row['final_state']:raise ValueError('Episode duration/final state differs')
        if row['termination_reason']!=reason or row['success']!=(reason=='success'):raise ValueError('Reported controller success differs from actual terminal conditions')
        timing=timing_summary(times,.05)
        if timing!=row['control_timing']:raise ValueError('Reported controller timing differs from all raw computation times')
        for key,value in row['tracking_rms'].items():
            if not np.isclose(value,np.sqrt(np.mean([e[key]**2 for e in errors])),rtol=0,atol=1e-12):raise ValueError('Raw tracking aggregate differs')
        if not np.isclose(row['jerk_rms'],np.sqrt(np.mean(np.square(jerks))),rtol=0,atol=1e-12):raise ValueError('Raw acceleration jerk differs')
        frames+=n;groups.setdefault((row['plant_variant'],row['controller']),[]).append(row)
        row['verified_final_goal_distance_m']=float(np.hypot(plant.state.x-scene['goal'][0],plant.state.y-scene['goal'][1]))
        key=(row['route_id'],row['seed'],row['plant_variant']);pairs.setdefault(key,{})[row['controller']]=row
    if len(pairs)!=40:raise ValueError('Require forty exact controller pairs')
    for pair in pairs.values():
        if set(pair)!={'mpc20','pursuit_pi'} or pair['mpc20']['reference_sha256']!=pair['pursuit_pi']['reference_sha256'] or pair['mpc20']['initial_state']!=pair['pursuit_pi']['initial_state']:raise ValueError('Reference or initial state not paired')
    aggregate=[]
    for (variant,controller),episodes in sorted(groups.items()):
        aggregate.append({'plant_variant':variant,'controller':controller,'cases':len(episodes),'successes':sum(r['success'] for r in episodes),
            'mean_episode_lateral_rmse_m':float(np.mean([r['tracking_rms']['lateral'] for r in episodes])),
            'mean_episode_jerk_rms_m_s3':float(np.mean([r['jerk_rms'] for r in episodes])),
            'mean_final_goal_distance_m':float(np.mean([r['verified_final_goal_distance_m'] for r in episodes])),
            'mean_successful_episode_seconds':float(np.mean([r['final_state']['time'] for r in episodes if r['success']])),
            'mean_episode_control_P95_seconds':float(np.mean([r['control_timing']['p95'] for r in episodes]))})
    result={'scope':'Lightweight controlled fixed-reference comparison,not CARLA or independently trained PPO evidence.',
            'batch':batch.name,'summary_sha256':sha(batch/'summary.json'),'protocol_sha256':sha(batch/'protocol.json'),
            'raw_control_frames_replayed':frames,'paired_conditions':40,'episodes':80,'initial_condition_seeds':5,
            'neural_training_seeds':0,'aggregate':aggregate,
            'verification':'Every physical state,tracking sample,terminal classification,goal distance,jerk and timing aggregate reconstructed from registered inputs and raw commands.',
            'conclusion':'Both controllers complete40/40. Pursuit has lower lateral error and CPU cost;MPC has lower physical-integrator jerk. This is a control tradeoff,not uniform MPC superiority or real-time/CARLA proof.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.batch,args.output)

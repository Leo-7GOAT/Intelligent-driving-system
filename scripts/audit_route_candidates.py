"""Compare three proposal catalogs at recorded states, without ticking CARLA."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.planning import analytic_connection
from navstack.road_grid import RoadGrid
from navstack.subgoals import route_frame_candidates
from navstack.vehicle import VehiclePlant,VehicleState
from navstack.world import World
from scripts.audit_native_candidates import match_scenes


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();batch=Path(a.batch);output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve prior diagnostic')
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    protocol=read(batch/'protocol.json');base=read(batch/'H/config.json')['env']
    scenes=match_scenes(protocol['routes'],[read(f) for f in sorted((batch/'H').glob('scene_*.json'))])
    terminal={};trace=batch/'H/evaluation_trace.jsonl.gz'
    with gzip.open(trace,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);terminal[row['case_id']]=row['info']['trace'][-1]['state']
    rows=[]
    for case,measured in enumerate(scenes):
        scene=measured['scene']
        for label,state in [('initial',measured['initial']['initial']),('terminal',terminal[case])]:
            for mode in ('ego_arcs','goal_frame_terminal','route_frame_terminal'):
                cfg=EnvConfig.from_dict({**base,'subgoal_mode':mode if mode!='route_frame_terminal' else 'ego_arcs'})
                env=NavigationEnv(cfg);env.world=World(scene['width'],scene['height'],road=RoadGrid(scene['road_grid']))
                env.plant=VehiclePlant(cfg.vehicle,VehicleState(**state));env.goal=scene['goal']
                catalog=route_frame_candidates(env.state,scene['native_layout']['path'],cfg.vehicle) if mode=='route_frame_terminal' else tuple(env.decode_action(i) for i in range(31))
                cache={};clear=[];witness=[]
                for i,(target,speed) in enumerate(catalog[1:],1):
                    if target not in cache:
                        free=env.world.free(target,cfg.vehicle)
                        path=analytic_connection((env.state.x,env.state.y,env.state.yaw),target,env.world,cfg.vehicle) if free else None
                        cache[target]=(free,path is not None)
                    free,valid=cache[target]
                    if free:clear.append(i)
                    if valid:witness.append(i)
                assert vars(env.state)==state
                rows.append({'case_id':case,'geometry_id':scene['native_layout']['geometry_id'],'state_label':label,
                    'state':state,'mode':mode,'clear_actions':clear,'analytic_witness_actions':witness,'candidates':catalog,
                    'unique_targets':len(cache),'road_sha256':scene['road_grid']['sha256']})
                env.close()
    result={'scope':'Offline geometric proposal diagnosis only; no PPO updates or actual execution, no claim analytic rejection proves absence of a Hybrid A* path',
        'source_trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'proposal_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'navstack/subgoals.py').read_bytes()).hexdigest(),
        'route_prior':'Uses known survey centreline tangent/arc length; any future learned/rule/direct comparison must receive comparable route information',
        'rows':rows}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for r in rows:print(r['case_id'],r['state_label'],r['mode'],len(r['clear_actions']),len(r['analytic_witness_actions']))


if __name__=='__main__':main()

"""Bounded offline B3 candidate/plan coverage from measured scenes, no execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.config import EnvConfig
from navstack.env import NavigationEnv,rule_action
from navstack.world import World
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehiclePlant,VehicleState


def probe(batch,output,final_goal=False):
    read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
    batch=Path(batch);protocol=read(batch/'protocol.json')
    if not (batch/'summary.json').is_file():raise ValueError('Require complete measured scene inventory')
    config=EnvConfig.from_dict({**read(batch/'source_config.json')['env'],'route_goal_candidate':final_goal})
    if config.planner.max_seconds is not None:raise ValueError('Offline comparison needs a fixed node budget')
    rows=[]
    with Path(output).open('x',encoding='utf-8') as stream:
        for case in range(len(protocol['routes'])):
            path=batch/f'scene_{case:04d}.json';scene=read(path);raw=scene['scene'];initial=scene['initial'];state=initial['initial']
            if raw['native_layout']['geometry_id']!=protocol['routes'][case]['geometry_id']:
                raise ValueError('Scene route differs from registered inventory')
            env=NavigationEnv(config)
            env.world=World(raw['width'],raw['height'],raw['static'],road=RoadGrid(raw['road_grid']))
            env.world.road_collision_backend=config.road_collision_backend
            env.plant=VehiclePlant(config.vehicle,VehicleState(state['x'],state['y'],yaw=state['yaw'],v=state['v']))
            env.goal=tuple(initial['goal']);env.navigation_route=raw['native_layout']['path']
            start=time.perf_counter();action=rule_action(env);elapsed=time.perf_counter()-start
            target,speed=env.decode_action(action);cached=getattr(env,'_rule_plan_cache',None)
            rows.append({'case_id':case,'scene_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'selected_action':action,'target':target,'target_speed':speed,'selection_seconds':elapsed,
                'search':env.rule_diagnostics,'selected_cached_plan':None if cached is None else {
                    'status':cached[2].status,'nodes':cached[2].expanded_nodes,'samples':len(cached[2].path),
                    'all_samples_free':all(env.world.free(p,config.vehicle) for p in cached[2].path)}})
            print(json.dumps(rows[-1]),flush=True)
        result={'scope':'Candidate/plan coverage on measured initial scenes; no vehicle execution or task-success claim',
            'batch':batch.name,'protocol_sha256':hashlib.sha256((batch/'protocol.json').read_bytes()).hexdigest(),
            'final_goal_candidate':final_goal,'node_budget_per_search':config.planner.max_nodes,
            'new_controls':0,'optimizer_updates':0,'rows':rows}
        json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--final-goal-candidate',action='store_true')
    args=parser.parse_args()
    probe(args.batch,args.output,args.final_goal_candidate)

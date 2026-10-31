"""Regenerate audited B3initial references using its frozen planner,no plant steps."""
import argparse
from dataclasses import asdict
from collections import deque
import gzip,hashlib,json,math,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def export(parent,output):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve existing fixed reference registration')
    protocol=read(parent/'protocol.json');audit=read(parent/'raw_audit.json');source=parent/'source'
    if audit['protocol_sha256']!=sha(parent/'protocol.json'):raise ValueError('Original audited rule protocol changed')
    for name,digest in protocol['source_hashes'].items():
        if sha(source/name)!=digest:raise ValueError('Original frozen source changed')
    sys.path.insert(0,str(source))
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.road_grid import RoadGrid
    from navstack.vehicle import VehicleState,Command
    from navstack.world import World
    from navstack.trajectory import parameterize
    import navstack.env
    if not Path(navstack.env.__file__).resolve().is_relative_to(source):raise ValueError('Fresh process must import original frozen planner')
    first={}
    with gzip.open(parent/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['case_id']<4:first.setdefault(row['case_id'],row)
    if set(first)!=set(range(4)):raise ValueError('Every original curve reference required')
    config=EnvConfig.from_dict(protocol['config']);rows=[];output.mkdir(parents=True)
    for i in range(4):
        record=read(parent/f'scene_{i:04d}.json');scene,initial=record['scene'],record['initial']
        env=NavigationEnv(config);env.world=World(scene['width'],scene['height'],scene['static'],road=RoadGrid(scene['road_grid']))
        env.world.road_collision_backend=config.road_collision_backend
        env.goal=initial['goal'];env.navigation_route=scene['native_layout']['path']
        env.plant=SimpleNamespace(state=VehicleState(**initial['initial']),previous_command=Command(),delay_queue=deque())
        env.episode={'plan_times':[],'plan_failures':0};env._rule_plan_cache=None
        action=first[i]['policy_action']
        before=env.state;mode,detail=env._prepare_reference(action)
        recorded=first[i]['info']['plan']
        if mode!='reference' or env.state!=before:raise ValueError('Fixed global reference unavailable from frozen planner')
        if action==first[i]['policy_action'] and (detail['status']!=recorded['status'] or detail['expanded_nodes']!=recorded['expanded_nodes'] or abs(detail['cost']-recorded['cost'])>1e-7):
            raise ValueError('Frozen initial reference differs from the actual recorded planning decision')
        ref=env.reference
        if not env.world.free_many(np.c_[ref.x,ref.y,ref.yaw],config.vehicle).all():raise ValueError('Reconstructed reference is not full-body free')
        spatial=np.c_[ref.x,ref.y,ref.yaw];extension=None
        if math.hypot(ref.x[-1]-env.goal[0],ref.y[-1]-env.goal[1])>.8:
            # A planner query from a PREDICTED local-reference endpoint,not a
            # substituted CARLA state. One fixed extension budget,no retries.
            predicted=tuple(map(float,spatial[-1]));plan=env.planner.plan(predicted,tuple(env.goal),env.world)
            if plan.status!='success':raise ValueError('Single declared global reference extension unavailable')
            spatial=np.vstack((spatial,plan.path[1:]))
            extension={'scope':'Counterfactual planning from predicted reference endpoint,zero plant steps',
                       'predicted_pose':predicted,'expanded_nodes':plan.expanded_nodes,'cost':plan.cost}
        ref=parameterize(spatial,1.,0.,config.vehicle,0.,stop=True)
        if not env.world.free_many(np.c_[ref.x,ref.y,ref.yaw],config.vehicle).all():raise ValueError('Joined fixed reference is not body-free')
        if math.hypot(ref.x[-1]-env.goal[0],ref.y[-1]-env.goal[1])>.8:raise ValueError('Fixed reference does not reach original global goal')
        payload={'scope':'Frozen production planner query from audited initial CARLA state,zero new controls or PPO updates.',
          'case_id':i,'geometry_id':protocol['routes'][i]['geometry_id'],'road_sha256':scene['road_grid']['sha256'],'goal':env.goal,
          'source_scene_sha256':sha(parent/f'scene_{i:04d}.json'),'recorded_action':first[i]['policy_action'],'fixed_query_action':action,
          'counterfactual_query':action!=first[i]['policy_action'],
          'global_extension':extension,'target_speed_m_s':1.,
          'recorded_plan':recorded,'reconstructed_plan':detail,
          'reference':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(ref).items()}}
        path=output/f'case_{i:02d}.json';path.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
        rows.append({'case_id':i,'file':path.name,'sha256':sha(path),'geometry_id':payload['geometry_id'],'road_sha256':payload['road_sha256']})
    manifest={'scope':'Four fixed production references for controller-only development comparison. No policy result selects a reference.',
      'parent_batch':parent.name,'parent_protocol_sha256':sha(parent/'protocol.json'),'raw_audit_sha256':sha(parent/'raw_audit.json'),
      'export_script_sha256':sha(Path(__file__)),'references':rows,'new_physical_controls':0,'PPO_updates':0}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.parent,args.output)

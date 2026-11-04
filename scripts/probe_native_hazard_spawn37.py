"""Spawn-only infrastructure preflight, never a policy evaluation or training run.

Run only on this task's empty owned server before its worker. Each requested
lateral offset is tested on every registered route, with all failures retained.
Python3.7-compatible to use the existing CARLA client.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import carla
from navstack.native_route import local_layout
from navstack.native_hazards import resolve_native_hazard


def probe(protocol, output, offsets):
    protocol=Path(protocol)
    data=json.loads(protocol.read_text(encoding='utf-8-sig'))
    claim=json.loads((protocol.parent/'infrastructure/server_start.json').read_text(encoding='utf-8-sig'))
    if not claim.get('owned_by_task') or claim['rpc_port']!=data['carla_rpc_port']:
        raise RuntimeError('Require matching owned-server startup record')
    client=carla.Client(data['carla_rpc_host'],data['carla_rpc_port']);client.set_timeout(15.)
    world=client.get_world();settings=world.get_settings()
    if settings.synchronous_mode or any(a.type_id.startswith(('vehicle.','walker.','sensor.')) for a in world.get_actors()):
        raise RuntimeError('Spawn preflight requires an empty asynchronous pre-worker server')
    road=world.get_map();digest=hashlib.sha256(road.to_opendrive().encode()).hexdigest()
    blueprint=world.get_blueprint_library().find('walker.pedestrian.0001')
    rows=[]
    # Exclusive output reserved before the first reversible actor creation.
    with Path(output).open('x',encoding='utf-8') as stream:
        for offset in offsets:
            for index,route in enumerate(data['routes']):
                layout=local_layout(route,road.name,digest)
                spec={'version':'native_hazard_v1','kind':'crossing_walker','distance_m':10.,'lateral_m':offset}
                resolved=resolve_native_hazard(layout,spec);x,y,yaw=resolved['spawn_pose']
                angle=math.radians(layout['angle']);co,si=math.cos(angle),math.sin(angle)
                transform=carla.Transform(carla.Location(x=layout['origin'][0]+x*co+y*si,
                    y=layout['origin'][1]+x*si-y*co,z=layout['spawn_z']),
                    carla.Rotation(yaw=layout['angle']-math.degrees(yaw)))
                actor=None;row={'case_id':index,'spec':spec,'world_spawn_xyz':[transform.location.x,transform.location.y,transform.location.z]}
                try:
                    actor=world.spawn_actor(blueprint,transform)
                    row.update(status='spawned',actor_id=actor.id,actor_type=actor.type_id)
                except RuntimeError as exc:
                    row.update(status='spawn_failed',error=str(exc))
                finally:
                    if actor is not None:
                        # Actor.destroy() can consult a pre-spawn client cache
                        # and return False before issuing any server request.
                        # Destroy exactly our returned actor ID through RPC.
                        response=client.apply_batch_sync([carla.command.DestroyActor(actor.id)],False)[0]
                        row['destroyed']=not bool(response.error)
                        if response.error:raise RuntimeError('Owned probe actor cleanup failed: '+response.error)
                rows.append(row)
                time.sleep(.05)
        world.wait_for_tick(5.)
        remaining=[a.id for a in world.get_actors() if a.type_id.startswith(('vehicle.','walker.','sensor.'))]
        result={'scope':'Spawn-only infrastructure preflight; no ego, controls, PPO, rewards or success-rate evidence',
            'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'rows':rows,'remaining_actor_ids':remaining}
        json.dump(result,stream,indent=2)
        if remaining:raise RuntimeError('Server no longer empty after preflight')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--lateral',type=float,nargs='+',default=[-3.5,-2.5,2.5,3.5])
    args=parser.parse_args()
    probe(args.protocol,args.output,args.lateral)

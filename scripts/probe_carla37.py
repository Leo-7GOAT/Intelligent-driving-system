"""Read-only CARLA 0.9.15 probe; runs under the isolated CPython 3.7 client."""
from __future__ import print_function
import argparse
import json
import os
import platform
import sys
import time
import carla

parser=argparse.ArgumentParser()
parser.add_argument('--host',default='127.0.0.1')
parser.add_argument('--port',type=int,default=2000)
parser.add_argument('--output',required=True)
parser.add_argument('--ego-snapshot',action='store_true',help='Read one frame snapshot of the uniquely named task ego; never tick or control')
args=parser.parse_args()
result={'host':args.host,'port':args.port,'interpreter':sys.executable,'python':sys.version,
        'platform':platform.platform(),'client_package':carla.__file__,'observed_epoch':time.time()}
try:
    client=carla.Client(args.host,args.port);client.set_timeout(15.)
    world=client.get_world()
    result.update(client_version=client.get_client_version(),server_version=client.get_server_version(),
                  available_maps=client.get_available_maps(),map=world.get_map().name,
                  actors=[{'id':a.id,'type':a.type_id,'role':a.attributes.get('role_name')} for a in world.get_actors() if a.type_id.startswith(('vehicle.','walker.','sensor.'))],
                  synchronous_mode=world.get_settings().synchronous_mode,
                  fixed_delta_seconds=world.get_settings().fixed_delta_seconds,status='CONNECTED')
    if result['client_version']!='0.9.15' or result['server_version']!='0.9.15':result['status']='VERSION_MISMATCH'
    if args.ego_snapshot and result['status']=='CONNECTED':
        # In synchronous mode a newly connected client can have an empty stale
        # actor cache until the owner's next tick. Wait for that frame; never
        # produce a tick ourselves or infer actor absence from the stale cache.
        snapshot=world.wait_for_tick(30.)
        result['actors']=[{'id':a.id,'type':a.type_id,'role':a.attributes.get('role_name')}
                         for a in world.get_actors() if a.type_id.startswith(('vehicle.','walker.','sensor.'))]
        owned=[a for a in world.get_actors() if a.type_id.startswith('vehicle.')
               and a.attributes.get('role_name')=='codex_ppo_ego']
        if len(owned)!=1:raise RuntimeError('Expected exactly one task ego for read-only snapshot')
        actor=snapshot.find(owned[0].id)
        if actor is None:raise RuntimeError('Ego is absent from the selected world snapshot')
        transform=actor.get_transform();velocity=actor.get_velocity()
        result['ego_snapshot']={'frame':snapshot.frame,'elapsed_seconds':snapshot.timestamp.elapsed_seconds,
            'actor_id':owned[0].id,'location_xyz':[transform.location.x,transform.location.y,transform.location.z],
            'yaw_degrees':transform.rotation.yaw,'velocity_xyz':[velocity.x,velocity.y,velocity.z],
            'scope':'One read-only CARLA frame; no tick, spawn, control, or evaluation-outcome claim'}
except Exception as exc:
    result.update(status='SNAPSHOT_UNAVAILABLE' if args.ego_snapshot and result.get('status')=='CONNECTED'
                  else 'CONNECTION_FAILED',error=repr(exc))
with open(args.output,'w',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result,indent=2))

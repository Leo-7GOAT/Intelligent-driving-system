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
except Exception as exc:result.update(status='CONNECTION_FAILED',error=repr(exc))
with open(args.output,'w',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result,indent=2))

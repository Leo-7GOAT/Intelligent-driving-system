"""Read-only semantic ground rays on the recorded owned CARLA server; never tick or control."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import carla


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(input_path,server_record,port,owned_module):
    data,server=read(input_path),read(server_record)
    if server.get('owned_by_task') is not True:raise ValueError('Require a task-owned server record')
    if server.get('rpc_host')!='127.0.0.1' or port!=server.get('rpc_port'):
        raise ValueError('Query only the recorded local owned server endpoint')
    stamp=re.sub(r'(\.\d{6})\d+',r'\1',server['shipping_creation_utc'])
    expected=datetime.fromisoformat(stamp).timestamp()
    spec=importlib.util.spec_from_file_location('owned_identity',str(owned_module))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    owner=module.OwnedProcess(server['shipping_pid'],expected,server['launched_executable'],allow_terminate=False)
    try:
        if owner.poll() is not None:raise ValueError('Owned server already ended')
        client=carla.Client('127.0.0.1',port);client.set_timeout(10.)
        if client.get_client_version()!='0.9.15' or client.get_server_version()!='0.9.15':
            raise ValueError('Require matching installed0.9.15 SDK and server')
        world=client.get_world()
        if hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest()!=data['opendrive_sha256']:
            raise ValueError('Server map differs from recorded native geometry')
        before=world.get_snapshot().frame;x,y,z=data['recorded_offroad_point_world'];queries=[]
        for dx,dy in ((0.,0.),(.02,0.),(-.02,0.),(0.,.02),(0.,-.02)):
            hits=world.cast_ray(carla.Location(x=x+dx,y=y+dy,z=z+2.),carla.Location(x=x+dx,y=y+dy,z=z-1.))
            queries.append(dict(xy=[x+dx,y+dy],hits=[dict(label=str(p.label),
                location=[p.location.x,p.location.y,p.location.z]) for p in hits]))
        if owner.poll() is not None:raise ValueError('Owned server changed during read-only probe')
        return dict(scope=__doc__,script_sha256=sha(__file__),input_sha256=sha(input_path),
            server_record_sha256=sha(server_record),owned_identity_source_sha256=sha(owned_module),
            opendrive_sha256=data['opendrive_sha256'],client_version=client.get_client_version(),
            server_version=client.get_server_version(),owned_server_pid=server['shipping_pid'],
            frame_before=before,frame_after=world.get_snapshot().frame,queries=queries,
            issued_tick_requests=0,issued_control_requests=0,
            limits='Concurrent owned evaluation may advance frames. Engine surface labels do not alone prove lane legality or swept-body safety. Original failures and termination rules are unchanged.')
    finally:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True)
    p.add_argument('--owned-server-record',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--owned-module',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Preserve original ground-probe evidence')
    result=run(a.input,a.owned_server_record,a.port,a.owned_module)
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps([{'xy':r['xy'],'labels':[p['label'] for p in r['hits']]} for r in result['queries']]))

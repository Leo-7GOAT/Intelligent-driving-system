"""Operational read-only progress evidence from an already-owned synchronous world."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.owned_process import OwnedProcess


def run(batch,identity):
    import carla
    batch=Path(batch).resolve();identity=Path(identity)
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    p=read(batch/'protocol.json');ids=read(identity);start=read(batch/'infrastructure/server_start.json')
    if not start.get('owned_by_task') or ids['server']['pid']!=start['shipping_pid']:
        raise ValueError('Owned server identity required')
    handles=[OwnedProcess(**ids[key],allow_terminate=False) for key in ['worker','server']]
    def save(path,value):
        temporary=path.with_suffix('.frame_watch.tmp');temporary.write_text(json.dumps(value),encoding='utf-8');os.replace(str(temporary),str(path))
    save(batch/'native_frame_watch_registration.json',dict(scope=__doc__,identity=ids,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_controls=0,new_ticks=0))
    client=carla.Client(p.get('carla_rpc_host','127.0.0.1'),p['carla_rpc_port']);client.set_timeout(5.)
    world=client.get_world();last=None;last_write=0.;observations=0
    try:
        while time.time()<p['deadline_epoch'] and all(h.poll() is None for h in handles):
            try:snapshot=world.wait_for_tick(20.)
            except RuntimeError:continue
            if last is not None and snapshot.frame>last and time.time()-last_write>=5:
                observations+=1;last_write=time.time()
                save(batch/'progress.json',dict(scope='Read-only native frame advancement,not policy optimization counters',
                    observer_pid=os.getpid(),worker_pid=ids['worker']['pid'],frame=snapshot.frame,
                    previous_observed_frame=last,epoch=last_write,observations=observations,new_controls=0,new_ticks=0))
            last=snapshot.frame
    finally:
        for h in handles:h.kernel.CloseHandle(h.handle)
        save(batch/'native_frame_watch_final.json',dict(epoch=time.time(),observations=observations,new_controls=0,new_ticks=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.batch,a.identity)

"""One fresh local-witness H trial against the completed unshared control."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.config import VehicleConfig
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import audit,read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config


def prepare(parent):
    parent=Path(parent).resolve();require_finished_parent(parent)
    old=read(parent/'protocol.json');control='H_unshared_v1'
    if not old.get('actor_head_experiment') or control not in old['methods'] or old.get('recovery'):
        raise ValueError('Require the completed fresh head comparison')
    if (old['controlled_budget_per_method'],old['rollout_actions'],old['ppo_epochs'])!=(4096,32,10):
        raise ValueError('Use the fixed matched control budget/schedule')
    if audit(parent,control)!=read(parent/control/'endpoint_raw_audit.json'):
        raise ValueError('Control raw audit absent or stale')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed')
    for name in ['planning.py','world.py','road_grid.py','road_grid_batch.py','trajectory.py',
                 'mpc.py','carla_bridge.py','carla_actuation.py','vehicle.py','native_route.py']:
        if sha(ROOT/'navstack'/name)!=sha(parent/'source/navstack'/name):
            raise ValueError('This single-factor trial must preserve low-level code: '+name)
    if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:
        raise ValueError('Calibration changed')
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    with ProcessLease(ROOT/'runs/native_local_filter_prepare.lock'):
        pointer=ROOT/'runs/native_local_filter.json'
        if pointer.exists():raise FileExistsError('Preserve existing local-filter registration')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_local_filter')
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'calibration',batch/'calibration')
        for directory in ('navstack','scripts','tests'):
            for source in (ROOT/directory).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and (
                        source.suffix in ('.py','.ps1') or 'fixtures' in source.parts):
                    dest=batch/'source'/source.relative_to(ROOT)
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        protocol=deepcopy(old);protocol.pop('jobs');protocol.pop('actor_head_experiment')
        now=time.time()
        protocol.update(scope='Fresh H sufficient-local-connector filter; matched completed unshared control, known full roads; not learned-contribution or independent confirmation',
            methods=['H'],candidate_head='unshared_local_v1',created_epoch=now,deadline_epoch=now+900,
            parent_batch=parent.name,parent_control=control,parent_raw_audit_sha256=sha(parent/control/'endpoint_raw_audit.json'),
            hypothesis='Avoid repeatedly selecting a connector without a local witness when other witnessed local references exist; missing global continuation must not reject actions',
            comparison='Fresh seed11, same unshared tensors/critic/RNG, reward, normalization, PPO,4096controls,all4full routes and unchanged low level; only local action filtering differs',
            representation_note='Wait always allowed; use clear/local witness only, ignore goal witness; if none known permit all31actions. No rule picks the PPO action. B3 shares the exact filter; B1 retains common candidate information. Extra geometric prior must not be claimed as learned value.',
            stopping_rule='One fixed4096-control endpoint with all4probes,900s wall, zero retries; retain every failure. Separate initialization/rule/feedback comparisons before any learned-value claim.',
            resource_limits={**old['resource_limits'],'wall_seconds':900})
        cal=read(batch/'calibration/calibration.json');expected=read(parent/control/'config.json')['env']
        expected['local_candidate_filter']=True
        if training_config(protocol,'H',VehicleConfig(**cal['vehicle'])).to_dict()!=expected:
            raise ValueError('Unexpected environment change beyond the registered filter')
        protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p)
            for p in (batch/'source').rglob('*') if p.is_file()}
        write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        write_json(pointer,{'batch':str(batch),'protocol_sha256':sha(batch/'protocol.json')})
    print(batch,flush=True);return batch


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True)
    prepare(p.parse_args().parent)

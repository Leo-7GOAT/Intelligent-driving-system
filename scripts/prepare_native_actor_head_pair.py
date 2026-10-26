"""Matched fresh H actor-head experiment on the four original full routes."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import audit,read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import registered_jobs


def prepare(parent):
    parent=Path(parent).resolve();require_finished_parent(parent)
    old=read(parent/'protocol.json')
    if old.get('methods')!=['H0','H','B1'] or old.get('jobs') or old.get('recovery'):
        raise ValueError('Require the complete fresh three-method parent')
    if old.get('subgoal_mode')!='route_frame_terminal' or old.get('episode_seconds')!=45:
        raise ValueError('Require the four original full-route development tasks')
    if audit(parent,'H')!=read(parent/'H/endpoint_raw_audit.json'):
        raise ValueError('Parent H audit absent or stale')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed')
    if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    with ProcessLease(ROOT/'runs/native_actor_head_prepare.lock'):
        pointer=ROOT/'runs/native_actor_head_pair.json'
        if pointer.exists():raise FileExistsError('Preserve existing actor-head registration')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_actor_head')
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'calibration',batch/'calibration')
        for folder in ('navstack','scripts','tests'):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and (
                        source.suffix in ('.py','.ps1') or 'fixtures' in source.parts):
                    target=batch/'source'/source.relative_to(ROOT)
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        jobs=[{'name':'H_'+head,'method':'H','gradient_clip_mode':'global','candidate_head':head}
              for head in ('shared','unshared_v1')]
        protocol=deepcopy(old);now=time.time()
        protocol.update(scope='Fresh matched H shared-versus-unshared decision head; original full training roads, one seed, no formal hierarchy/feedback superiority claim',
            created_epoch=now,deadline_epoch=now+1800,methods=[j['name'] for j in jobs],jobs=jobs,
            actor_head_experiment=True,parent_batch=parent.name,parent_raw_audit_sha256=sha(parent/'H/endpoint_raw_audit.json'),
            controlled_budget_per_method=4096,checkpoint_thresholds=[4096],rollout_actions=32,ppo_epochs=10,
            road_collision_backend='batch_sat_v1',
            hypothesis='Independent logits can learn global-state dependence that cancels in a shared linear score of identical candidate embeddings; actual execution must demonstrate any improvement',
            comparison='Fresh seed11 both arms; identical seeded encoders/critic and construction RNG, same reward/normalization/full routes/PPO/low level/budget; only final actor head differs',
            differences_from_original_parent=['ten epochs shared by both arms','batch_sat_v1 shared by both arms','4096control sanity budget, all four original full-route probes retained'],
            representation_note='Shared head remains default. Unshared head adds30750parameters and sacrifices candidate permutation equivariance; fixed route-catalog slots retained; no mask, teacher, rule selector or planner change',
            stopping_rule='One endpoint per arm with all4full-route probes; no automatic retries; do not enlarge without actual learning evidence',
            resource_limits={**old['resource_limits'],'wall_seconds':1800})
        protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p)
            for p in (batch/'source').rglob('*') if p.is_file()}
        registered_jobs(protocol)
        write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        write_json(pointer,{'batch':str(batch),'protocol_sha256':sha(batch/'protocol.json')})
    print(batch,flush=True);return batch


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True)
    prepare(parser.parse_args().parent)

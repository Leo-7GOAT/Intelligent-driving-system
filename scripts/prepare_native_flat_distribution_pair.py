"""Fresh clipped/tanh Gaussian B1 sanity pair on short measured-road prefixes."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.experiments import write_json
from navstack.native_route import local_layout
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import audit, read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import registered_jobs


def short_route(route, length=6.):
    points = route['points']
    distance = 0.
    end = None
    for index in range(1, len(points)):
        distance += math.hypot(points[index]['x']-points[index-1]['x'],
                               points[index]['y']-points[index-1]['y'])
        if distance >= length:
            end = index+1
            break
    if end is None:
        raise ValueError('Parent route is shorter than the registered sanity task')
    result = deepcopy(route)
    result['points'] = deepcopy(points[:end])
    result['geometry_id'] = hashlib.sha256(json.dumps([
        [round(p[k],3) for k in ('x','y','yaw_degrees_carla')]
        for p in result['points']]).encode()).hexdigest()
    result['parent_geometry_id'] = route['geometry_id']
    result['parent_length_m'] = route.get('length_m')
    result['parent_heading_change_degrees'] = route.get('heading_change_degrees')
    result['length_m'] = distance
    result['heading_change_degrees'] = (
        result['points'][-1]['yaw_degrees_carla']-result['points'][0]['yaw_degrees_carla']+180)%360-180
    result.pop('selection_target_degrees',None)
    result['sanity_prefix_length_m'] = distance
    local_layout(result, result['map'], result['opendrive_sha256'])
    return result


def prepare(parent, controls=4096, wall_seconds=1800):
    if type(controls) is not int or not 2048 <= controls <= 16384:
        raise ValueError('Bounded sanity budget requires 2048..16384 controls per arm')
    if not 900 <= wall_seconds <= 3600:
        raise ValueError('Bounded sanity window requires 900..3600 seconds')
    parent = Path(parent).resolve()
    require_finished_parent(parent)
    old = read(parent/'protocol.json')
    if old.get('recovery') or old.get('jobs') or old.get('methods') != ['H0','H','B1']:
        raise ValueError('Require the completed fresh three-method parent')
    if old.get('subgoal_mode') != 'route_frame_terminal':
        raise ValueError('Require the versioned route geometry')
    for name, digest in old['source_hashes'].items():
        if sha(parent/'source'/name) != digest:
            raise ValueError('Parent frozen source changed')
    if sha(parent/'calibration/calibration.json') != old['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    if audit(parent,'B1') != read(parent/'B1/endpoint_raw_audit.json'):
        raise ValueError('Parent B1 raw audit absent or stale')
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError('Resource guard: '+str(reason))
    with ProcessLease(ROOT/'runs/native_flat_distribution_prepare.lock'):
        pointer = ROOT/'runs/native_flat_distribution_pair.json'
        if pointer.exists():
            raise FileExistsError('Preserve the existing distribution pair registration')
        batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_flat_distribution')
        batch.mkdir()
        (batch/'infrastructure').mkdir()
        shutil.copytree(parent/'calibration',batch/'calibration')
        for directory in ('navstack','scripts','tests'):
            for source in (ROOT/directory).rglob('*'):
                if source.is_file() and ('__pycache__' not in source.parts) and (
                        source.suffix in ('.py','.ps1') or 'fixtures' in source.parts):
                    target = batch/'source'/source.relative_to(ROOT)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
        protocol = deepcopy(old)
        now = time.time()
        jobs = [{'name':'B1_'+name,'method':'B1','gradient_clip_mode':'global',
                 'flat_action_distribution':name}
                for name in ('clipped_gaussian','tanh_gaussian_v1')]
        protocol.update(scope='Fresh B1 six-metre native-road sanity pair; sampled Gaussian clipping versus corrected tanh density; not full-task competence or independent confirmation',
            created_epoch=now,deadline_epoch=now+wall_seconds,
            flat_distribution_experiment=True,jobs=jobs,methods=[j['name'] for j in jobs],
            routes=[short_route(r) for r in old['routes']],parent_batch=parent.name,
            parent_raw_audit_sha256=sha(parent/'B1/endpoint_raw_audit.json'),
            controlled_budget_per_method=controls,checkpoint_thresholds=[controls],
            rollout_actions=32,ppo_epochs=10,episode_seconds=12.,
            road_collision_backend='batch_sat_v1',
            hypothesis='Bounded sampled actions may improve basic flat-PPO learning by avoiding many-to-one actuator clipping; no asserted root cause',
            comparison='Both arms fresh seed11, same short tasks/reward/encoder/normalization/optimizer/interaction budget; only action distribution differs',
            differences_from_parent=['six-metre route prefixes','12-second episode deadline','ten PPO epochs shared by both arms','batch_sat_v1 shared by both arms; no inherited weights'],
            success_gate='All four deterministic sanity probes complete safely; no full-route or generalization claim',
            stopping_rule='One fixed endpoint per arm, all four probes retained; no automatic retry. If unsuccessful, retain negative result and do not expand this variant without new evidence.',
            resource_limits={'wall_seconds':wall_seconds,'disk_min_GiB':12,'ram_min_GiB':4,'gpu_stop_temperature_C':85,'automatic_retries':0},
            disk_free_bytes=shutil.disk_usage(ROOT).free)
        protocol['source_hashes'] = {p.relative_to(batch/'source').as_posix():sha(p)
            for p in (batch/'source').rglob('*') if p.is_file()}
        registered_jobs(protocol)
        write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        write_json(pointer,{'batch':str(batch),'protocol_sha256':sha(batch/'protocol.json')})
    print(batch,flush=True)
    return batch


def prepare_cadence(parent,hold=2,controls=20480,wall_seconds=3600):
    """Fresh cadence development against an already completed historical control."""
    if hold not in (2,10) or type(controls) is not int or not 2048<=controls<=100000 or not 900<=wall_seconds<=7200:
        raise ValueError('Cadence development budget outside supported bounds')
    parent=Path(parent).resolve();require_finished_parent(parent);old=read(parent/'protocol.json')
    if old.get('methods')!=['B1'] or old.get('flat_action_catalog')!='speed_steer_3x7_v1':
        raise ValueError('Require completed categorical direct B1 development')
    baseline=audit(parent,'B1')
    if baseline!=read(parent/'B1/endpoint_raw_audit.json'):raise ValueError('Historical control audit changed')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Historical source changed')
    if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:raise ValueError('Calibration changed')
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    with ProcessLease(ROOT/'runs/native_cadence_prepare.lock'):
        pointer=ROOT/'runs/native_flat_cadence.json'
        if pointer.exists():raise FileExistsError('Preserve registered cadence development')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_flat_cadence')
        batch.mkdir();(batch/'infrastructure').mkdir();shutil.copytree(parent/'calibration',batch/'calibration')
        for directory in ('navstack','scripts','tests'):
            for source in (ROOT/directory).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and (source.suffix in ('.py','.ps1') or 'fixtures' in source.parts):
                    target=batch/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        protocol=deepcopy(old)
        for key in ('jobs','recovery','evaluation_only','own_success_replay','auxiliary_actor_fit','parent_records',
                    'method_budgets','candidate_head','freeze_observation_statistics','flat_distribution_experiment',
                    'actor_head_experiment','gradient_clipping_experiment','rollout_length_experiment','curriculum'):
            protocol.pop(key,None)
        now=time.time();rollout=320//hold
        protocol.update(scope='Fresh native categorical flat PPO cadence development; no teacher/replay/inherited weights; historical2Hzcontrol includes one restart, not a strict causal frequency ablation',
            fresh_cadence_experiment=True,flat_hold=hold,rollout_actions=rollout,ppo_batch_size=rollout,ppo_epochs=10,
            controlled_budget_per_method=controls,checkpoint_thresholds=[min(controls,4096),min(controls,10000),controls],
            created_epoch=now,deadline_epoch=now+wall_seconds,parent_batch=parent.name,
            historical_control={'batch':parent.name,'endpoint_raw_audit_sha256':sha(parent/'B1/endpoint_raw_audit.json'),
                'lineage_controls':baseline['lineage_controls'],'restart_scope':'Completed control is reused, not retrained; its continuation reset is disclosed'},
            hypothesis='More frequent direct steering may learn the longer curve without a planner; keep nominal320control rollout, per-second gamma/GAE and ten one-minibatch epochs',
            comparison='Same known6mgoals/12sdeadline/reward/vehicle/actuator/encoder/seed11 and nominal20480physical budget as the retained2Hzdevelopment lineage; fresh10Hz versus resumed2Hz is exploratory',
            stopping_rule='One fixed physical-budget endpoint, all four frozen goals and all failures retained; no automatic retry or independent-confirmation claim',
            resource_limits={**old['resource_limits'],'wall_seconds':wall_seconds},disk_free_bytes=shutil.disk_usage(ROOT).free)
        protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file()}
        registered_jobs(protocol);write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        write_json(pointer,{'batch':str(batch),'protocol_sha256':sha(batch/'protocol.json')})
    print(batch,flush=True);return batch


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True)
    parser.add_argument('--controls',type=int,default=4096)
    parser.add_argument('--wall-seconds',type=int,default=1800)
    parser.add_argument('--cadence-hold',type=int,choices=[2,10])
    args=parser.parse_args()
    prepare(args.parent,args.controls,args.wall_seconds) if args.cadence_hold is None else prepare_cadence(args.parent,args.cadence_hold,args.controls,args.wall_seconds)

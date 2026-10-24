"""Fresh H0/H/B1 policies after audited route-catalog rule coverage, never resume."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import VehicleConfig
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_rule import audit, read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config


def require_coverage(parent):
    require_finished_parent(parent)
    protocol = read(parent/'protocol.json')
    if protocol['config']['subgoal_mode'] != 'route_frame_terminal':
        raise ValueError('Require the versioned route catalog')
    raw = audit(parent)
    if raw != read(parent/'raw_audit.json'):
        raise ValueError('Rule raw audit is absent or stale')
    if raw['cases_verified'] != 4 or raw['termination_counts'] != {'success': 4}:
        raise ValueError('Registered development coverage gate requires all four tasks')
    return protocol


def prepare(parent, controls=20480, wall_seconds=21600):
    if type(controls) is not int or not 10240 <= controls <= 100000:
        raise ValueError('Use a bounded initial 10240..100000 control budget')
    if not 900 <= wall_seconds <= 21600:
        raise ValueError('Use a bounded 900..21600 second window')
    parent = Path(parent).resolve()
    with ProcessLease(ROOT/'runs/native_route_learning_prepare.lock'):
        pointer = ROOT/'runs/native_route_learning.json'
        if pointer.exists():
            raise FileExistsError('Preserve the existing route-learning registration')
        old = require_coverage(parent)
        okay, reason = resources_ok(ROOT)
        if not okay or shutil.disk_usage(ROOT).free < 12*1024**3:
            raise RuntimeError('Resource guard: '+str(reason))
        now = time.time()
        batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_route_learning')
        protocol = {key: old[key] for key in ('routes', 'calibration_sha256', 'carla_rpc_host', 'carla_rpc_port', 'carla_rpc_port_span')}
        protocol.update(scope='Fresh route-catalog H0/H/B1 PPO development; no inherited models or independent confirmation',
            created_epoch=now, deadline_epoch=now+wall_seconds,
            parent_rule_batch=parent.name, parent_rule_audit_sha256=sha(parent/'raw_audit.json'),
            methods=['H0','H','B1'], seed=11, controlled_budget_per_method=controls,
            checkpoint_thresholds=sorted({10000,controls}), rollout_actions=32, ppo_epochs=2,
            episode_seconds=old['config']['deadline'], subgoal_mode='route_frame_terminal',
            continuation_feedback=old['config']['execution_feedback']=='continuation',
            road_geometry_version=old['config']['road_geometry_version'],
            feedback_ablation=True, road_candidate_policy=True, deterministic_training_route_evaluation=True,
            cleanup_owned_carla=True, reset_ticks_counted_separately=True,
            representation_note='Known route tangent candidates shared by H0/H/B1; H0 zeroes witness/outcome feedback only; no action mask or rule substitution',
            resource_limits={'wall_seconds':wall_seconds,'disk_min_GiB':12,'ram_min_GiB':4,'gpu_stop_temperature_C':85,'automatic_retries':0},
            disk_free_bytes=shutil.disk_usage(ROOT).free)
        vehicle = VehicleConfig(**old['config']['vehicle'])
        if training_config(protocol,'H',vehicle).to_dict() != old['config']:
            raise ValueError('Fresh H training differs from the demonstrated environment')
        batch.mkdir()
        (batch/'infrastructure').mkdir()
        shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent/'calibration',batch/'calibration')
        shutil.copy2(Path(__file__),batch/'source/scripts'/Path(__file__).name)
        protocol['source_hashes'] = {p.relative_to(batch/'source').as_posix():sha(p)
            for p in (batch/'source').rglob('*') if p.is_file() and p.suffix in ('.py','.ps1')}
        write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        with pointer.open('x',encoding='utf-8') as stream:
            import json
            json.dump({'batch':str(batch),'protocol_sha256':sha(batch/'protocol.json')},stream,indent=2)
    print(batch,flush=True)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent',required=True)
    parser.add_argument('--controls',type=int,default=20480)
    parser.add_argument('--wall-seconds',type=int,default=21600)
    args = parser.parse_args()
    prepare(args.parent,args.controls,args.wall_seconds)

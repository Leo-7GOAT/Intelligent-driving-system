"""Copy a pristine supervised initializer to explicit v4 geometry before evaluation.

No old snapshot, checkpoint, task gate, policy tensor or optimizer is changed.
The resulting native representation gate must finish before PPO preparation.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import VehicleConfig
from navstack.experiments import write_json
from navstack.independent_clip_ppo import load_ppo
from navstack.native_route import REFINED_GRID_VERSION
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.prepare_native_geometry_migration import OVERRIDES, migrated_config
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def require_pristine(parent, protocol):
    prepared = read(parent/'prepared.json')
    methods=protocol.get('methods')
    if (prepared['status'] != 'PREPARED_NOT_STARTED'
            or prepared['protocol_sha256'] != sha(parent/'protocol.json')
            or protocol.get('evaluation_only') is not True
            or methods not in (['H', 'H0'], ['B1'])
            or (methods==['H','H0'] and protocol.get('shared_endpoint_mask_v1') is not True)
            or protocol.get('road_geometry_version') != 'carla_driving_indexed_union_64_v3'
            or protocol.get('controlled_budget_per_method') != 0
            or protocol.get('method_budgets') != {method:0 for method in methods}):
        raise ValueError('Require pristine registered zero-PPO H/H0 or direct B1 initializer')
    forbidden = ('launcher_execution.json', 'activation.json', 'guard_final.json',
                 'failure.json', 'launcher_failure.json', 'H', 'H0', 'B1')
    if any((parent/name).exists() for name in forbidden) or any((parent/'infrastructure').iterdir()):
        raise ValueError('Initializer has execution evidence; preserve its lifecycle')


def prepare(parent):
    parent = Path(parent).resolve()
    protocol = read(parent/'protocol.json')
    require_pristine(parent, protocol)
    for name, digest in protocol['source_hashes'].items():
        if sha(parent/'source'/name) != digest:
            raise ValueError('Frozen initializer source differs')
    if sha(parent/'calibration/calibration.json') != protocol['calibration_sha256']:
        raise ValueError('Physical calibration differs')
    dataset = parent/'measured_demonstrations.npz'
    if sha(dataset) != protocol['extra_supervision']['dataset_sha256']:
        raise ValueError('Teacher input provenance differs')
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError(str(reason))
    vehicle = VehicleConfig(**read(parent/'calibration/calibration.json')['vehicle'])
    target = deepcopy(protocol)
    target['road_geometry_version'] = REFINED_GRID_VERSION
    proof = {}
    for method in protocol['methods']:
        origin = parent/'resume'/method
        metadata = verified_checkpoint_metadata(origin)
        model = load_ppo(origin/'model.zip', device='cpu')
        if (metadata['physics_steps'] or metadata['optimizer_steps']
                or model.num_timesteps or model._n_updates or model.policy.optimizer.state):
            raise ValueError('Require untouched PPO counters and optimizer')
        migrated_config(read(origin/'config.json'), target, vehicle, method)
        proof[method] = metadata
    with ProcessLease(parent/'initializer_geometry_prepare.lock'):
        registration = parent/'initializer_geometry_v4_registration.json'
        if registration.exists():
            raise FileExistsError('Do not repeat registered initializer migration')
        batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_initialization_v4')
        batch.mkdir()
        (batch/'infrastructure').mkdir()
        shutil.copytree(parent/'source', batch/'source', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent/'calibration', batch/'calibration')
        shutil.copytree(parent/'resume', batch/'resume')
        shutil.copy2(dataset, batch/dataset.name)
        for method in protocol['methods']:
            origin, destination = parent/'resume'/method, batch/'resume'/method
            write_json(destination/'config.json', migrated_config(read(origin/'config.json'), target, vehicle, method))
            for name in ('model.zip', 'normalization.pkl', 'resume_state.pkl'):
                if sha(destination/name) != sha(origin/name):
                    raise ValueError('Initializer model, statistics or RNG changed')
            if verified_checkpoint_metadata(destination) != proof[method]:
                raise ValueError('Initializer metadata changed')
        for relative in OVERRIDES:
            shutil.copy2(ROOT/relative, batch/'source'/relative)
        shutil.copy2(Path(__file__), batch/'source/scripts'/Path(__file__).name)
        now = time.time()
        target.update(created_epoch=now, deadline_epoch=now+1800, parent_batch=parent.name,
            resource_limits={**protocol['resource_limits'], 'wall_seconds': 1800},
            scope='Unchanged zero-PPO supervised weights on explicit v4 geometry; complete development representation gate,not trained seeds or final confirmation',
            initializer_geometry_migration={
                'from': protocol['road_geometry_version'], 'to': REFINED_GRID_VERSION,
                'parent_protocol_sha256': sha(parent/'protocol.json'),
                'checkpoint_bytes_unchanged': ['model.zip', 'normalization.pkl', 'resume_state.pkl'],
                'changed_environment_fields': ['road_geometry_version'],
                'new_controls': 0, 'PPO_updates': 0, 'supervised_updates': 0,
                'teacher_scope': 'Original v3 training-only teacher views reused and charged; no new supervision'})
        target['source_hashes'] = {p.relative_to(batch/'source').as_posix(): sha(p)
            for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        write_json(batch/'protocol.json', target)
        receipt = {'batch': batch.name, 'protocol_sha256': sha(batch/'protocol.json')}
        write_json(batch/'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': receipt['protocol_sha256']})
        with registration.open('x', encoding='utf-8') as stream:
            import json
            json.dump(receipt, stream, indent=2)
        print(batch, flush=True)
        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    prepare(parser.parse_args().parent)

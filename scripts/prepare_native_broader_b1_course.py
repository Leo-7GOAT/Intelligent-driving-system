"""Register a single bounded36TRAIN/24validation PPO study after the preserved basic cohort."""
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
from navstack.experiments import signature
from navstack.native_partitions import validate_native_partitions
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the registered broader learning study')
    parent = ROOT / 'runs/20261101T121859752667Z_native_B1_shared_parking_cohort_seed22'
    inventory = ROOT / 'runs/20261101T011945179519Z_native_matched_broader_feedback'
    require_finished_parent(parent)
    old = read(parent / 'protocol.json')
    result = read(ROOT / 'runs/B1_shared_parking_five_seed_execution_20261101/seed22_learning_audit.json')
    if not result['endpoint']['basic_gate_passed'] or result['new_actual_PPO_controls'] != 10240:
        raise ValueError('Canonical development seed22 requires its actual completed basic learning gate')
    source = parent / 'source'
    for name, digest in old['source_hashes'].items():
        if sha(source / name) != digest:
            raise ValueError('Frozen basic-course source changed')
    origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    if metadata['model_sha256'] != result['final_model_sha256']:
        raise ValueError('Require the fixed final actual-trained endpoint')
    p = deepcopy(old)
    reference = read(inventory / 'protocol.json')
    for name in ('routes', 'native_case_manifest', 'native_validation'):
        p[name] = deepcopy(reference[name])
    partition = validate_native_partitions(p)
    if partition['training_conditions'] != 36 or partition['validation_conditions'] != 24:
        raise ValueError('Require the whole existing training/validation inventory')
    calibration = read(parent / 'calibration/calibration.json')
    cfg = training_config(p, 'B1', VehicleConfig(**calibration['vehicle']))
    if signature(cfg) != read(origin / 'config.json')['signature']:
        raise ValueError('Broader distribution changes learned policy/environment semantics')
    okay, reason = resources_ok(ROOT)
    if not okay:
        raise RuntimeError(str(reason))
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_B1_broader_shared_parking_PPO')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    shutil.copytree(source, batch / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent / 'calibration', batch / 'calibration')
    shutil.copytree(origin, batch / 'resume/B1')
    if (parent / 'measured_demonstrations.npz').exists():
        shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    # Preserve all numerical/control modules. The current worker/case dispatcher
    # supports separate validation;bridge changes only record settled walker Z.
    overlays = ['scripts/native_road_learning.py', 'navstack/native_cases.py', 'navstack/native_partitions.py', 'scripts/carla_bridge37.py']
    for name in overlays:
        shutil.copy2(ROOT / name, batch / 'source' / name)
        p['source_hashes'][name] = sha(batch / 'source' / name)
    preserved = ['navstack/env.py', 'navstack/config.py', 'navstack/vehicle.py', 'navstack/carla_adapter.py',
        'navstack/carla_actuation.py', 'navstack/carla_bridge.py', 'navstack/road_grid.py', 'navstack/road_candidate_policy.py',
        'navstack/planning.py', 'navstack/mpc.py', 'navstack/trajectory.py', 'navstack/parking_assist.py']
    for name in preserved:
        if sha(batch / 'source' / name) != old['source_hashes'][name]:
            raise ValueError('Unrelated numerical/control source changed:' + name)
    p.pop('shared_parking_PPO')
    p.update(method_budgets={'B1': 20480}, controlled_budget_per_method=20480,
        checkpoint_thresholds=list(range(1280, 20481, 1280)),
        scope='Single canonical healthy-basic seed22 learns on36existingTRAINconditions with24existingdevelopmentvalidation. Extra actual PPO,not five-seed/full-task qualification or final confirmation.',
        resource_limits={**old['resource_limits'], 'wall_seconds': 7200, 'automatic_retries': 0},
        broader_B1_PPO={'version': 'existing36TRAIN_broader_B1_shared_wrapper_v1',
            'new_actual_PPO_controls': 20480, 'source_model_sha256': metadata['model_sha256'],
            'normalization_sha256': metadata['normalization_sha256'], 'new_supervised_updates': 0,
            'inventory_protocol_sha256': sha(inventory / 'protocol.json'), 'partition': partition,
            'parent_basic_audit_sha256': sha(ROOT / 'runs/B1_shared_parking_five_seed_execution_20261101/seed22_learning_audit.json'),
            'preserved_low_level_source_hashes': {n: old['source_hashes'][n] for n in preserved},
            'declared_source_overlays': overlays,
            'selection': 'Canonical seed22 fixed final endpoint only;all36TRAINand24knownvalidation conditions. No best checkpoint selection.',
            'purpose': 'Weak14/24different-road validation motivates training on broader geometry/static/walker states instead of treating the four-road course as a full-task baseline.',
            'limits': 'One selected development seed already passes the basic gate. This study does not waive the five-independent-policy requirement or claim a matched H/B1budget. No final conditions or validation labels used for training.'})
    summary, raw = read(parent / 'B1/summary.json'), read(parent / 'B1/endpoint_raw_audit.json')
    p['recovery']['B1'].update(parent_batch=parent.name, parent_checkpoint='B1/' + origin.name,
        lineage_controls=summary['lineage_control_steps'],
        consumed_controls_lower_bound=summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'],
        retained_optimizer_steps=old['recovery']['B1']['retained_optimizer_steps'] + summary['optimizer_steps'],
        model_sha256=metadata['model_sha256'], normalization_sha256=metadata['normalization_sha256'])
    name = 'scripts/prepare_native_broader_b1_course.py'
    shutil.copy2(__file__, batch / 'source' / name)
    p['source_hashes'][name] = sha(batch / 'source' / name)
    now = time.time()
    p.update(created_epoch=now, deadline_epoch=now + 7200)
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch / 'protocol.json')})
    output.mkdir(parents=True)
    atomic_json(output / 'registration.json', {'batch': batch.name, 'protocol_sha256': sha(batch / 'protocol.json'),
        'dependency_execution': 'B1_shared_parking_five_seed_execution_20261101', 'status': 'PREPARED_NOT_STARTED',
        'requested_new_PPO_controls': 20480, 'new_controls_so_far': 0})
    print(batch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)

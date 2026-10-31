"""Initialize actual B2 PPO from the matched pristine H learner,not trained H."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.audit_native_ppo_configuration import verify as verify_learner
from scripts.native_road_learning import training_config
from scripts.prepare_native_planner_ablation import planner_replacement_record
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_supervised_ppo import copy_walker_spawn_harness


def training_protocol(original, metadata, record, parent_name, controls, wall_seconds):
    if (type(controls) is not int or controls < 1280 or controls > 204800
            or type(wall_seconds) is not int or not 1800 <= wall_seconds <= 14400):
        raise ValueError('Require bounded integer training and wall budgets')
    if (original.get('evaluation_only') or 'H' not in original.get('methods', [])
            or original.get('jobs') or original.get('own_success_replay')
            or not original.get('extra_supervision')
            or not original.get('freeze_observation_statistics')):
        raise ValueError('Require the canonical frozen-statistics H training recipe')
    if (any(metadata[k] != 0 for k in ('physics_steps', 'optimizer_steps', 'lineage_control_steps'))
            or original['recovery']['H']['lineage_controls'] != 0
            or original['recovery']['H'].get('retained_optimizer_steps', 0) != 0):
        raise ValueError('B2 training requires pristine initialization,not a trained H continuation')
    reference_budget = original.get('method_budgets', {}).get('H', original['controlled_budget_per_method'])
    if controls != reference_budget:
        raise ValueError('B2 physical budget must match its declared H reference')
    if record['env']['method'] != 'B2' or record['env']['planner']['grid_postprocess'] != 'body_spline_v1':
        raise ValueError('Require the declared body-feasible Grid A* comparator')
    protocol = deepcopy(original)
    for key in ('supervised_representation_gates', 'supervised_ppo_intervention',
                'initialization_gate_profile', 'geometry_initialization', 'planner_ablation'):
        protocol.pop(key, None)
    protocol.update(methods=['B2'], evaluation_only=False,
                    controlled_budget_per_method=controls, method_budgets={'B2': controls},
                    grid_postprocess='body_spline_v1', parent_batch=parent_name,
                    recovery={'B2': {**deepcopy(original['recovery']['H']),
                                     'checkpoint': 'resume/B2', 'parent_batch': parent_name,
                                     'parent_checkpoint': 'resume/H'}},
                    scope='Actual B2 PPO from the exact matched zero-PPO H initializer;Grid A* body-spline planning with unchanged observations,reward,MPC and physical task. Development experiment,not final confirmation.')
    protocol['resource_limits']['wall_seconds'] = wall_seconds
    protocol['resource_limits']['automatic_retries'] = 0
    return protocol


def prepare(parent, controls=20480, wall_seconds=7200, reference_checkpoint=None):
    parent = Path(parent).resolve()
    original = read(parent / 'protocol.json')
    origin = parent / 'resume/H'
    metadata = verified_checkpoint_metadata(origin)
    record = planner_replacement_record(read(origin / 'config.json'), 'body_spline_v1')
    record['scope'] = 'Same pristine H actor before any PPO;new B2 planner-specific PPO training follows'
    protocol = training_protocol(original, metadata, record, parent.name, controls, wall_seconds)
    reference_checkpoint = (Path(reference_checkpoint).resolve() if reference_checkpoint is not None
                            else ROOT / 'runs/20261031T025409165666Z_native_supervised_ppo/resume/H')
    profile = verify_learner(origin, reference_checkpoint)
    for name, digest in original['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Parent frozen source changed: ' + name)
    if sha(parent / 'calibration/calibration.json') != original['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    dataset = parent / 'measured_demonstrations.npz'
    if sha(dataset) != original['extra_supervision']['dataset_sha256']:
        raise ValueError('Shared teacher dataset changed')
    generated = training_config(protocol, 'B2', EnvConfig.from_dict(record['env']).vehicle)
    if generated.to_dict() != record['env']:
        raise ValueError('Runtime B2 settings differ from the registered planner replacement')
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError(str(reason))
    with ProcessLease(parent / 'fresh_B2_training_prepare.lock'):
        registration = parent / 'fresh_B2_body_spline_training_registration.json'
        if registration.exists():
            raise FileExistsError('Preserve the already registered B2 trial')
        batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_planner_training')
        batch.mkdir()
        (batch / 'infrastructure').mkdir()
        shutil.copytree(parent / 'source', batch / 'source', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent / 'calibration', batch / 'calibration')
        shutil.copy2(dataset, batch / 'measured_demonstrations.npz')
        destination = batch / 'resume/B2'
        destination.parent.mkdir()
        shutil.copytree(origin, destination)
        write_json(destination / 'config.json', record)
        for name in ('model.zip', 'normalization.pkl', 'resume_state.pkl'):
            if sha(origin / name) != sha(destination / name):
                raise ValueError('Pristine learning bytes changed: ' + name)
        verified_checkpoint_metadata(destination)
        verify_learner(destination, origin)
        bridge = 'scripts/carla_bridge37.py'
        protocol['walker_spawn_initialization'] = copy_walker_spawn_harness(
            parent / 'source' / bridge, ROOT / bridge, batch / 'source' / bridge)
        shutil.copy2(Path(__file__), batch / 'source/scripts' / Path(__file__).name)
        write_json(batch / 'parent_protocol.json', original)
        protocol['planner_training_initialization'] = {
            'version': 'fresh_B2_body_spline_v1',
            'reference_H_batch': parent.name, 'reference_H_checkpoint': 'resume/H',
            'reference_protocol_sha256': sha(parent / 'protocol.json'),
            'model_sha256': sha(destination / 'model.zip'),
            'normalization_sha256': sha(destination / 'normalization.pkl'),
            'resume_state_sha256': sha(destination / 'resume_state.pkl'),
            'complete_saved_learner_verification': profile,
            'learner_reference_model_sha256': sha(reference_checkpoint / 'model.zip'),
            'initial_weights_normalization_rng_byte_identical': True,
            'environment_changes': {'method': ['H', 'B2'],
                                    'planner.grid_postprocess': ['legacy', 'body_spline_v1']},
            'new_supervised_updates': 0, 'PPO_updates_before_start': 0,
            'limits': 'Shared H teacher priors and previous planner development costs remain disclosed;this does not equalize total historical tuning effort. New task-specific optimizer updates must actually run before calling B2 trained.',
        }
        now = time.time()
        protocol.update(created_epoch=now, deadline_epoch=now + wall_seconds)
        protocol['source_hashes'] = {p.relative_to(batch / 'source').as_posix(): sha(p)
                                     for p in (batch / 'source').rglob('*')
                                     if p.is_file() and '__pycache__' not in p.parts}
        write_json(batch / 'protocol.json', protocol)
        prepared = {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch / 'protocol.json')}
        write_json(batch / 'prepared.json', prepared)
        write_json(registration, {'batch': batch.name, **prepared})
        print(batch, flush=True)
        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--controls', type=int, default=20480)
    parser.add_argument('--wall-seconds', type=int, default=7200)
    parser.add_argument('--reference-checkpoint', type=Path,
                        help='Explicit canonical pristine H checkpoint;allows public capsules outside the original run layout')
    args = parser.parse_args()
    prepare(args.parent, args.controls, args.wall_seconds, args.reference_checkpoint)

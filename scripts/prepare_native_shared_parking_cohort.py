"""Stage all five fixed-budget actual PPO courses from the registered independent actor fits."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig, VehicleConfig
from navstack.experiments import signature
from navstack.independent_clip_ppo import load_ppo
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare(plan_path, output):
    import numpy as np
    import torch
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = read(plan_path)
    if output.exists():
        raise FileExistsError('Preserve the prepared five-seed cohort')
    if [r['seed'] for r in plan['parents']] != [11, 22, 33, 44, 55] or plan['requested_new_PPO_controls_each'] != 10240:
        raise ValueError('Require all five predeclared seeds and the fixed common control budget')
    reference = ROOT / 'runs/20261101T112758733226Z_native_B1_shared_parking_PPO'
    clearance = ROOT / 'runs/20261101T121222965681Z_native_B1_validation_remainder'
    require_finished_parent(reference)
    require_finished_parent(clearance)
    rp, cp = read(reference / 'protocol.json'), read(clearance / 'protocol.json')
    for parent, protocol in ((reference, rp), (clearance, cp)):
        for name, digest in protocol['source_hashes'].items():
            if sha(parent / 'source' / name) != digest:
                raise ValueError('Verified common source changed')
    if (sha(reference / 'calibration/calibration.json') != sha(clearance / 'calibration/calibration.json')
            or read(reference / 'calibration/calibration.json')['recommended_adapter']['control_delivery'] != 'batch_sync_v1'):
        raise ValueError('Common calibrated delivery differs')
    torch.set_num_threads(1)
    inputs, hashes, moments = [], set(), None
    for row in plan['parents']:
        fit = ROOT / f"runs/B1_shared_parking_cohort_fit_seed{row['seed']}_20261101"
        result = read(fit / 'complete.json')
        metadata = verified_checkpoint_metadata(fit / 'checkpoint')
        if (result['plan_sha256'] != sha(plan_path) or result['seed'] != row['seed']
                or result['new_auxiliary_Adam_steps'] != 1024
                or not result['PPO_critic_logstd_optimizer_counters_RMS_retained']
                or result['source_model_sha256'] != row['model_sha256']
                or metadata['model_sha256'] != result['model_sha256']):
            raise ValueError('Incomplete or mismatched independent fit')
        if result['actor_after_sha256'] in hashes:
            raise ValueError('A trained actor was duplicated across seeds')
        hashes.add(result['actor_after_sha256'])
        model = load_ppo(fit / 'checkpoint/model.zip', device='cpu')
        if (model.n_steps, model.batch_size, model.n_epochs) != (1280, 320, 10):
            raise ValueError('Unequal PPO collection/update recipe')
        if any(group['lr'] != 3e-6 for group in model.policy.optimizer.param_groups):
            raise ValueError('Unequal retained PPO learning rate')
        with (fit / 'checkpoint/normalization.pkl').open('rb') as stream:
            norm = pickle.load(stream)
        current = (norm.obs_rms.mean, norm.obs_rms.var, norm.obs_rms.count)
        if moments is None:
            moments = tuple(np.copy(v) for v in current)
        for a, b in zip(current, moments):
            np.testing.assert_array_equal(a, b)
        inputs.append((row, fit, result, metadata))
    output.mkdir(parents=True)
    shutil.copy2(plan_path, output / 'registered_plan.json')
    batches = []
    for row, fit, result, metadata in inputs:
        okay, reason = resources_ok(ROOT)
        if not okay:
            raise RuntimeError(str(reason))
        parent = ROOT / row['batch']
        require_finished_parent(parent)
        p = deepcopy(read(parent / 'protocol.json'))
        if p.get('own_success_replay') or p.get('jobs') or p.get('native_validation'):
            raise ValueError('Unexpected active auxiliary learning or case distribution')
        p.update(seed=row['seed'], methods=['B1'], evaluation_only=False,
            goal_parking_assist=True, freeze_observation_statistics=True,
            controlled_budget_per_method=10240, method_budgets={'B1': 10240},
            deterministic_training_route_evaluation=True,
            checkpoint_thresholds=list(range(1280, 10241, 1280)),
            scope='One of five registered independent actual-PPO basic courses with common TRAIN-only actor initialization and fixed shared parking/delivery. No full-task or final confirmation claim.',
            resource_limits={**p['resource_limits'], 'wall_seconds': 7200, 'automatic_retries': 0},
            shared_parking_PPO={'version': plan['version'], 'source_model_sha256': metadata['model_sha256'],
                'normalization_sha256': metadata['normalization_sha256'], 'new_PPO_controls': 10240,
                'new_supervised_updates': 0, 'prior_auxiliary_Adam_steps': 1024,
                'registered_plan_sha256': sha(plan_path), 'fit_report_sha256': sha(fit / 'complete.json'),
                'offline_screen_passed_before_learning': result['offline_screen_passed'],
                'selection': 'All five seeds;fixed final10240-control endpoint and all four basic roads. No best checkpoint or seed selection.',
                'limits': 'Independent old20480-control parents plus common1024TRAIN-only actor Adam steps. This is a new learning course;offline diagnostic screens are retained and qualification is measured only after actual PPO. Not fromscratch,full-task,added-PPO or matched H/B1 evidence.'})
        vehicle = VehicleConfig(**read(reference / 'calibration/calibration.json')['vehicle'])
        cfg = training_config(p, 'B1', vehicle)
        saved = read(fit / 'checkpoint/config.json')
        if signature(EnvConfig.from_dict({**cfg.to_dict(), 'goal_parking_assist': False})) != saved['signature']:
            raise ValueError('An unrelated policy/environment configuration changed')
        batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_B1_shared_parking_cohort_seed' + str(row['seed']))
        batch.mkdir()
        (batch / 'infrastructure').mkdir()
        shutil.copytree(reference / 'source', batch / 'source', ignore=shutil.ignore_patterns('__pycache__'))
        bridge = 'scripts/carla_bridge37.py'
        shutil.copy2(clearance / 'source' / bridge, batch / 'source' / bridge)
        shutil.copytree(reference / 'calibration', batch / 'calibration')
        shutil.copytree(fit / 'checkpoint', batch / 'resume/B1')
        shutil.copy2(batch / 'resume/B1/config.json', batch / 'resume/B1/config_before_shared_wrapper.json')
        atomic_json(batch / 'resume/B1/config.json', {**saved, 'env': cfg.to_dict(), 'signature': signature(cfg),
            'explicit_wrapper_transition': 'Shared parking v2 fixed throughout new PPO;weights,optimizer,RMS and RNG bytes retained.'})
        shutil.copytree(fit, batch / 'fit_evidence', ignore=shutil.ignore_patterns('checkpoint'))
        shutil.copy2(ROOT / plan['dataset'], batch / 'fit_evidence/fit_inputs.npz')
        origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
        shutil.copytree(origin, batch / 'fit_evidence/initial_checkpoint')
        if (parent / 'measured_demonstrations.npz').exists():
            shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
        summary, raw = read(parent / 'B1/summary.json'), read(parent / 'B1/endpoint_raw_audit.json')
        previous = p['recovery']['B1']
        p['recovery'] = {'B1': {'checkpoint': 'resume/B1', 'parent_batch': parent.name,
            'parent_checkpoint': 'B1/' + origin.name, 'lineage_controls': summary['lineage_control_steps'],
            'consumed_controls_lower_bound': summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'],
            'retained_optimizer_steps': previous.get('retained_optimizer_steps', 0) + summary['optimizer_steps'],
            'model_sha256': metadata['model_sha256'], 'normalization_sha256': metadata['normalization_sha256']}}
        name = 'scripts/prepare_native_shared_parking_cohort.py'
        shutil.copy2(__file__, batch / 'source' / name)
        p['source_hashes'] = {**rp['source_hashes'], bridge: sha(batch / 'source' / bridge), name: sha(batch / 'source' / name)}
        p['calibration_sha256'] = sha(batch / 'calibration/calibration.json')
        p['shared_parking_PPO']['walker_initialization_version'] = 'native_walker_clearance_020_v1'
        now = time.time()
        p.update(created_epoch=now, deadline_epoch=now + 7200)
        atomic_json(batch / 'protocol.json', p)
        atomic_json(batch / 'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch / 'protocol.json')})
        batches.append({'seed': row['seed'], 'batch': batch.name, 'protocol_sha256': sha(batch / 'protocol.json'),
                        'initial_model_sha256': metadata['model_sha256'], 'offline_screen_passed': result['offline_screen_passed']})
        atomic_json(output / 'status.json', {'phase': 'PREPARING_FIVE_SHARED_WRAPPER_COURSES', 'batches': batches})
    source_sets = [read(ROOT / 'runs' / row['batch'] / 'protocol.json')['source_hashes'] for row in batches]
    if any(source != source_sets[0] for source in source_sets):
        raise ValueError('The frozen worker source must be identical across all five seeds')
    atomic_json(output / 'status.json', {'phase': 'ALL_FIVE_PREPARED_NOT_DRIVEN', 'batches': batches,
        'registered_plan_sha256': sha(plan_path), 'independent_actors': len(hashes), 'new_controls': 0})
    return batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.plan, args.output)

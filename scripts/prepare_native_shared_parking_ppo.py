"""Register one fixed-budget PPO retention course with shared parking fixed throughout."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare():
    pointer = ROOT / 'runs/native_B1_shared_parking_PPO_registration_20261101.json'
    if pointer.exists():
        raise FileExistsError('Preserve the registered course; no automatic repetition')
    parent = ROOT / 'runs/20261101T104233573082Z_native_shared_parking_v2_B1'
    require_finished_parent(parent)
    old = read(parent / 'protocol.json')
    evaluation = read(parent / 'B1/evaluation.json')
    raw = read(parent / 'B1/endpoint_raw_audit.json')
    summary = read(parent / 'B1/summary.json')
    origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    if (old['methods'] != ['B1'] or not old['goal_parking_assist']
            or metadata['model_sha256'] != '72d465ab4e553c189fd04c5e472a5fcc7cc0eafa371ee7c5c62f6eb9e0ab5844'
            or len(evaluation['episodes']) != 4 or not all(e['success'] for e in evaluation['episodes'])
            or raw['protocol_sha256'] != sha(parent / 'protocol.json')):
        raise ValueError('Require the complete audited shared-wrapper development transfer')
    calibration = read(parent / 'calibration/calibration.json')
    if calibration['recommended_adapter']['control_delivery'] != 'batch_sync_v1':
        raise ValueError('Synchronous low-level delivery is required')
    for name, digest in old['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Frozen parent source changed: ' + name)
    okay, reason = resources_ok(ROOT)
    if not okay:
        raise RuntimeError(str(reason))
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_B1_shared_parking_PPO')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    shutil.copytree(parent / 'source', batch / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent / 'calibration', batch / 'calibration')
    shutil.copytree(origin, batch / 'resume/B1')
    if (parent / 'measured_demonstrations.npz').exists():
        shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    if verified_checkpoint_metadata(batch / 'resume/B1') != metadata:
        raise ValueError('Copied checkpoint differs')
    now = time.time()
    p = deepcopy(old)
    transfer = p.pop('shared_parking_transfer')
    p.update(evaluation_only=False, controlled_budget_per_method=10240,
             method_budgets={'B1': 10240}, checkpoint_thresholds=list(range(1280, 10241, 1280)),
             deterministic_training_route_evaluation=True, freeze_observation_statistics=True,
             created_epoch=now, deadline_epoch=now + 7200,
             resource_limits={**old['resource_limits'], 'wall_seconds': 7200, 'automatic_retries': 0},
             scope='One warm-started real PPO development retention course with shared parking v2 fixed during all training and evaluation. Not five independent seeds or final confirmation.',
             shared_parking_PPO={
                 'version': 'fixed_shared_parking_PPO10240_v1', 'source_model_sha256': metadata['model_sha256'],
                 'normalization_sha256': metadata['normalization_sha256'], 'new_PPO_controls': 10240,
                 'new_supervised_updates': 0, 'prior_transfer': transfer,
                 'low_level_fixed': 'batch_sync_v1 and inside_goal_pose_overspeed_brake_v2; no runtime source, gain, goal or reward changes',
                 'initialization': 'Retain every trained weight, log_std, PPO Adam state, observation moment and RNG byte. No additional labels or auxiliary fitting.',
                 'selection': 'Only the final 10240-control endpoint; all four original TRAIN roads, no best-checkpoint selection.',
                 'basic_gate': 'At least 3/4 successes AND zero collision/bounds; no automatic full-task promotion.',
                 'failure_rule': 'Retain all failures; no automatic retry or unregistered additional budget.',
                 'limits': 'Single seed22, supervised warm start plus inherited 30720 PPO controls. Does not establish independent five-seed learning, added PPO value or hierarchy advantage.'})
    p['recovery'] = {'B1': {
        **old['recovery']['B1'], 'checkpoint': 'resume/B1', 'parent_batch': parent.name,
        'parent_checkpoint': 'B1/' + origin.name, 'lineage_controls': summary['lineage_control_steps'],
        'consumed_controls_lower_bound': summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'],
        'retained_optimizer_steps': old['recovery']['B1']['retained_optimizer_steps'] + summary['optimizer_steps']}}
    name = 'scripts/prepare_native_shared_parking_ppo.py'
    shutil.copy2(__file__, batch / 'source' / name)
    p['source_hashes'][name] = sha(batch / 'source' / name)
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch / 'protocol.json')})
    atomic_json(pointer, {'batch': batch.name, 'protocol_sha256': sha(batch / 'protocol.json'), 'new_controls_so_far': 0})
    print(batch, flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()

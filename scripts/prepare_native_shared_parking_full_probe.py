"""Probe all twelve known development conditions after one actual trained basic gate."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.native_cases import registered_native_cases
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare():
    pointer = ROOT / 'runs/native_B1_shared_parking_full_probe_registration_20261101.json'
    if pointer.exists():
        raise FileExistsError('Preserve the single registered full-condition development probe')
    parent = ROOT / 'runs/20261101T112758733226Z_native_B1_shared_parking_PPO'
    require_finished_parent(parent)
    result_path = ROOT / 'reports/native_B1_shared_parking_PPO_20261101.json'
    result = read(result_path)
    old = read(parent / 'protocol.json')
    if (result['batch'] != parent.name or result['protocol_sha256'] != sha(parent / 'protocol.json')
            or result['new_actual_PPO_controls'] != 10240 or not result['endpoint']['basic_gate_passed']
            or not result['all_engine_controls_acknowledged']):
        raise ValueError('Require the actual complete audited basic learning gate')
    origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    if metadata['model_sha256'] != result['final_model_sha256']:
        raise ValueError('Require the predeclared final trained endpoint')
    for name, digest in old['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Parent frozen source changed')
    p = deepcopy(old)
    p['native_case_manifest'] = deepcopy(old['basic_motion_learning']['original_full_case_manifest'])
    cases = registered_native_cases(p)
    if len(cases) != 12:
        raise ValueError('All twelve original development conditions are required')
    okay, reason = resources_ok(ROOT)
    if not okay:
        raise RuntimeError(str(reason))
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_B1_shared_parking_full_probe')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    shutil.copytree(parent / 'source', batch / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent / 'calibration', batch / 'calibration')
    shutil.copytree(origin, batch / 'resume/B1')
    if (parent / 'measured_demonstrations.npz').exists():
        shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    now = time.time()
    p.pop('freeze_observation_statistics', None)
    prior_course = p.pop('shared_parking_PPO')
    p.update(evaluation_only=True, controlled_budget_per_method=0, method_budgets={'B1': 0},
             checkpoint_thresholds=[], created_epoch=now, deadline_epoch=now + 3600,
             resource_limits={**old['resource_limits'], 'wall_seconds': 3600, 'automatic_retries': 0},
             scope='All12original KNOWN development conditions after a single shared-wrapper trained basic gate. Frozen weights, zero new learning; not a five-seed full-task baseline or new final confirmation.',
             qualified_basic_full_probe={
                 'version': 'all12known_conditions_after_shared_parking_PPO_v1',
                 'basic_report_sha256': sha(result_path), 'source_model_sha256': metadata['model_sha256'],
                 'source_normalization_sha256': metadata['normalization_sha256'],
                 'inherited_course': prior_course['version'], 'new_PPO_updates': 0,
                 'selection': 'Every original four-road curve/static/walker condition; fixed final trained model only.',
                 'limits': 'Diagnostic evaluation only; does not waive the five independent trained-basic gates required for formal full-task comparison. No full-task training or final set is activated.'})
    summary = read(parent / 'B1/summary.json')
    raw = read(parent / 'B1/endpoint_raw_audit.json')
    p['recovery'] = {'B1': {
        **old['recovery']['B1'], 'checkpoint': 'resume/B1', 'parent_batch': parent.name,
        'parent_checkpoint': 'B1/' + origin.name, 'lineage_controls': summary['lineage_control_steps'],
        'consumed_controls_lower_bound': summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'],
        'retained_optimizer_steps': old['recovery']['B1']['retained_optimizer_steps'] + summary['optimizer_steps'],
        'model_sha256': metadata['model_sha256'], 'normalization_sha256': metadata['normalization_sha256']}}
    name = 'scripts/prepare_native_shared_parking_full_probe.py'
    shutil.copy2(__file__, batch / 'source' / name)
    p['source_hashes'][name] = sha(batch / 'source' / name)
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch / 'protocol.json')})
    atomic_json(pointer, {'batch': batch.name, 'protocol_sha256': sha(batch / 'protocol.json'), 'new_controls_so_far': 0})
    print(batch, flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()

"""Freeze a bounded new-TRAIN-road native recovery collection, without fitting."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
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
from scripts.collect_native_recovery import VERSION


def patch_worker(text):
    marker = '                if protocol.get("deterministic_training_route_evaluation",False):'
    if text.count(marker) != 1 or 'native_recovery_collection' in text:
        raise ValueError('Require exactly one unchanged parent evaluation dispatch')
    replacement = '''                if protocol.get('native_recovery_collection') is not None:
                    from scripts.collect_native_recovery import collect
                    result.update(collect(config,client,cases,loaded,vec,output,protocol,
                        RecordedRoadEnv,lambda:guard(batch,protocol)))
                    write_json(output/"summary.json", result)
                elif protocol.get("deterministic_training_route_evaluation",False):'''
    return text.replace(marker, replacement)


def prepare():
    parent = ROOT/'runs/20261101T011350364179Z_native_B1_basic_lr_extension'
    origin = ROOT/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay'
    teacher = ROOT/'runs/20261031T025409165666Z_native_supervised_ppo/H/config.json'
    inventory = ROOT/'experiments/native_development_train36_validation24_v1.json'
    pointer = ROOT/'runs/native_executed_teacher_mixture_registration_20261101.json'
    if pointer.exists():
        raise FileExistsError('Preserve registered collection; do not repeat its controls')
    require_finished_parent(parent)
    old = read(parent/'protocol.json'); metadata = verified_checkpoint_metadata(origin)
    summary = read(parent/'B1/summary.json'); raw = read(parent/'B1/endpoint_raw_audit.json')
    if (old['methods'] != ['B1'] or metadata['model_sha256'] != '4f20570844ff17da318c601893be8523ddf504193d21775f9553a3d5c8530ca0'
            or metadata['normalization_sha256'] != 'db467fb35b0f78a989e8e042b0400036cada30cfad36b41fba727a4f432e5df5'
            or metadata['lineage_control_steps'] != 20480 or metadata.get('additional_actor_fit_steps') != 512):
        raise ValueError('Require the already evaluated fixed replay endpoint and exact training lineage')
    for name, digest in old['source_hashes'].items():
        if sha(parent/'source'/name) != digest: raise ValueError('Parent frozen source changed: '+name)
    layout = read(inventory); indices = [4, 5, 6, 7]
    routes = [layout['routes'][i] for i in indices]
    ids = [r['geometry_id'] for r in routes]
    excluded = {r['geometry_id'] for r in old['routes']+layout['native_validation']['routes']}
    if (len(set(ids)) != 4 or set(ids) & excluded
            or [r['geometry_id'] for r in layout['routes'][:4]] != [r['geometry_id'] for r in old['routes']]
            or not all({'route_index': i, 'hazard': None} in layout['native_case_manifest']['cases'] for i in indices)):
        raise ValueError('Collection roads must be original TRAIN inventory roads, disjoint from original four and validation')
    okay, reason = resources_ok(ROOT)
    if not okay: raise RuntimeError(str(reason))
    batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_TRAIN_recovery_collection')
    batch.mkdir(); (batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source', batch/'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration', batch/'calibration')
    shutil.copytree(origin, batch/'resume/B1')
    shutil.copy2(parent/'measured_demonstrations.npz', batch/'measured_demonstrations.npz')
    (batch/'collection_inputs').mkdir()
    shutil.copy2(teacher, batch/'collection_inputs/teacher_config.json')
    shutil.copy2(inventory, batch/'collection_inputs/TRAIN_inventory.json')
    shutil.copy2(ROOT/'scripts/prepare_native_recovery_collection.py', batch/'collection_inputs/executed_preparation.py')
    worker = batch/'source/scripts/native_road_learning.py'
    # Preserve every other parent source byte, including the physical plant.
    original = worker.read_bytes()
    newline = '\r\n' if b'\r\n' in original else '\n'
    modified = patch_worker(original.decode('utf-8').replace('\r\n', '\n'))
    worker.write_bytes(modified.replace('\n', newline).encode('utf-8'))
    changed = ['scripts/native_road_learning.py']
    for name in ('collect_native_recovery.py', 'query_native_persistent_teacher.py'):
        shutil.copy2(ROOT/'scripts'/name, batch/'source/scripts'/name); changed.append('scripts/'+name)
    p = deepcopy(old)
    for name in ('jobs', 'freeze_observation_statistics', 'own_success_replay', 'native_validation', 'native_camera_evaluation'):
        p.pop(name, None)
    now = time.time()
    p.update(scope='TRAIN-only recovery collection on4new TRAIN roads; fixed teacher/learner mixture,zero optimization. Mixture outcomes are not B1 performance.',
        routes=routes, native_case_manifest={'version': 'native_development_cases_v1', 'split': 'development',
            'cases': [{'route_index': i, 'hazard': None} for i in range(4)]},
        evaluation_only=True, deterministic_training_route_evaluation=False,
        controlled_budget_per_method=0, method_budgets={'B1': 0}, checkpoint_thresholds=[],
        created_epoch=now, deadline_epoch=now+2400, parent_batch=parent.name,
        resource_limits={**old['resource_limits'], 'wall_seconds': 2400, 'automatic_retries': 0},
        native_recovery_collection={'version': VERSION, 'teacher_probability': .5, 'block_controls': 10,
            'max_controls': 3600, 'mixture_seed': 20261101, 'reset_seed': 8401,
            'original_TRAIN_route_indices': indices, 'geometry_ids': ids,
            'TRAIN_inventory': 'collection_inputs/TRAIN_inventory.json', 'TRAIN_inventory_sha256': sha(inventory),
            'teacher_config': 'collection_inputs/teacher_config.json', 'teacher_config_sha256': sha(teacher),
            'source_model_sha256': metadata['model_sha256'], 'source_normalization_sha256': metadata['normalization_sha256'],
            'source_checkpoint': str(origin.relative_to(ROOT)).replace('\\','/'),
            'new_PPO_updates': 0, 'new_supervised_updates': 0,
            'fit_labels': 'Only valid ACTUALLY EXECUTED teacher controls; learner-step teacher proposals excluded by explicit mask.',
            'outcomes': 'All four episodes and every transition retained,including collision,bounds,deadline;no retries or endpoint selection.',
            'source_overlay': changed, 'parent_worker_sha256': sha(parent/'source/scripts/native_road_learning.py')},
        recovery={'B1': {**old['recovery']['B1'], 'checkpoint': 'resume/B1', 'parent_batch': parent.name,
            'parent_checkpoint': 'B1/'+read(parent/'B1/latest_checkpoint.json')['folder'],
            'lineage_controls': summary['lineage_control_steps'],
            'consumed_controls_lower_bound': summary['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
            'retained_optimizer_steps': old['recovery']['B1']['retained_optimizer_steps']+summary['optimizer_steps']}})
    p['extra_supervision']['supervised_optimizer_steps'] += 512
    p['extra_supervision']['collection_actor_replay'] = {'steps': 512, 'model_sha256': metadata['model_sha256'], 'new_this_collection': 0}
    p['source_hashes'] = deepcopy(old['source_hashes'])
    for name in changed: p['source_hashes'][name] = sha(batch/'source'/name)
    atomic_json(batch/'protocol.json', p)
    atomic_json(batch/'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch/'protocol.json')})
    atomic_json(pointer, {'batch': batch.name, 'protocol_sha256': sha(batch/'protocol.json'), 'controls_so_far': 0})
    return batch


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(prepare(), flush=True)

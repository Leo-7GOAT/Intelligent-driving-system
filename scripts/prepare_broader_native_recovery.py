"""Freeze one bounded teacher/learner collection on every existing TRAIN condition."""
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
from navstack.native_cases import registered_native_cases
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.collect_native_recovery import VERSION, validate_collection
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare(parent, teacher, output):
    parent, teacher, output = map(lambda p: Path(p).resolve(), (parent, teacher, output))
    if output.exists() or (parent / 'broader_recovery_registration.json').exists():
        raise FileExistsError('Preserve the registered collection; never repeat its controls')
    require_finished_parent(parent)
    require_finished_parent(teacher)
    old, teacher_protocol = read(parent / 'protocol.json'), read(teacher / 'protocol.json')
    evaluation = read(parent / 'B1/evaluation.json')
    raw, summary = read(parent / 'B1/endpoint_raw_audit.json'), read(parent / 'B1/summary.json')
    inventory = ROOT / 'experiments/native_development_train36_validation24_v1.json'
    layout = read(inventory)
    origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    config = EnvConfig.from_dict(read(parent / 'B1/config.json')['env'])
    if (not old.get('measured_actor_validation_probe') or old['seed'] != 22
            or metadata['model_sha256'] != '5f42425596c4f1555f5f9870f2a18f3ccc4bda337e5fbe197e28a14e667456d3'
            or evaluation['model_sha256'] != metadata['model_sha256']
            or metadata['lineage_control_steps'] != 51200
            or len(evaluation['episodes']) != 24 or sum(e['success'] for e in evaluation['episodes']) != 4
            or raw['protocol_sha256'] != sha(parent / 'protocol.json')
            or old['routes'] != layout['routes'] or old['native_case_manifest'] != layout['native_case_manifest']
            or teacher_protocol['routes'] != old['routes']
            or teacher_protocol['native_case_manifest'] != old['native_case_manifest']
            or teacher_protocol['data_role'] != 'TRAIN_EXTRA_SUPERVISION'
            or sha(teacher / 'source_config.json') != teacher_protocol['source_config_sha256']):
        raise ValueError('Require the fixed failed development endpoint and exact completed TRAIN teacher')
    for batch, protocol in ((parent, old), (teacher, teacher_protocol)):
        for name, digest in protocol['source_hashes'].items():
            if sha(batch / 'source' / name) != digest:
                raise ValueError('Frozen source changed: ' + name)
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 18 * 1024**3:
        raise RuntimeError('Collection resource guard: ' + str(reason))
    p = deepcopy(old)
    for name in ('native_validation', 'measured_actor_validation_probe', 'measured_actor_basic_probe',
                 'freeze_observation_statistics', 'own_success_replay'):
        p.pop(name, None)
    now = time.time()
    p.update(data_role='TRAIN_EXTRA_SUPERVISION', evaluation_only=True,
        deterministic_training_route_evaluation=False, checkpoint_thresholds=[],
        controlled_budget_per_method=0, method_budgets={'B1': 0}, parent_batch=parent.name,
        scope='All36existingTRAINconditions;fixed50percent teacher per10controls;all outcomes retained. Extra supervision collection,not policy evaluation or PPO.',
        created_epoch=now, deadline_epoch=now + 3600,
        resource_limits={**old['resource_limits'], 'wall_seconds': 3600, 'automatic_retries': 0})
    cases = registered_native_cases(p)
    p['native_recovery_collection'] = dict(version=VERSION, route_profile='broader_TRAIN36_v1',
        teacher_probability=.5, block_controls=10, max_controls=32400, mixture_seed=20261102, reset_seed=9401,
        original_TRAIN_case_indices=list(range(36)), TRAIN_case_manifest=deepcopy(p['native_case_manifest']),
        geometry_ids=[c['native_route']['geometry_id'] for c in cases],
        TRAIN_inventory='collection_inputs/TRAIN_inventory.json', TRAIN_inventory_sha256=sha(inventory),
        teacher_config='collection_inputs/teacher_config.json', teacher_config_sha256=sha(teacher / 'source_config.json'),
        teacher_parent=teacher.name, teacher_parent_protocol_sha256=sha(teacher / 'protocol.json'),
        source_model_sha256=metadata['model_sha256'], source_normalization_sha256=metadata['normalization_sha256'],
        source_lineage_control_steps=51200, exclude_intervened_teacher_labels=True,
        new_PPO_updates=0, new_supervised_updates=0, validation_conditions_used=0, final_conditions_used=0,
        selection='Fixed canonical22,all36TRAINconditions,all outcomes;zero retries or endpoint selection.',
        fit_labels='Only valid actually selected teacher controls without physical helper intervention. Unselected teacher proposals are diagnostics only.',
        rationale='Nominal TRAIN teacher fit passed basic4/4 but failed knownVAL4/24. Test closed-loop recovery-state coverage;VALstates/labels are not training data.')
    validate_collection(p, config, cases)
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_broader_TRAIN_recovery')
    batch.mkdir(); (batch / 'infrastructure').mkdir(); (batch / 'collection_inputs').mkdir()
    for name in ('source', 'calibration'):
        shutil.copytree(parent / name, batch / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(origin, batch / 'resume/B1')
    shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    shutil.copy2(teacher / 'source_config.json', batch / 'collection_inputs/teacher_config.json')
    shutil.copy2(inventory, batch / 'collection_inputs/TRAIN_inventory.json')
    shutil.copy2(__file__, batch / 'collection_inputs/executed_preparation.py')
    overlays = ('collect_native_recovery.py', 'query_native_persistent_teacher.py', 'audit_native_recovery_collection.py')
    for name in overlays:
        shutil.copy2(ROOT / 'scripts' / name, batch / 'source/scripts' / name)
        p['source_hashes']['scripts/' + name] = sha(batch / 'source/scripts' / name)
    if "if protocol.get('native_recovery_collection') is not None:" not in (batch / 'source/scripts/native_road_learning.py').read_text(encoding='utf-8'):
        raise ValueError('Require the existing native collection dispatch; do not alter physical worker')
    p['native_recovery_collection']['source_overlay'] = ['scripts/' + n for n in overlays]
    p['recovery']['B1'].update(parent_batch=parent.name, parent_checkpoint='B1/' + origin.name,
        consumed_controls_lower_bound=summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'],
        model_sha256=metadata['model_sha256'], normalization_sha256=metadata['normalization_sha256'])
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(batch / 'protocol.json')))
    registration = dict(batch=batch.name, protocol_sha256=sha(batch / 'protocol.json'),
        source_model_sha256=metadata['model_sha256'], cases=36, data_role='TRAIN_EXTRA_SUPERVISION',
        new_controls_so_far=0, new_PPO_updates=0, new_supervised_updates=0,
        maximum_physical_controls=32400, maximum_wall_seconds=3600, automatic_retries=0)
    output.mkdir(); atomic_json(output / 'registration.json', registration)
    atomic_json(parent / 'broader_recovery_registration.json', registration)
    print(batch, flush=True)


def prepare_probe(fit, output):
    """Stage the single fixed recovery endpoint on the original basic TRAIN roads."""
    fit, output = Path(fit).resolve(), Path(output).resolve()
    if output.exists() or (fit / 'basic_probe_registration.json').exists():
        raise FileExistsError('Preserve the fixed recovery probe')
    result, registration = read(fit / 'complete.json'), read(fit / 'registration.json')
    recipe_path = ROOT / registration.get('recipe_file', 'protocols/native_B1_broader_executed_recovery_20261102.json')
    recipe = read(recipe_path)
    parent = ROOT / 'runs' / registration['collection']
    require_finished_parent(parent)
    old = read(parent / 'protocol.json')
    metadata = verified_checkpoint_metadata(fit / 'checkpoint')
    expected_steps=recipe['steps'];retained_steps=recipe.get('retained_actor_optimizer_steps',expected_steps)
    if (registration['recipe_sha256'] != sha(recipe_path) or registration['seed'] != 22
            or registration['steps'] != expected_steps or result['actual_Adam_step_each'] != [retained_steps]
            or result['auxiliary_parameter_states'] != 20
            or result['new_controls'] != 0 or result['new_PPO_updates'] != 0
            or result['model_sha256'] != metadata['model_sha256']
            or registration['collection_audit_sha256'] != sha(parent / 'B1/collection_raw_audit.json')
            or recipe['collection_batch'] != parent.name):
        raise ValueError('Require the one completed predeclared endpoint and actual collection audit')
    for name, digest in registration['source_hashes'].items():
        if sha(fit / name) != digest: raise ValueError('Executed actor fit source changed')
    for name, digest in old['source_hashes'].items():
        if sha(parent / 'source' / name) != digest: raise ValueError('Frozen physical source changed')
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 18 * 1024**3:
        raise RuntimeError('Probe resource guard: ' + str(reason))
    p = deepcopy(old)
    p.pop('native_recovery_collection'); p.pop('data_role')
    p.update(routes=deepcopy(old['routes'][:4]),
        native_case_manifest=dict(version='native_development_cases_v1', split='development',
            cases=[dict(route_index=i, hazard=None) for i in range(4)]),
        deterministic_training_route_evaluation=True,
        scope='Fixed canonical22after executed TRAINrecovery teaching;pure model.predict on original4basicTRAINroads. No teacher runtime,newPPOorfinalconditions.')
    now = time.time()
    p.update(created_epoch=now, deadline_epoch=now + 1800, parent_batch=parent.name,
        resource_limits={**old['resource_limits'], 'wall_seconds': 1800, 'automatic_retries': 0})
    p['recovery']['B1'].update(parent_batch=parent.name, parent_checkpoint='reviewed_recovery_actor/seed22',
        consumed_controls_lower_bound=old['recovery']['B1']['consumed_controls_lower_bound'] + read(parent / 'B1/collection.json')['collection_controls'],
        model_sha256=metadata['model_sha256'], normalization_sha256=metadata['normalization_sha256'])
    p['extra_supervision']['supervised_optimizer_steps'] += retained_steps
    p['measured_actor_basic_probe'] = dict(version='canonical22_executed_recovery_basic_v1',
        fit_complete_sha256=sha(fit / 'complete.json'), recipe_sha256=sha(recipe_path),
        additional_actor_Adam_steps=retained_steps, newly_performed_actor_steps=expected_steps,
        objective_version=registration['version'], cumulative_measured_actor_steps=1024+retained_steps,
        teacher_data_role='TRAIN_EXTRA_SUPERVISION', native_teacher_runtime=False,
        gate='At least3/4successANDzero collision/bounds. KnownVALonly after this gate;no automaticPPOpromotion.',
        new_PPO_updates=0, new_final_conditions=0)
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_recovery_actor_basic22')
    batch.mkdir(); (batch / 'infrastructure').mkdir()
    for name in ('source', 'calibration'):
        shutil.copytree(parent / name, batch / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(fit / 'checkpoint', batch / 'resume/B1')
    shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    evidence = batch / 'measured_actor_evidence'; evidence.mkdir()
    for name in ('registration.json', 'complete.json', 'auxiliary_Adam.pt', 'sampler_state.json', *registration['source_hashes']):
        if Path(name).name!=name:raise ValueError('Fit source evidence must be a local basename')
        shutil.copy2(fit / name, evidence / name)
    for name in ('collection_raw_audit.json', 'strict_delivery.json', 'collection.json'):
        shutil.copy2(parent / 'B1' / name, evidence / name)
    shutil.copy2(recipe_path, evidence / 'recipe.json')
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(batch / 'protocol.json')))
    output.mkdir()
    registered = dict(batch=batch.name, protocol_sha256=sha(batch / 'protocol.json'), fit=fit.relative_to(ROOT).as_posix(),
        source_model_sha256=metadata['model_sha256'], new_controls_so_far=0, new_PPO_updates=0, final_conditions=0)
    atomic_json(output / 'registration.json', registered); atomic_json(fit / 'basic_probe_registration.json', registered)
    print(batch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent'); parser.add_argument('--teacher')
    parser.add_argument('--probe-fit')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.probe_fit:
        if args.parent or args.teacher: parser.error('Probe and collection preparation are distinct')
        prepare_probe(args.probe_fit, args.output)
    else:
        if not args.parent or not args.teacher: parser.error('Collection requires parent and teacher')
        prepare(args.parent, args.teacher, args.output)

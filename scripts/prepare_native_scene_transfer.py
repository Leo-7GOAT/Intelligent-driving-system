"""Freeze a completed native PPO endpoint for scene or compute development probes.

The actor scene can change, or the explicit memo option can probe the original
scene with verified cache hooks only. These zero-update probes are not recovery,
independent training, or final confirmation. The reference B3 batch supplies scene
provenance only; its possibly older execution code is not adopted.
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
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import audit, read, sha
from scripts.audit_native_rule import audit as rule_audit
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import registered_native_cases
from scripts.prepare_native_evaluation_recovery import evaluation_lineage
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.audit_native_action_availability import verify_source


def transfer_protocol(old, reference, lineage, now, wall_seconds, failure_memo=False):
    if type(failure_memo) is not bool:
        raise ValueError('Failure memo requires explicit registration')
    if (type(wall_seconds) is not int or not 900 <= wall_seconds <= 3600
            or old.get('jobs') or old.get('curriculum')
            or old.get('scene_transfer') or old.get('evaluation_only')
            or not old.get('methods') or set(old['methods']) - {'H', 'H0', 'B1'}
            or len(set(old['methods'])) != len(old['methods'])
            or set(lineage) != set(old['methods'])):
        raise ValueError('Require a completed native PPO endpoint and bounded transfer window')
    if len(old.get('routes', [])) != 4 or old['routes'] != reference.get('routes'):
        raise ValueError('Reference must preserve all four original route geometries')
    same_scene = reference.get('native_hazard') == old.get('native_hazard')
    if same_scene and not failure_memo:
        raise ValueError('Reference does not change the scene family/specification')
    protocol = deepcopy(old)
    for key in ('own_success_replay', 'native_hazard'):
        protocol.pop(key, None)
    if reference.get('native_hazard') is not None:
        protocol['native_hazard'] = deepcopy(reference['native_hazard'])
    registered_native_cases(protocol)  # Validate every resolved spawn before staging.
    protocol.update(
        created_epoch=now, deadline_epoch=now + wall_seconds,
        evaluation_only=True, controlled_budget_per_method=0,
        method_budgets={method: 0 for method in old['methods']}, checkpoint_thresholds=[],
        recovery=deepcopy(lineage),
        scope='Frozen native-PPO scene-family transfer on four known development roads; '
              'zero new training or auxiliary updates, not independent confirmation',
        comparison='Reference B3 supplies audited scene specifications only; its older '
                   'execution code is not a matched architecture comparator. Parent '
                   'policy, normalizer, low level, reward and success gates remain fixed.',
        scene_transfer={'from': deepcopy(old.get('native_hazard')),
                        'to': deepcopy(reference.get('native_hazard')),
                        'training_updates_added': 0, 'routes_unchanged': True,
                        'split': 'development; known roads and previously inspected reference scenes'},
        resource_limits={**old['resource_limits'], 'wall_seconds': wall_seconds})
    if failure_memo:
        from navstack.planner_memo import VERSION
        protocol['planner_failure_memo'] = VERSION
        protocol['planner_compute_intervention'] = {
            'scope': 'Exact static deterministic failed-search memoization; zero new nodes on hits, '
                     'original search nodes separately recorded. No action/trajectory substitution.',
            'same_actor_scene_as_parent': same_scene}
        if same_scene:
            protocol['scope'] = 'Frozen native-PPO compute probe on the original known development scenes; zero new training'
    return protocol


def memo_source_overrides(parent):
    """Allow exactly the memo hook/metadata changes, never unrelated low-level edits."""
    changes = {
        'navstack/env.py': [(
            "        if getattr(plan, 'cache_hit', False):\n"
            "            detail.update(cache_hit=True, cached_expanded_nodes=plan.cached_expanded_nodes,\n"
            "                          cache_scope='exact_static_failure_v1; current expanded_nodes are zero')\n", '')],
        'scripts/native_road_learning.py': [(
            'from navstack.planner_memo import configure_planner_memo\n', ''), (
            'configure_planner_memo(RecordedRoadEnv(config, client, cases, output), protocol)',
            'RecordedRoadEnv(config, client, cases, output)'), (
            '                    configure_planner_memo(evaluation, protocol)\n', '')]}
    for name, replacements in changes.items():
        current = (ROOT / name).read_text(encoding='utf-8')
        for addition, original in replacements:
            if current.count(addition) != 1:
                raise ValueError('Memo hook source structure changed: ' + name)
            current = current.replace(addition, original)
        if current != (parent / 'source' / name).read_text(encoding='utf-8'):
            raise ValueError('Memo probe would change unrelated execution code: ' + name)
    protocol = read(parent / 'protocol.json')
    for name in ('navstack/planning.py', 'navstack/world.py', 'navstack/road_grid.py', 'navstack/subgoals.py'):
        verify_source(ROOT / name, parent / 'source' / name, protocol['source_hashes'][name])
    return [*changes, 'navstack/planner_memo.py']


def prepare(parent, reference, wall_seconds=1800, failure_memo=False):
    parent, reference = Path(parent).resolve(), Path(reference).resolve()
    require_finished_parent(parent)
    require_finished_parent(reference)
    old, target = read(parent / 'protocol.json'), read(reference / 'protocol.json')
    if rule_audit(reference) != read(reference / 'raw_audit.json'):
        raise ValueError('Reference rule raw audit changed')
    lineage, origins = {}, {}
    for method in old['methods']:
        if audit(parent, method) != read(parent / method / 'endpoint_raw_audit.json'):
            raise ValueError('Parent endpoint raw audit changed')
        summary = read(parent / method / 'summary.json')
        name = read(parent / method / 'latest_checkpoint.json')['folder']
        if Path(name).name != name:
            raise ValueError('Checkpoint must be a direct child')
        origin = parent / method / name
        metadata = verified_checkpoint_metadata(origin)
        if (not summary['reload_exact'] or summary['stopped_reason'] is not None
                or summary['optimizer_steps'] <= 0 or summary['physics_steps'] <= 0
                or any(metadata[k] != summary[k] for k in ('physics_steps', 'actions', 'optimizer_steps'))
                or metadata['model_sha256'] != sha(parent / method / 'model.zip')
                or metadata['normalization_sha256'] != sha(parent / method / 'normalization.pkl')
                or read(origin / 'config.json') != read(parent / method / 'config.json')):
            raise ValueError('Require the exact completed trained PPO endpoint')
        origins[method] = (origin, metadata)
        lineage[method] = evaluation_lineage(old.get('recovery', {}).get(method, {}),
                                           summary, parent.name, name, method)
    protocol = transfer_protocol(old, target, lineage, time.time(), wall_seconds, failure_memo)
    overrides = memo_source_overrides(parent) if failure_memo else []
    for name, digest in old['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Parent frozen source changed')
    if sha(parent / 'calibration/calibration.json') != old['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError(str(reason))
    registration = parent / ('scene_transfer_' + sha(reference / 'protocol.json')[:16]
                              + ('_memo' if failure_memo else '') + '.json')
    with ProcessLease(parent / 'scene_transfer_prepare.lock'):
        if registration.exists():
            raise FileExistsError('Preserve the already registered transfer probe')
        batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_native_scene_transfer')
        batch.mkdir()
        write_json(registration, {'batch': batch.name, 'status': 'registered_before_staging'})
        (batch / 'infrastructure').mkdir()
        shutil.copytree(parent / 'source', batch / 'source', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent / 'calibration', batch / 'calibration')
        shutil.copy2(__file__, batch / 'source/scripts' / Path(__file__).name)
        for name in overrides:
            shutil.copy2(ROOT / name, batch / 'source' / name)
        if overrides:
            protocol['harness_overrides'] = overrides
        for method, (origin, metadata) in origins.items():
            destination = batch / 'resume' / method
            shutil.copytree(origin, destination)
            if verified_checkpoint_metadata(destination) != metadata:
                raise ValueError('Copied frozen checkpoint changed')
        # Keep optional teacher provenance self-contained; it is never replayed.
        dataset = parent / 'measured_demonstrations.npz'
        if dataset.is_file():
            if sha(dataset) != old['extra_supervision']['dataset_sha256']:
                raise ValueError('Registered teacher dataset changed')
            shutil.copy2(dataset, batch / dataset.name)
        write_json(batch / 'parent_protocol.json', old)
        write_json(batch / 'reference_scene_protocol.json', target)
        protocol.update(parent_batch=parent.name,
                        parent_protocol_sha256=sha(parent / 'protocol.json'),
                        reference_scene_batch=reference.name,
                        reference_scene_audit_sha256=sha(reference / 'raw_audit.json'))
        protocol['source_hashes'] = {p.relative_to(batch / 'source').as_posix(): sha(p)
                                     for p in (batch / 'source').rglob('*')
                                     if p.is_file() and '__pycache__' not in p.parts}
        write_json(batch / 'protocol.json', protocol)
        write_json(batch / 'prepared.json', {'status': 'PREPARED_NOT_STARTED',
                                             'protocol_sha256': sha(batch / 'protocol.json')})
        write_json(registration, {'batch': batch.name, 'status': 'PREPARED_NOT_STARTED',
                                  'protocol_sha256': sha(batch / 'protocol.json')})
    print(batch, flush=True)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--reference-rule', required=True)
    parser.add_argument('--wall-seconds', type=int, default=1800)
    parser.add_argument('--failure-memo', action='store_true')
    args = parser.parse_args()
    prepare(args.parent, args.reference_rule, args.wall_seconds, args.failure_memo)

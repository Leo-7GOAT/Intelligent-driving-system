"""Paired frozen-policy deployment probes;add the same brake,retain native gates."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import signature, write_json
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import audit, read, sha
from scripts.prepare_native_evaluation_recovery import recorded_lifecycle_gone
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.prepare_native_supervised_ppo import copy_walker_spawn_harness

VERSION = 'additive_shared_brake_v1'


def require_closed_method_parent(parent):
    """A completed method may survive a failed sibling;never reuse a live batch."""
    final = read(parent / 'guard_final.json')
    if (type(final.get('exit_code')) is not int
            or read(parent / 'infrastructure/server_stop.json')['status'] != 'owned_empty_server_stopped'):
        raise ValueError('Require terminal guard and verified owned cleanup')
    lock = parent / 'run_registered_native.lock'
    if lock.exists():
        with lock.open('r+b'):
            pass
    # Public evidence archives omit operational lock files. Terminal cleanup and
    # independent worker/watchdog liveness verification remain mandatory below.
    identities = [read(parent / name)['pid'] for name in ('worker_process.json', 'guard.json')]
    if not recorded_lifecycle_gone(identities, final['epoch']):
        raise ValueError('Recorded parent worker/watchdog is still live or unverified')
    # Complete method counters,all evaluation cases,raw frames and saved learning
    # bytes are verified separately below. A sibling failure is copied explicitly.


def deployment_record(original, enabled):
    if type(enabled) is not bool:
        raise ValueError('Brake state must be an explicit boolean')
    cfg = EnvConfig.from_dict(original['env'])
    if cfg.shared_shield:
        raise ValueError('Parent must use the native safety profile')
    cfg.shared_shield = enabled
    return {**original, 'env': cfg.to_dict(), 'signature': signature(cfg),
            'scope': 'Frozen policy;paired deployment-only common brake,not retrained safety adaptation'}


def environment_overlay(original):
    before = '        if not self.config.shared_shield and not dynamic_safe(\n'
    if original.count(before) != 1:
        raise ValueError('Native trajectory safety gate structure differs')
    # Preserve the original native trajectory check in BOTH arms. The stock flag
    # otherwise disables it,which would confound addition of a common brake.
    return original.replace(before, '        if not dynamic_safe(\n')


def worker_overlay(original):
    tree = ast.parse(original)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'training_config']
    if len(functions) != 1:
        raise ValueError('Require one known native configuration function')
    function = functions[0]
    lines = original.splitlines(keepends=True)
    body = ''.join(lines[function.lineno-1:function.end_lineno])
    if body.count('    return config') != 1:
        raise ValueError('Unknown configuration return structure')
    addition = (
        "    profile=protocol.get('deployment_shared_brake')\n"
        "    if (not isinstance(profile,dict) or profile.get('version')!='additive_shared_brake_v1'\n"
        "            or type(profile.get('enabled')) is not bool or not protocol.get('evaluation_only')\n"
        "            or protocol.get('controlled_budget_per_method')!=0):\n"
        "        raise ValueError('Common brake is registered only for zero-update deployment probes')\n"
        "    config.shared_shield=profile['enabled']\n"
    )
    changed = ''.join(lines[:function.lineno-1]) + body.replace('    return config', addition + '    return config') + ''.join(lines[function.end_lineno:])
    # No reset,training,evaluation,actuation,termination or accounting edits.
    def without_config(text):
        value = ast.parse(text)
        value.body = [n for n in value.body if not isinstance(n, ast.FunctionDef) or n.name != 'training_config']
        return ast.dump(value, include_attributes=False)
    if without_config(original) != without_config(changed):
        raise ValueError('Deployment override changed another runtime function')
    return changed


def prepare(parent, method, wall_seconds=3600):
    parent = Path(parent).resolve()
    if method not in ('H', 'H0', 'B1', 'B2') or type(wall_seconds) is not int or not 900 <= wall_seconds <= 3600:
        raise ValueError('Require a supported policy and bounded evaluation window')
    require_closed_method_parent(parent)
    old = read(parent / 'protocol.json')
    if method not in old['methods'] or old.get('evaluation_only'):
        raise ValueError('Require an actually PPOtrained parent')
    raw = audit(parent, method)
    if raw != read(parent / method / 'endpoint_raw_audit.json'):
        raise ValueError('Parent raw audit changed')
    inventory = old.get('native_validation', {}).get('manifest', old.get('native_case_manifest'))
    expected = len(inventory['cases']) if inventory is not None else len(old['routes'])
    episodes = read(parent / method / 'evaluation.json')['episodes']
    if sorted(row['case_id'] for row in episodes) != list(range(expected)):
        raise ValueError('Require every registered parent evaluation case,not a successful subset')
    folder = parent / method
    summary = read(folder / 'summary.json')
    origin = folder / read(folder / 'latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    if (summary['stopped_reason'] is not None or not summary['reload_exact']
            or summary['physics_steps'] <= 0 or summary['optimizer_steps'] <= 0
            or any(metadata[k] != summary[k] for k in ('physics_steps', 'optimizer_steps', 'lineage_control_steps'))
            or sha(folder / 'model.zip') != metadata['model_sha256']):
        raise ValueError('Require the complete trained endpoint,not an earlier selected checkpoint')
    records = {enabled: deployment_record(read(origin / 'config.json'), enabled) for enabled in (False, True)}
    for name, digest in old['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Parent source changed: ' + name)
    if sha(parent / 'calibration/calibration.json') != old['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    changes = {'navstack/env.py': environment_overlay((parent / 'source/navstack/env.py').read_text(encoding='utf-8-sig')),
               'scripts/native_road_learning.py': worker_overlay((parent / 'source/scripts/native_road_learning.py').read_text(encoding='utf-8-sig'))}
    with ProcessLease(parent / ('shared_brake_' + method + '_prepare.lock')):
        registration = parent / ('shared_brake_' + method + '_pair_registration.json')
        if registration.exists():
            raise FileExistsError('Preserve the registered pair;no repeated preparation')
        entries = []
        atomic_json(registration, {'status': 'PREPARING', 'batches': entries})
        try:
            for enabled in (False, True):
                okay, reason = resources_ok(ROOT)
                if not okay or shutil.disk_usage(ROOT).free < 12 * 1024**3:
                    raise RuntimeError(str(reason))
                mode = 'ON' if enabled else 'OFF'
                batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_shared_brake_' + mode)
                batch.mkdir()
                (batch / 'infrastructure').mkdir()
                entries.append({'enabled': enabled, 'batch': batch.name})
                atomic_json(registration, {'status': 'PREPARING', 'batches': entries})
                shutil.copytree(parent / 'source', batch / 'source', ignore=shutil.ignore_patterns('__pycache__'))
                shutil.copytree(parent / 'calibration', batch / 'calibration')
                for name, text in changes.items():
                    (batch / 'source' / name).write_text(text, encoding='utf-8')
                destination = batch / 'resume' / method
                destination.parent.mkdir()
                shutil.copytree(origin, destination)
                write_json(destination / 'config.json', records[enabled])
                for name in ('model.zip', 'normalization.pkl', 'resume_state.pkl'):
                    if sha(destination / name) != sha(origin / name):
                        raise ValueError('Frozen learning bytes changed')
                verified_checkpoint_metadata(destination)
                if old.get('extra_supervision'):
                    dataset = parent / 'measured_demonstrations.npz'
                    if sha(dataset) != old['extra_supervision']['dataset_sha256']:
                        raise ValueError('Registered historical teacher data changed')
                    shutil.copy2(dataset, batch / dataset.name)
                bridge = 'scripts/carla_bridge37.py'
                cleanup = copy_walker_spawn_harness(parent / 'source' / bridge, ROOT / bridge, batch / 'source' / bridge)
                shutil.copy2(Path(__file__), batch / 'source/scripts' / Path(__file__).name)
                write_json(batch / 'parent_protocol.json', old)
                write_json(batch / 'parent_training_summary.json', summary)
                write_json(batch / 'parent_guard_final.json', read(parent / 'guard_final.json'))
                if (parent / 'failure.json').exists():
                    write_json(batch / 'parent_sibling_failure_retained.json', read(parent / 'failure.json'))
                if (parent / 'launcher_failure.json').exists():
                    write_json(batch / 'parent_launcher_failure_retained.json', read(parent / 'launcher_failure.json'))
                protocol = deepcopy(old)
                for name in ('jobs', 'own_success_replay', 'supervised_representation_gates', 'initialization_gate_profile',
                             'supervised_ppo_intervention', 'freeze_observation_statistics'):
                    protocol.pop(name, None)
                prior = old.get('recovery', {}).get(method, {})
                now = time.time()
                protocol.update(methods=[method], evaluation_only=True, controlled_budget_per_method=0,
                    method_budgets={method: 0}, checkpoint_thresholds=[], created_epoch=now,
                    deadline_epoch=now + wall_seconds, parent_batch=parent.name,
                    walker_spawn_initialization=cleanup,
                    recovery={method: {**deepcopy(prior), 'checkpoint': 'resume/' + method,
                        'parent_batch': parent.name, 'parent_checkpoint': origin.name,
                        'lineage_controls': summary['lineage_control_steps'],
                        'consumed_controls_lower_bound': summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'],
                        'retained_optimizer_steps': prior.get('retained_optimizer_steps', 0) + summary['optimizer_steps']}},
                    deployment_shared_brake={'version': VERSION, 'enabled': enabled,
                        'native_reference_safety_gate_retained': True, 'new_training_controls': 0,
                        'new_optimizer_updates': 0, 'model_sha256': metadata['model_sha256'],
                        'normalization_sha256': metadata['normalization_sha256'],
                        'brake_source_sha256': sha(batch / 'source/navstack/shield.py'),
                        'source_overrides': {name: {'parent_sha256': sha(parent / 'source' / name),
                            'initialized_sha256': sha(batch / 'source' / name)} for name in changes},
                        'walker_initialization': cleanup,
                        'parent_had_failure_or_interruption': ((parent / 'failure.json').exists()
                            or (parent / 'launcher_failure.json').exists()
                            or read(parent / 'guard_final.json')['exit_code'] != 0
                            or read(parent / 'guard_final.json').get('reason') is not None),
                        'limits': 'Same existing current-curvature stopping brake,not a safety certificate. Deployment intervention only;policy was not retrained under this wrapper. Native safety gates remain active;OFF/ONreplays are paired,not independent training seeds.'},
                    scope='Paired frozen-policy DEVELOPMENT deployment comparison,common post-command brake OFF/ON;all original native safety gates retained;zero PPO or auxiliary updates')
                protocol['resource_limits']['wall_seconds'] = wall_seconds
                protocol['source_hashes'] = {p.relative_to(batch / 'source').as_posix(): sha(p)
                    for p in (batch / 'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
                write_json(batch / 'protocol.json', protocol)
                entries[-1]['protocol_sha256'] = sha(batch / 'protocol.json')
                write_json(batch / 'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': entries[-1]['protocol_sha256']})
            atomic_json(registration, {'status': 'PREPARED_BOTH_ARMS_NOT_STARTED', 'batches': entries})
        except BaseException as error:
            atomic_json(registration, {'status': 'FAILED_PREPARATION_PARTIALS_RETAINED', 'error': repr(error), 'batches': entries})
            raise
    print(str(registration), flush=True)
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--method', required=True, choices=['H', 'H0', 'B1', 'B2'])
    parser.add_argument('--wall-seconds', type=int, default=3600)
    args = parser.parse_args()
    prepare(args.parent, args.method, args.wall_seconds)

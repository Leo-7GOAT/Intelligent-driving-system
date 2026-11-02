"""Queue a uniform, current-interface B3 TRAIN-only teacher collection after all registered B1 courses."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import signature
from navstack.owned_process import OwnedProcess
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha, read_prefix
from scripts.audit_native_control_delivery import control_errors
from scripts.evaluate_native_rule import require_finished_parent

PARENT = ROOT / 'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO'
DEPENDENCY = ROOT / 'runs/B1_additional_broader_cohort_20261102'


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the existing teacher preparation')
    require_finished_parent(PARENT)
    eligibility = read(ROOT / 'runs/broader_teacher_eligibility_20261102.json')
    if eligibility['coverage_complete'] or len(eligibility['covered_TRAIN_cases']) != 31:
        raise ValueError('Inspect the recorded old-interface reuse preflight before new collection')
    okay, reason = resources_ok(ROOT)
    if not okay:
        raise RuntimeError(str(reason))
    old = read(PARENT / 'protocol.json')
    for name, digest in old['source_hashes'].items():
        if sha(PARENT / 'source' / name) != digest:
            raise ValueError('Frozen shared source changed')
    cfg = EnvConfig.from_dict(read(PARENT / 'B1/config.json')['env'])
    cfg = EnvConfig.from_dict({**cfg.to_dict(), 'method': 'H', 'flat_hold': 10,
        'endpoint_candidate_filter': True})
    if cfg.to_dict() != read(ROOT / 'runs/20261101T104917Z_native_rule/protocol.json')['config']:
        raise ValueError('Teacher must retain the complete verified strong B3 configuration')
    if not cfg.goal_parking_assist or cfg.planner.max_seconds is not None:
        raise ValueError('Shared versioned helper and fixed node budget required')
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_B3_broader_TRAIN_teacher')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    for name in ('source', 'calibration'):
        shutil.copytree(PARENT / name, batch / name, ignore=shutil.ignore_patterns('__pycache__'))
    atomic_json(batch / 'source_config.json', dict(env=cfg.to_dict(), signature=signature(cfg),
        scope='Same physical constraints, map/candidate information, parking helper, planner and MPC; B3 rule selects the shared H catalog.'))
    atomic_json(batch / 'case_manifest.json', old['native_case_manifest'])
    now = time.time()
    p = dict(scope='Uniform physical B3 teacher collection on ALL36existingTRAINconditions; extra supervision only,not PPO/validation/final confirmation.',
        parent_batch=PARENT.name, parent_root_relative=PARENT.relative_to(ROOT).as_posix(),
        parent_protocol_sha256=sha(PARENT / 'protocol.json'), source_method='B1_configuration_to_shared_H_rule_interface',
        source_config_sha256=sha(batch / 'source_config.json'), config=cfg.to_dict(), routes=deepcopy(old['routes']),
        native_case_manifest=deepcopy(old['native_case_manifest']), case_manifest_sha256=sha(batch / 'case_manifest.json'),
        rule='existing_navstack.env.rule_action', rule_speed_policy='native_score',
        rule_limitation='Existing route-frame strong rule with unchanged shared planner/control/safety; no learned decision claim.',
        created_epoch=now, deadline_epoch=now + 3600, calibration_sha256=old['calibration_sha256'],
        cleanup_owned_carla=True, carla_rpc_host=old['carla_rpc_host'], carla_rpc_port=old['carla_rpc_port'],
        carla_rpc_port_span=old['carla_rpc_port_span'], resource_limits={**old['resource_limits'], 'wall_seconds': 3600},
        training_physics_steps=0, optimizer_updates=0, source_hashes=deepcopy(old['source_hashes']),
        data_role='TRAIN_EXTRA_SUPERVISION', validation_conditions_used=0, final_conditions_used=0,
        teacher_collection=dict(all_original_36_TRAIN_conditions=True, existing_basic_failures_retained=True,
            old_reuse_preflight_sha256=sha(ROOT / 'runs/broader_teacher_eligibility_20261102.json'),
            rationale='Strict whole-successful-episode/engine-ack reuse of all10oldH/H0TRAINstreams covers31/36only. Fresh uniform same-interface collection avoids lowering the filter or using VALlabels.',
            no_new_policy_fit=True, no_model_selection=True, all_cases_and_failures_retained=True))
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(batch / 'protocol.json')))
    output.mkdir()
    atomic_json(output / 'registration.json', dict(batch=batch.name, protocol_sha256=sha(batch / 'protocol.json'),
        data_role='TRAIN_EXTRA_SUPERVISION', cases=36, validation_cases=0, PPO_updates=0,
        dependency_execution=DEPENDENCY.name, source_config_sha256=sha(PARENT / 'B1/config.json'),
        maximum_native_wall_seconds=3600, maximum_dataset_wall_seconds=1800, automatic_retries=0,
        new_controls_so_far=0, model_fitting_started=False))
    print(batch, flush=True)


def run(output):
    output = Path(output).resolve()
    if (output / 'started.json').exists():
        raise FileExistsError('Never relaunch a started teacher observer')
    registration = read(output / 'registration.json')
    if not read(output / 'validation.json')['frozen_import_and_launcher_verified']:
        raise ValueError('Actual frozen preflight is required')
    identity = read(output / 'dependency_identity.json')
    owner = OwnedProcess(identity['pid'], identity['creation_epoch'], identity['executable'], allow_terminate=False)
    if owner.poll() is not None:
        raise ValueError('Dependency already ended; inspect completion before recovery')
    batch = ROOT / 'runs' / registration['batch']
    source = output / 'source'
    for directory in ('scripts', 'navstack'):
        shutil.copytree(ROOT / directory, source / directory, ignore=shutil.ignore_patterns('__pycache__'))
    hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    atomic_json(output / 'source_manifest.json', hashes)
    atomic_json(output / 'started.json', dict(pid=os.getpid(), epoch=time.time(), dependency=identity, automatic_retries=0))
    deadline = time.monotonic() + 43200
    def status(phase):
        atomic_json(output / 'status.json', dict(phase=phase, pid=os.getpid(), epoch=time.time(), batch=batch.name))
    def call(args, label, cap):
        for name, digest in hashes.items():
            if sha(source / name) != digest:
                raise ValueError('Frozen observer changed')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Overall finite observer budget exhausted')
        with (output / (label + '.log')).open('xb') as log:
            child = subprocess.Popen(list(map(str, args)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(output / 'last_child.json', dict(pid=child.pid, label=label, epoch=time.time()))
            code = child.wait(timeout=min(cap, remaining))
        if code:
            raise RuntimeError('Child failed without automatic retry: ' + label)
    try:
        with ProcessLease(output / 'observer.lock'):
            while owner.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('Dependency wait exhausted')
                status('WAITING_FOR_EXACT_BROADER_B1_COHORT_EXIT')
                time.sleep(10)
            if ((DEPENDENCY / 'failure.json').exists()
                    or read(DEPENDENCY / 'status.json')['phase'] != 'ALL_FOUR_ADDITIONAL_BROADER_COURSES_AUDITED'):
                raise ValueError('Existing locked cohort must complete and be audited first')
            if sha(batch / 'protocol.json') != registration['protocol_sha256']:
                raise ValueError('Unstarted teacher protocol changed')
            okay, reason = resources_ok(ROOT)
            if not okay:
                raise RuntimeError(str(reason))
            status('COLLECTING_ALL36_TRAIN_TEACHER')
            call([shutil.which('pwsh'), '-NoProfile', '-File', source / 'scripts/run_registered_native.ps1',
                '-Batch', batch, '-Kind', 'rule', '-Python', ROOT / '.venv/Scripts/python.exe',
                '-CarlaPython', ROOT / '.carla-python37/runtime/python.exe', '-ActivateQueued', '-UseCurrentGuard'], 'native', 4200)
            require_finished_parent(batch)
            call([sys.executable, batch / 'source/scripts/audit_native_rule.py', '--batch', batch], 'raw_audit', 600)
            raw = read(batch / 'raw_audit.json')
            streams = list((batch / 'infrastructure').glob('*_responses.jsonl.gz'))
            if len(streams) != 1 or sha(streams[0]) != raw['bridge_response_sha256']:
                raise ValueError('Audited teacher bridge stream differs')
            responses, error = read_prefix(streams[0])
            if error:
                raise ValueError(error)
            acknowledged = 0
            for wrapper in responses:
                response = wrapper['response']
                if 'vehicle_control' not in response:
                    continue
                applied = response['applied_vehicle_control']
                if (control_errors(response['vehicle_control'], applied)
                        or response.get('control_delivery') != 'batch_sync_v1'
                        or response.get('applied_control_matches_request') is not True
                        or applied['manual_gear_shift'] is not True or applied['gear'] != 1
                        or applied['hand_brake'] is not False or applied['reverse'] is not False):
                    raise ValueError('Teacher control lacks an exact synchronous engine acknowledgement')
                acknowledged += 1
            if acknowledged != raw['evaluation_controls_verified']:
                raise ValueError('Teacher acknowledgement inventory differs')
            atomic_json(output / 'strict_delivery.json', dict(verified_controls=acknowledged,
                all_engine_controls_acknowledged=True, tolerance=1e-5, new_controls=0))
            status('BUILDING_EXECUTED_TRAIN_DIRECT_VIEWS')
            conversion = output / 'conversion_source'
            shutil.copytree(batch / 'source', conversion, ignore=shutil.ignore_patterns('__pycache__'))
            # Exact frozen reconstruction; executed teacher braking is a nominal
            # direct action, not another target-policy helper intervention.
            atomic_json(output / 'conversion_source_manifest.json', dict(
                physical_source_hashes=read(batch / 'protocol.json')['source_hashes'],
                explicit_read_only_conversion_overlays={},
                reason='Unchanged measured direct views;teacher-state labels are not counterfactual B1 episodes or value targets.'))
            call([sys.executable, conversion / 'scripts/build_native_measured_views.py',
                '--reference-config', PARENT / 'B1/config.json', '--teachers', batch,
                '--output', output / 'direct_views', '--wall-seconds', '1800'], 'direct_views', 1860)
            status('ALL36_TRAIN_TEACHER_AND_DIRECT_VIEWS_COMPLETE')
    except BaseException as error:
        atomic_json(output / 'failure.json', dict(error=repr(error), epoch=time.time()))
        status('STOPPED_WITH_FAILURE_NO_RETRY')
        raise
    finally:
        owner.kernel.CloseHandle(owner.handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    (run if args.run else prepare)(args.output)

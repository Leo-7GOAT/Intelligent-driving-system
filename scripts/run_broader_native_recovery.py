"""Continue one live TRAIN collection through audited fixed fitting and gated probes."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.owned_process import OwnedProcess
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent


def basic_gate(evaluation):
    episodes = evaluation['episodes']
    if sorted(e['case_id'] for e in episodes) != [0, 1, 2, 3]:
        raise ValueError('Require every original basic TRAIN road')
    return sum(e['success'] for e in episodes) >= 3 and not any(
        e['termination_reason'] in ('collision', 'out_of_bounds') for e in episodes)


def run(execution, completed_collection=False):
    execution = Path(execution).resolve()
    if (execution / 'continuation_started.json').exists():
        raise FileExistsError('Never restart a launched continuation or repeat physical controls')
    registration = read(execution / 'registration.json')
    profile = registration.get('fit_profile','original')
    if profile not in ('original','steering_v2','convergence_v3'): raise ValueError('Unknown registered fixed fit profile')
    batch = ROOT / 'runs' / registration['batch']
    if sha(batch / 'protocol.json') != registration['protocol_sha256']:
        raise ValueError('Registered collection protocol changed')
    identity = read(batch / 'worker_process.json')
    owner = None
    if completed_collection:
        require_finished_parent(batch)
        if not (batch / 'B1/collection.json').exists():
            raise ValueError('Completed-collection adoption requires the actual saved dataset/result')
    else:
        owner = OwnedProcess(identity['pid'], identity['creation_epoch'], identity['executable'], allow_terminate=False)
        if owner.poll() is not None:
            owner.kernel.CloseHandle(owner.handle)
            raise ValueError('Worker already ended; inspect its final state before an explicit continuation')
    files = [p for directory in ('scripts', 'navstack') for p in (ROOT / directory).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.ps1') and '__pycache__' not in p.parts]
    files.append(ROOT / 'protocols/native_B1_broader_executed_recovery_20261102.json')
    if profile=='steering_v2': files.append(ROOT / 'protocols/native_B1_broader_executed_recovery_steering_20261102.json')
    if profile=='convergence_v3': files.append(ROOT / 'protocols/native_B1_broader_executed_recovery_convergence_20261102.json')
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    atomic_json(execution / 'continuation_source_manifest.json', hashes)
    atomic_json(execution / 'continuation_started.json', dict(pid=os.getpid(), epoch=time.time(),
        dependency=identity, completed_collection_adoption=completed_collection, automatic_retries=0, maximum_wall_seconds=10800))
    deadline = time.monotonic() + 10800
    def status(phase, **extra):
        atomic_json(execution / 'status.json', dict(phase=phase, pid=os.getpid(), epoch=time.time(), **extra))
    def call(args, label, timeout, env=None):
        for name, digest in hashes.items():
            if sha(ROOT / name) != digest: raise ValueError('Continuation source changed: ' + name)
        okay, reason = resources_ok(ROOT)
        if not okay: raise RuntimeError(str(reason))
        with (execution / (label + '.log')).open('xb') as stream:
            process = subprocess.Popen(list(map(str, args)), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            atomic_json(execution / 'last_child.json', dict(pid=process.pid, label=label, epoch=time.time()))
            code = process.wait(timeout=min(timeout, max(1, deadline - time.monotonic())))
        if code: raise RuntimeError('Child failed; preserve logs and stop without retry: ' + label)
    def native(current, label):
        status(label.upper())
        call([shutil.which('pwsh'), '-NoProfile', '-File', ROOT / 'scripts/run_registered_native.ps1',
            '-Batch', current, '-UseCurrentGuard'], label, 4200)
        require_finished_parent(current)
    def endpoint(current, label):
        env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(current / 'source'), PYTHONIOENCODING='utf-8')
        call([sys.executable, ROOT / 'scripts/audit_native_method_endpoint.py', '--batch', current, '--method', 'B1'], label + '_raw', 900, env)
        call([sys.executable, ROOT / 'scripts/audit_native_policy_inference.py', '--batch', current, '--method', 'B1',
            '--output', current / 'B1/measured_start_inference.json'], label + '_starts', 900, env)
        call([sys.executable, ROOT / 'scripts/audit_native_control_delivery.py', '--batch', current,
            '--output', current / 'B1/strict_delivery.json'], label + '_delivery', 900)
        if not read(current / 'B1/strict_delivery.json')['strict_delivery_gate_passed']:
            raise ValueError('Engine delivery gate failed')
    try:
        with ProcessLease(execution / 'continuation.lock'):
            while owner is not None and owner.poll() is None:
                if time.monotonic() > deadline: raise TimeoutError('Finite continuation wait exhausted')
                status('WAITING_FOR_EXACT_LIVE_TRAIN_COLLECTION')
                time.sleep(10)
            if owner is not None and owner.poll() != 0: raise RuntimeError('Collection worker failed; do not retry')
            # The verified worker exits before its wrapper finishes owned cleanup.
            cleanup_deadline = time.monotonic() + 90
            while not (batch / 'guard_final.json').exists() or not (batch / 'infrastructure/server_stop.json').exists():
                if time.monotonic() > cleanup_deadline: raise TimeoutError('Owned lifecycle cleanup did not finish')
                time.sleep(2)
            require_finished_parent(batch)
            status('AUDITING_ALL_MEASURED_TRAIN_RECOVERY_CONTROLS')
            raw_path = batch / 'B1/collection_raw_audit.json'
            if completed_collection and raw_path.exists():
                proof = read(raw_path)
                if (proof['protocol_sha256'] != sha(batch / 'protocol.json')
                        or proof['collection_result_sha256'] != sha(batch / 'B1/collection.json')
                        or proof['script_sha256'] != sha(ROOT / 'scripts/audit_native_recovery_collection.py')
                        or not proof['all_measured_observations_bit_exact'] or not proof['all_learner_proposals_bit_exact']):
                    raise ValueError('Existing completed collection audit differs')
                atomic_json(execution / 'adopted_completed_audit.json', dict(sha256=sha(raw_path), new_controls=0, new_updates=0))
            else:
                call([sys.executable, ROOT / 'scripts/audit_native_recovery_collection.py', '--batch', batch,
                    '--output', raw_path], 'collection_raw', 1200)
            delivery_path = batch / 'B1/strict_delivery.json'
            if completed_collection and delivery_path.exists():
                proof = read(delivery_path)
                if (not proof['strict_delivery_gate_passed']
                        or proof['protocol_sha256'] != sha(batch / 'protocol.json')
                        or any(sha(batch / 'infrastructure' / s['name']) != s['sha256'] for s in proof['bridge_streams'])
                        or proof['counts']['controls'] != read(batch / 'B1/collection.json')['collection_controls']):
                    raise ValueError('Existing collection delivery audit differs')
                atomic_json(execution / 'adopted_delivery_audit.json', dict(sha256=sha(delivery_path), new_controls=0, new_updates=0))
            else:
                call([sys.executable, ROOT / 'scripts/audit_native_control_delivery.py', '--batch', batch,
                    '--output', delivery_path], 'collection_delivery', 900)
            fit = execution / 'canonical22_fit'
            status('FIXED_TRAIN_ONLY_RECOVERY_ACTOR_FIT')
            call([sys.executable, ROOT / 'scripts/fit_broader_native_recovery.py', '--collection', batch,
                '--views', ROOT / 'runs/B3_broader_TRAIN_teacher_execution_v4_20261102/direct_views',
                '--reference-config', ROOT / 'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO/B1/config.json',
                '--output', fit, '--profile', profile], 'canonical22_fit', 720)
            basic_output = execution / 'basic_probe'
            call([sys.executable, ROOT / 'scripts/prepare_broader_native_recovery.py', '--probe-fit', fit,
                '--output', basic_output], 'prepare_basic', 180)
            basic = ROOT / 'runs' / read(basic_output / 'registration.json')['batch']
            native(basic, 'basic_native'); endpoint(basic, 'basic')
            call([sys.executable, ROOT / 'scripts/audit_native_flat_sequence.py', '--batch', basic,
                '--output', basic / 'B1/whole_sequence_inference.json'], 'basic_sequence', 900)
            passed = basic_gate(read(basic / 'B1/evaluation.json'))
            if not passed:
                status('FIXED_RECOVERY_ENDPOINT_FAILED_BASIC_GATE', basic_batch=basic.name)
                atomic_json(execution / 'complete.json', dict(basic_batch=basic.name, basic_gate_passed=False,
                    validation_started=False, research_acceptance='UNMET'))
                return
            validation_output = execution / 'validation_probe'
            call([sys.executable, ROOT / 'scripts/prepare_measured_actor_validation_probe.py', '--parent', basic,
                '--output', validation_output], 'prepare_validation', 180)
            validation = ROOT / 'runs' / read(validation_output / 'registration.json')['batch']
            native(validation, 'known_validation_native'); endpoint(validation, 'known_validation')
            evaluated = read(validation / 'B1/evaluation.json')['episodes']
            atomic_json(execution / 'complete.json', dict(basic_batch=basic.name, basic_gate_passed=True,
                validation_batch=validation.name, successes=sum(e['success'] for e in evaluated), cases=len(evaluated),
                new_PPO_updates=0, final_conditions=0, research_acceptance='UNMET'))
            status('FIXED_RECOVERY_DEVELOPMENT_PROBES_COMPLETE_NO_PPO_PROMOTION')
    except BaseException as error:
        atomic_json(execution / 'continuation_failure.json', dict(error=repr(error), epoch=time.time()))
        status('CONTINUATION_STOPPED_WITH_FAILURE')
        raise
    finally:
        if owner is not None: owner.kernel.CloseHandle(owner.handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--execution', required=True)
    parser.add_argument('--completed-collection', action='store_true', help='Explicit adoption after verified owned cleanup;never starts the collection again')
    args = parser.parse_args(); run(args.execution, args.completed_collection)

"""Run all five registered actual-PPO courses serially, retaining failures and frozen audit tools."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent


def require_all_five(state):
    rows = state['batches']
    if (state['phase'] != 'ALL_FIVE_PREPARED_NOT_DRIVEN' or [r['seed'] for r in rows] != [11, 22, 33, 44, 55]
            or len({r['batch'] for r in rows}) != 5 or len({r['initial_model_sha256'] for r in rows}) != 5
            or any(Path(r['batch']).name != r['batch'] for r in rows)):
        raise ValueError('Every registered independent course is required without duplicate batches or actors')
    return rows


def run(preparation, execution):
    preparation, execution = Path(preparation).resolve(), Path(execution).resolve()
    rows = require_all_five(read(preparation / 'status.json'))
    if execution.exists():
        raise FileExistsError('Never restart an existing queue; inspect/recover its exact child instead')
    require_finished_parent(ROOT / 'runs/20261101T121222965681Z_native_B1_validation_remainder')
    execution.mkdir(parents=True)
    shutil.copy2(preparation / 'status.json', execution / 'prepared_courses.json')
    source = execution / 'source'
    for directory in ('navstack', 'scripts'):
        for original in (ROOT / directory).rglob('*'):
            if original.is_file() and original.suffix in ('.py', '.ps1') and '__pycache__' not in original.parts:
                target = source / original.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
    hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    atomic_json(execution / 'audit_source_manifest.json', hashes)
    atomic_json(execution / 'started.json', {'pid': os.getpid(), 'epoch': time.time(), 'automatic_retries': 0,
        'maximum_observer_wall_seconds': 43200, 'preparation_sha256': sha(preparation / 'status.json')})
    completed = []
    deadline = time.monotonic() + 43200

    def status(phase, **extra):
        atomic_json(execution / 'status.json', {'phase': phase, 'pid': os.getpid(), 'epoch': time.time(),
                    'completed': completed, **extra})

    def call(arguments, label, cap, env=None):
        for name, digest in hashes.items():
            if sha(source / name) != digest:
                raise ValueError('Frozen queue/audit source changed')
        timeout = min(cap, deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError('Finite queue observer deadline exhausted')
        with (execution / (label + '.log')).open('xb') as stream:
            process = subprocess.Popen(list(map(str, arguments)), cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            atomic_json(execution / 'last_child.json', {'pid': process.pid, 'epoch': time.time(), 'label': label})
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise TimeoutError('Observer expired;native watchdog retains ownership;no automatic retry')
        if code:
            raise RuntimeError('Queue child failed;retain logs and stop without retry:' + label)

    try:
        with ProcessLease(execution / 'queue.lock'):
            for row in rows:
                okay, reason = resources_ok(ROOT)
                if not okay:
                    raise RuntimeError(str(reason))
                batch = ROOT / 'runs' / row['batch']
                if sha(batch / 'protocol.json') != row['protocol_sha256']:
                    raise ValueError('An unstarted registered protocol changed')
                if any((batch / name).exists() for name in ('started.json', 'launcher_execution.json', 'guard_final.json')):
                    raise ValueError('Never replay an existing native lifecycle')
                prefix = 'seed' + str(row['seed'])
                status('ACTUAL_PPO_BASIC_COURSE_RUNNING', seed=row['seed'], batch=batch.name)
                call([shutil.which('pwsh'), '-NoProfile', '-File', source / 'scripts/run_registered_native.ps1',
                    '-Batch', batch, '-Python', ROOT / '.venv/Scripts/python.exe',
                    '-CarlaPython', ROOT / '.carla-python37/runtime/python.exe', '-ActivateQueued', '-UseCurrentGuard'],
                    prefix + '_lifecycle', 7800)
                require_finished_parent(batch)
                env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'), PYTHONIOENCODING='utf-8')
                call([sys.executable, source / 'scripts/audit_native_method_endpoint.py', '--batch', batch, '--method', 'B1'],
                    prefix + '_raw_audit', 300, env)
                call([sys.executable, source / 'scripts/audit_native_policy_inference.py', '--batch', batch, '--method', 'B1',
                      '--output', batch / 'B1/measured_start_inference.json'], prefix + '_measured_start', 300, env)
                status('VERIFYING_COMPLETED_ACTUAL_PPO', seed=row['seed'], batch=batch.name)
                output = execution / (prefix + '_learning_audit.json')
                call([sys.executable, source / 'scripts/analyze_native_shared_parking_ppo.py', '--batch', batch,
                      '--output', output], prefix + '_learning_audit', 300)
                call([sys.executable, source / 'scripts/audit_native_flat_sequence.py', '--batch', batch,
                      '--output', batch / 'B1/whole_sequence_inference.json'], prefix + '_whole_sequence', 600)
                result = read(output)
                if result['new_actual_PPO_controls'] != 10240 or result['seed'] != row['seed']:
                    raise ValueError('Actual course differs from the registered seed/budget')
                completed.append({'seed': row['seed'], 'batch': batch.name, 'learning_audit_sha256': sha(output),
                    'new_actual_PPO_controls': result['new_actual_PPO_controls'],
                    'new_actual_Adam_steps': result['new_actual_Adam_steps'],
                    'endpoint': result['endpoint'], 'model_sha256': result['final_model_sha256']})
                status('ONE_ACTUAL_PPO_COURSE_COMPLETE', seed=row['seed'])
            status('ALL_FIVE_QUALIFIED_AFTER_ACTUAL_PPO' if all(r['endpoint']['basic_gate_passed'] for r in completed)
                   else 'ALL_FIVE_COMPLETE_WITH_FAILED_BASIC_GATES')
    except BaseException as error:
        atomic_json(execution / 'failure.json', {'error': repr(error), 'epoch': time.time(), 'completed': completed})
        status('QUEUE_STOPPED_WITH_FAILURE')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--execution', required=True)
    args = parser.parse_args()
    run(args.preparation, args.execution)

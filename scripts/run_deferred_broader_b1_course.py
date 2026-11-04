"""Wait on the exact live basic-cohort process,then run one registered broader study without retries."""
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


def run(preparation, execution):
    preparation, execution = Path(preparation).resolve(), Path(execution).resolve()
    if execution.exists():
        raise FileExistsError('Preserve the existing deferred observer')
    registration = read(preparation / 'registration.json')
    identity = read(preparation / 'dependency_identity.json')
    owner = OwnedProcess(identity['pid'], identity['creation_epoch'], identity['executable'], allow_terminate=False)
    if owner.poll() is not None:
        owner.kernel.CloseHandle(owner.handle)
        raise ValueError('Dependency is already terminal;inspect its completed state before preparing another observer')
    batch = ROOT / 'runs' / registration['batch']
    if sha(batch / 'protocol.json') != registration['protocol_sha256']:
        raise ValueError('Registered unstarted broader protocol changed')
    execution.mkdir(parents=True)
    source = execution / 'source'
    for directory in ('navstack', 'scripts'):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.ps1') and '__pycache__' not in path.parts:
                target = source / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    atomic_json(execution / 'source_manifest.json', hashes)
    atomic_json(execution / 'started.json', {'pid': os.getpid(), 'epoch': time.time(), 'dependency_pid': owner.pid,
        'dependency_creation_epoch': identity['creation_epoch'], 'batch': batch.name, 'automatic_retries': 0})
    deadline = time.monotonic() + 43200
    def status(phase):
        atomic_json(execution / 'status.json', {'phase': phase, 'pid': os.getpid(), 'epoch': time.time(),
            'batch': batch.name, 'dependency_pid': owner.pid})
    def call(args, label, timeout, env=None):
        for name, digest in hashes.items():
            if sha(source / name) != digest:
                raise ValueError('Frozen observer/audit source changed')
        with (execution / (label + '.log')).open('xb') as stream:
            process = subprocess.Popen(list(map(str, args)), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            atomic_json(execution / 'last_child.json', {'pid': process.pid, 'label': label, 'epoch': time.time()})
            code = process.wait(timeout=min(timeout, max(1, deadline - time.monotonic())))
        if code:
            raise RuntimeError('Owned child failed;retain logs and stop without retry:' + label)
    try:
        with ProcessLease(execution / 'observer.lock'):
            while owner.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('Finite dependency wait exhausted')
                status('WAITING_FOR_EXACT_BASIC_COHORT_PROCESS_EXIT')
                time.sleep(10)
            dependency = ROOT / 'runs' / registration['dependency_execution']
            state = read(dependency / 'status.json')
            if ((dependency / 'failure.json').exists() or state['phase'] not in
                    ('ALL_FIVE_QUALIFIED_AFTER_ACTUAL_PPO', 'ALL_FIVE_COMPLETE_WITH_FAILED_BASIC_GATES')
                    or [r['seed'] for r in state['completed']] != [11, 22, 33, 44, 55]):
                raise ValueError('All five original courses must finish and retain their outcomes;no retry or partial selection')
            atomic_json(execution / 'completed_dependency.json', state)
            okay, reason = resources_ok(ROOT)
            if not okay:
                raise RuntimeError(str(reason))
            if sha(batch / 'protocol.json') != registration['protocol_sha256']:
                raise ValueError('Queued broader protocol changed before its first lifecycle')
            status('BROADER_ACTUAL_PPO_RUNNING')
            call([shutil.which('pwsh'), '-NoProfile', '-File', source / 'scripts/run_registered_native.ps1',
                '-Batch', batch, '-Python', ROOT / '.venv/Scripts/python.exe', '-CarlaPython', ROOT / '.carla-python37/runtime/python.exe',
                '-ActivateQueued', '-UseCurrentGuard'], 'native_lifecycle', 7800)
            require_finished_parent(batch)
            env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'), PYTHONIOENCODING='utf-8')
            status('VERIFYING_BROADER_ACTUAL_PPO')
            call([sys.executable, source / 'scripts/audit_native_method_endpoint.py', '--batch', batch, '--method', 'B1'], 'raw_audit', 600, env)
            call([sys.executable, source / 'scripts/audit_native_policy_inference.py', '--batch', batch, '--method', 'B1',
                '--output', batch / 'B1/measured_start_inference.json'], 'measured_starts', 600, env)
            call([sys.executable, source / 'scripts/analyze_native_broader_b1_course.py', '--batch', batch,
                '--output', execution / 'learning_audit.json'], 'learning_audit', 600)
            status('BROADER_STUDY_COMPLETE_AND_AUDITED')
    except BaseException as error:
        atomic_json(execution / 'failure.json', {'error': repr(error), 'epoch': time.time()})
        status('DEFERRED_STUDY_STOPPED_WITH_FAILURE')
        raise
    finally:
        owner.kernel.CloseHandle(owner.handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--execution', required=True)
    args = parser.parse_args()
    run(args.preparation, args.execution)

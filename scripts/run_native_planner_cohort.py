"""Finite B2 cohort after a verified predecessor;no experiment retries or kills."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(execution):
    execution = Path(execution).resolve()
    root = execution.parent.parent
    if execution.parent.name != 'runs' or not (root / '.git').exists():
        raise ValueError('Execution must be inside this project runs directory')
    frozen = execution / 'source'
    manifest = read(execution / 'source_manifest.json')

    def verify_sources():
        for name, digest in manifest.items():
            path = (frozen / name).resolve()
            if not path.is_relative_to(frozen) or sha(path) != digest:
                raise ValueError('Frozen orchestration source differs: ' + name)

    verify_sources()
    sys.path.insert(0, str(frozen))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json, resources_ok
    from scripts.audit_native_ppo_configuration import verify as verify_learner
    from scripts.evaluate_native_rule import require_finished_parent
    plan = read(execution / 'plan.json')
    if ([r['seed'] for r in plan['batches']] != [11, 22, 33, 44, 55]
            or type(plan['wait_seconds']) is not int or not 60 <= plan['wait_seconds'] <= 86400):
        raise ValueError('Require all five registered seeds and a bounded dependency wait')
    completed = []

    def status(phase, **extra):
        atomic_json(execution / 'status.json', {'phase': phase, 'pid': os.getpid(),
                    'epoch': time.time(), 'completed': completed, **extra})

    def call(command, log, timeout):
        with (execution / log).open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(list(map(str, command)), cwd=root,
                                     stdout=stream, stderr=subprocess.STDOUT)
            atomic_json(execution / 'last_child.json', {'pid': child.pid, 'epoch': time.time(),
                                                       'log': log, 'timeout_seconds': timeout})
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # The native lifecycle has its own watchdog and owns cleanup.
                raise RuntimeError('Observation expired;preserve native wrapper/watchdog PID' + str(child.pid))
            if code:
                raise RuntimeError('Registered child failed;preserve logs:' + log)

    owner = None
    try:
        with ProcessLease(execution / 'pipeline.lock'):
            with (execution / 'started.json').open('x', encoding='utf-8') as stream:
                json.dump({'epoch': time.time(), 'pid': os.getpid(),
                           'script_sha256': sha(Path(__file__)),
                           'plan_sha256': sha(execution / 'plan.json')}, stream)
            dependency = root / 'runs' / plan['dependency_execution']
            expected = 'B3_AND_FOUR_INDEPENDENT_B1_PPO_SEEDS_COMPLETE'
            until = time.monotonic() + plan['wait_seconds']
            while read(dependency / 'status.json')['phase'] != expected:
                if (dependency / 'failure.json').exists():
                    raise RuntimeError('Predecessor failed;unactivated B2 trials preserved')
                if time.monotonic() > until:
                    raise TimeoutError('Bounded predecessor wait exhausted')
                if owner is None:
                    identity = plan['dependency_process']
                    owner = OwnedProcess(identity['pid'], identity['creation_epoch'],
                                         identity['executable'], allow_terminate=False)
                if owner.poll() is not None:
                    if read(dependency / 'status.json')['phase'] == expected:
                        break
                    raise RuntimeError('Predecessor process exited without its completion marker')
                status('WAITING_FOR_REGISTERED_B1_COHORT', dependency_pid=owner.pid,
                       dependency_live_via_retained_handle=True)
                time.sleep(10)
            if (dependency / 'failure.json').exists():
                raise RuntimeError('Predecessor failure evidence must be resolved before activation')
            predecessor = read(dependency / 'status.json')['created']
            if [r['seed'] for r in predecessor] != [22, 33, 44, 55]:
                raise ValueError('Predecessor independent seed inventory differs')
            for row in predecessor:
                require_finished_parent(root / 'runs' / row['batch'])
            for row in plan['batches']:
                verify_sources()
                batch = root / 'runs' / row['batch']
                if sha(batch / 'protocol.json') != row['protocol_sha256']:
                    raise ValueError('Registered B2 protocol changed')
                protocol = read(batch / 'protocol.json')
                if (protocol['methods'] != ['B2'] or protocol['evaluation_only']
                        or protocol['seed'] != row['seed']
                        or protocol['method_budgets'] != {'B2': 20480}):
                    raise ValueError('B2 recipe or physical budget differs')
                verify_learner(batch / 'resume/B2', root / 'runs' / plan['reference'] / 'resume/H')
                okay, reason = resources_ok(root)
                if not okay:
                    raise RuntimeError(str(reason))
                status('RUNNING_REGISTERED_B2_PPO', seed=row['seed'], batch=batch.name)
                call([shutil.which('pwsh'), '-NoProfile', '-File', frozen / 'scripts/run_registered_native.ps1',
                      '-Batch', batch, '-Python', root / '.venv/Scripts/python.exe',
                      '-CarlaPython', root / '.carla-python37/runtime/python.exe',
                      '-ActivateQueued', '-UseCurrentGuard'], 'seed' + str(row['seed']) + '_lifecycle.log', 7800)
                require_finished_parent(batch)
                status('AUDITING_COMPLETE_B2_ENDPOINT', seed=row['seed'], batch=batch.name)
                call([sys.executable, frozen / 'scripts/finish_native_endpoint_evidence_frozen.py',
                      '--batch', batch.name, '--method', 'B2', '--report',
                      'native_B2_v4_seed' + str(row['seed']) + '_endpoint_20261031.json',
                      '--timeout-seconds', '600'], 'seed' + str(row['seed']) + '_audit.log', 660)
                completed.append({'seed': row['seed'], 'batch': batch.name})
                status('B2_SEED_COMPLETE', seed=row['seed'])
            status('ALL_FIVE_REGISTERED_B2_PPO_SEEDS_COMPLETE')
    except BaseException as error:
        with (execution / 'failure.json').open('x', encoding='utf-8') as stream:
            json.dump({'epoch': time.time(), 'error': repr(error), 'completed': completed,
                       'scope': 'No seed omissions,retries or process kills;all native evidence retained'}, stream, indent=2)
        raise
    finally:
        if owner is not None:
            owner.kernel.CloseHandle(owner.handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution', required=True)
    run(parser.parse_args().execution)

"""Finite B2 cohort after a verified predecessor;no experiment retries or kills."""
import argparse
from collections import Counter
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


def closed_failure_dependency(root, plan):
    """A failed B1 initialization may release the simulator for independent B2."""
    record = plan.get('closed_failed_predecessor')
    if record is None:
        return False
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.prepare_native_evaluation_recovery import recorded_lifecycle_gone
    dependency = root / 'runs' / plan['dependency_execution']
    failure = dependency / 'failure.json'
    if sha(failure) != record['failure_sha256'] or read(failure).get('created') != []:
        raise ValueError('Require the exact retained pre-PPO failure,without orphan learning children')
    batches = record['closed_native_batches']
    if not batches or any(Path(name).name != name for name in batches):
        raise ValueError('Require explicit closed native batch identities')
    for name in batches:
        require_finished_parent(root / 'runs' / name)
    if not recorded_lifecycle_gone(record['process_pids'], record['closed_epoch']):
        raise ValueError('Predecessor driver/worker/guard is still live or unverified')
    return True


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

    def call(command, log, timeout, environment=None):
        with (execution / log).open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(list(map(str, command)), cwd=root,
                                     stdout=stream, stderr=subprocess.STDOUT, env=environment)
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
            independent = closed_failure_dependency(root, plan)
            while not independent and read(dependency / 'status.json')['phase'] != expected:
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
            if independent:
                status('FAILED_B1_INITIALIZATION_RETAINED_STARTING_INDEPENDENT_B2',
                       B1_training_complete=False)
            else:
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
                environment = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'))
                inference = batch / 'B2/measured_start_inference.json'
                call([sys.executable, frozen / 'scripts/audit_native_method_endpoint.py',
                      '--batch', batch, '--method', 'B2'],
                     'seed' + str(row['seed']) + '_raw_audit.log', 300, environment)
                call([sys.executable, frozen / 'scripts/audit_native_policy_inference.py',
                      '--batch', batch, '--method', 'B2', '--output', inference],
                     'seed' + str(row['seed']) + '_inference_audit.log', 300, environment)
                folder = batch / 'B2'
                summary = read(folder / 'summary.json')
                episodes = read(folder / 'evaluation.json')['episodes']
                raw = read(folder / 'endpoint_raw_audit.json')
                actions = read(inference)
                current = read(batch / 'protocol.json')
                inventory = current.get('native_validation', {}).get('manifest', current['native_case_manifest'])
                expected_cases = len(inventory['cases'])
                if (sorted(e['case_id'] for e in episodes) != list(range(expected_cases))
                        or len(actions['results']) != expected_cases
                        or raw['training_control_frames_verified'] != summary['physics_steps']
                        or summary['optimizer_steps'] <= 0):
                    raise ValueError('Incomplete B2 learning or registered evaluation inventory')
                result = {
                    'scope': 'Actual planner-specific PPO and all registered development cases;not final confirmation',
                    'batch': batch.name, 'method': 'B2', 'seed': row['seed'],
                    'cases': expected_cases, 'successes': sum(e['success'] for e in episodes),
                    'termination_counts': dict(Counter(e['termination_reason'] for e in episodes)),
                    'new_training_controls': summary['physics_steps'],
                    'new_optimizer_steps': summary['optimizer_steps'],
                    'model_lineage_controls': summary['lineage_control_steps'],
                    'raw_controls_verified': raw['total_applied_controls_verified'],
                    'evaluation_controls': raw['evaluation_control_frames_verified'],
                    'measured_start_actions_reproduced': len(actions['results']),
                    'model_sha256': sha(folder / 'model.zip'),
                    'normalization_sha256': sha(folder / 'normalization.pkl'),
                    'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'),
                    'inference_audit_sha256': sha(inference),
                    'episodes': [{k: e[k] for k in ('case_id', 'success', 'termination_reason',
                                                   'physics_steps', 'progress')} for e in episodes],
                    'research_acceptance': 'UNMET',
                }
                report = root / 'reports' / ('native_B2_v4_seed' + str(row['seed']) + '_endpoint_20261031.json')
                with report.open('x', encoding='utf-8') as stream:
                    json.dump(result, stream, indent=2)
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

"""Twenty paired H/B1 deployment probes after the finite B2cohort;zero PPO."""
import argparse
import gzip
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


def b1_parent_map(state, profile, seed11=None):
    if profile == 'all_five_matured_v1':
        expected_phase, seeds = 'ALL_FIVE_MATURED_B1_PPO_SEEDS_COMPLETE', [11, 22, 33, 44, 55]
    elif profile == 'original_four_plus_seed11':
        expected_phase, seeds = 'B3_AND_FOUR_INDEPENDENT_B1_PPO_SEEDS_COMPLETE', [22, 33, 44, 55]
    else:
        raise ValueError('Unknown explicit B1 cohort profile')
    if state['phase'] != expected_phase or [r['seed'] for r in state['created']] != seeds:
        raise ValueError('Independent B1 cohort is incomplete')
    result = {r['seed']: r['batch'] for r in state['created']}
    if profile == 'original_four_plus_seed11':
        result[11] = seed11
    if any(not isinstance(name, str) or Path(name).name != name or name in ('.', '..') for name in result.values()):
        raise ValueError('Explicit direct-child policy batches are required')
    return result


def shared_trace_counts(path):
    controls = interventions = 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            ticks = json.loads(line)['info']['trace']
            if not isinstance(ticks, list) or not ticks:
                raise ValueError('Require the native per-decision measured control trace')
            for tick in ticks:
                value = tick['shared_shield_intervention']
                if type(value) is not bool:
                    raise ValueError('Shared intervention must be an explicit recorded boolean')
                controls += 1
                interventions += value
    return controls, interventions


def run(execution):
    execution = Path(execution).resolve()
    root, frozen = execution.parent.parent, execution / 'source'
    if execution.parent.name != 'runs' or not (root / '.git').exists():
        raise ValueError('Execution must be inside the project runs directory')
    manifest = read(execution / 'source_manifest.json')
    plan = read(execution / 'plan.json')
    b1_profile = plan.get('B1_cohort_profile', 'original_four_plus_seed11')
    if b1_profile not in ('original_four_plus_seed11', 'all_five_matured_v1'):
        raise ValueError('Unknown explicit B1 cohort profile')
    if [r['seed'] for r in plan['H_models']] != [11, 22, 33, 44, 55] or not 60 <= plan['wait_seconds'] <= 86400:
        raise ValueError('Require all five paired seeds and a bounded dependency wait')

    def verify():
        for name, digest in manifest.items():
            path = (frozen / name).resolve()
            if not path.is_relative_to(frozen) or sha(path) != digest:
                raise ValueError('Frozen orchestration source changed: ' + name)
        if sha(root / 'scripts/carla_bridge37.py') != plan['workspace_bridge_sha256']:
            raise ValueError('Prepared walker-initialization bridge changed')

    verify()
    sys.path.insert(0, str(frozen))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json
    from scripts import prepare_native_shared_brake as preparation
    from scripts.evaluate_native_rule import require_finished_parent
    preparation.ROOT = root
    completed, owner = [], None

    def status(phase, **extra):
        atomic_json(execution / 'status.json', {'phase': phase, 'epoch': time.time(),
                    'pid': os.getpid(), 'completed': completed, **extra})

    def call(command, log, timeout, environment=None):
        with (execution / log).open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(list(map(str, command)), cwd=root, env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
            atomic_json(execution / 'last_child.json', {'pid': child.pid, 'log': log,
                                                       'epoch': time.time(), 'timeout_seconds': timeout})
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError('Observation expired;native wrapper/watchdog retains ownership PID' + str(child.pid))
            if code:
                raise RuntimeError('Registered child failed;no retry:' + log)

    try:
        with ProcessLease(execution / 'pipeline.lock'):
            with (execution / 'started.json').open('x') as stream:
                json.dump({'pid': os.getpid(), 'epoch': time.time(), 'script_sha256': sha(Path(__file__)),
                           'plan_sha256': sha(execution / 'plan.json')}, stream)
            dependency = root / 'runs' / plan['dependency_execution']
            expected = ('ALL_FIVE_MATURED_B1_PPO_SEEDS_COMPLETE' if b1_profile == 'all_five_matured_v1'
                        else 'ALL_FIVE_REGISTERED_B2_PPO_SEEDS_COMPLETE')
            until = time.monotonic() + plan['wait_seconds']
            while read(dependency / 'status.json')['phase'] != expected:
                if (dependency / 'failure.json').exists() or time.monotonic() >= until:
                    raise RuntimeError('Predecessor failed or bounded wait exhausted;unactivated probes retained')
                if owner is None:
                    identity = plan['dependency_process']
                    owner = OwnedProcess(identity['pid'], identity['creation_epoch'], identity['executable'], allow_terminate=False)
                if owner.poll() is not None:
                    if read(dependency / 'status.json')['phase'] == expected:
                        break
                    raise RuntimeError('Predecessor exited before completion')
                status('WAITING_FOR_REGISTERED_B1_COHORT' if b1_profile == 'all_five_matured_v1'
                       else 'WAITING_FOR_REGISTERED_B2_COHORT', dependency_pid=owner.pid,
                       dependency_live_via_retained_handle=True)
                time.sleep(10)
            if (dependency / 'failure.json').exists():
                raise RuntimeError('Predecessor failure evidence retained')
            b2 = root / 'runs' / plan['B2_execution'] if b1_profile == 'all_five_matured_v1' else dependency
            if (b2 / 'failure.json').exists() or read(b2 / 'status.json')['phase'] != 'ALL_FIVE_REGISTERED_B2_PPO_SEEDS_COMPLETE':
                raise ValueError('Complete B2 predecessor is required')
            prior = read(b2 / 'status.json')['completed']
            if [r['seed'] for r in prior] != [11, 22, 33, 44, 55]:
                raise ValueError('Predecessor seed inventory differs')
            for row in prior:
                require_finished_parent(root / 'runs' / row['batch'])
            b1 = read(root / 'runs' / plan['B1_execution'] / 'status.json')
            b1_parents = b1_parent_map(b1, b1_profile, plan.get('B1_seed11'))
            parents = {(r['seed'], 'H'): r['batch'] for r in plan['H_models']}
            parents.update({(seed, 'B1'): batch for seed, batch in b1_parents.items()})
            reference = read(root / 'runs' / plan['H_models'][0]['batch'] / 'protocol.json')
            for seed in (11, 22, 33, 44, 55):
                for method in ('H', 'B1'):
                    verify()
                    parent = root / 'runs' / parents[(seed, method)]
                    protocol = read(parent / 'protocol.json')
                    if protocol['seed'] != seed or any(protocol.get(k) != reference.get(k) for k in
                            ('routes', 'native_case_manifest', 'native_validation', 'calibration_sha256', 'episode_seconds', 'road_geometry_version')):
                        raise ValueError('Safety comparison task or calibration differs')
                    for name, digest in plan['physical_source_hashes'].items():
                        if sha(parent / 'source' / name) != digest:
                            raise ValueError('Safety comparison physical source differs: ' + name)
                    entries = [r for r in plan['preprepared'] if r['seed'] == seed and r['method'] == method]
                    if not entries:
                        status('PREPARING_PAIRED_FROZEN_PROBES', seed=seed, method=method)
                        entries = preparation.prepare(parent, method)
                    if [r['enabled'] for r in entries] != [False, True]:
                        raise ValueError('Both native-preserving OFF/ONarms are required')
                    for row in entries:
                        verify()
                        batch = root / 'runs' / row['batch']
                        if sha(batch / 'protocol.json') != row['protocol_sha256']:
                            raise ValueError('Prepared safety probe changed')
                        probe = read(batch / 'protocol.json')
                        if (probe['methods'] != [method] or probe['seed'] != seed
                                or probe.get('evaluation_only') is not True
                                or probe['method_budgets'] != {method: 0}
                                or probe['deployment_shared_brake']['enabled'] is not row['enabled']):
                            raise ValueError('Probe model,zero-update budget or brake state differs')
                        mode = 'ON' if row['enabled'] else 'OFF'
                        prefix = 'seed' + str(seed) + '_' + method + '_' + mode
                        status('RUNNING_FROZEN_SHARED_BRAKE_PROBE', seed=seed, method=method, mode=mode, batch=batch.name)
                        call([shutil.which('pwsh'), '-NoProfile', '-File', frozen / 'scripts/run_registered_native.ps1',
                              '-Batch', batch, '-Python', root / '.venv/Scripts/python.exe',
                              '-CarlaPython', root / '.carla-python37/runtime/python.exe', '-ActivateQueued', '-UseCurrentGuard'],
                             prefix + '_lifecycle.log', 4200)
                        require_finished_parent(batch)
                        env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'))
                        inference = batch / method / 'measured_start_inference.json'
                        call([sys.executable, frozen / 'scripts/audit_native_method_endpoint.py', '--batch', batch, '--method', method],
                             prefix + '_raw.log', 300, env)
                        call([sys.executable, frozen / 'scripts/audit_native_policy_inference.py', '--batch', batch, '--method', method,
                              '--output', inference], prefix + '_inference.log', 300, env)
                        folder = batch / method
                        summary, raw = read(folder / 'summary.json'), read(folder / 'endpoint_raw_audit.json')
                        episodes = read(folder / 'evaluation.json')['episodes']
                        if summary['physics_steps'] or summary['optimizer_steps'] or summary['actor_l2_change'] != 0:
                            raise ValueError('Frozen deployment probe unexpectedly trained')
                        controls, shield = shared_trace_counts(folder / 'evaluation_trace.jsonl.gz')
                        count = len(reference['native_case_manifest']['cases'])
                        if (controls != raw['evaluation_control_frames_verified']
                                or sorted(e['case_id'] for e in episodes) != list(range(count))
                                or len(read(inference)['results']) != count):
                            raise ValueError('Incomplete paired scene/control inventory')
                        result = {'scope': 'Frozen-policy DEVELOPMENT deployment intervention;not retrained safety adaptation or final confirmation',
                                  'seed': seed, 'method': method, 'enabled': row['enabled'], 'batch': batch.name,
                                  'cases': count, 'successes': sum(e['success'] for e in episodes),
                                  'evaluation_controls_verified': controls, 'shared_brake_control_interventions': shield,
                                  'new_training_controls': 0, 'new_optimizer_updates': 0, 'actor_l2_change': 0,
                                  'model_sha256': sha(folder / 'model.zip'), 'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'),
                                  'episodes': episodes, 'research_acceptance': 'UNMET'}
                        with (root / 'reports' / ('native_shared_brake_' + prefix + '_20261031.json')).open('x') as stream:
                            json.dump(result, stream, indent=2)
                        completed.append({'seed': seed, 'method': method, 'enabled': row['enabled'], 'batch': batch.name})
                        status('SHARED_BRAKE_PROBE_COMPLETE', seed=seed, method=method, mode=mode)
            status('ALL_TWENTY_PAIRED_SHARED_BRAKE_PROBES_COMPLETE')
    except BaseException as error:
        with (execution / 'failure.json').open('x') as stream:
            json.dump({'epoch': time.time(), 'error': repr(error), 'completed': completed,
                       'scope': 'No experiment retries,seed omissions or process kills;all artifacts retained'}, stream, indent=2)
        raise
    finally:
        if owner is not None:
            owner.kernel.CloseHandle(owner.handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution', required=True)
    run(parser.parse_args().execution)

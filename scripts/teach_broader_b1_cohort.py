"""Wait for the registered TRAIN teacher, then fit every fixed B1 endpoint once."""
import argparse
from datetime import datetime, timezone
import json
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
from scripts.fit_broader_measured_actor import read, sha

TEACHER_EXECUTION = ROOT / 'runs/B3_broader_TRAIN_teacher_execution_v3_20261102'
COHORT = ROOT / 'runs/B1_additional_broader_cohort_20261102'
PILOT = ROOT / 'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO'


def prepare(output, completed_teacher=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the existing registered teaching observer')
    dependency = read(TEACHER_EXECUTION / 'status.json')
    if completed_teacher:
        if ((TEACHER_EXECUTION / 'failure.json').exists()
                or dependency['phase'] != 'ALL36_TRAIN_TEACHER_AND_DIRECT_VIEWS_COMPLETE'
                or not (TEACHER_EXECUTION / 'direct_views/manifest.json').is_file()):
            raise ValueError('Require the completed audited teacher and measured dataset')
        identity = None
    else:
        # Query Windows directly; do not add an optional process-monitor dependency.
        query = (f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={int(dependency['pid'])}';"
             "if (-not $p) { throw 'Dependency process missing' };"
             "[pscustomobject]@{pid=$p.ProcessId;executable=$p.ExecutablePath;"
             "creation_epoch=([DateTimeOffset]$p.CreationDate).ToUnixTimeMilliseconds()/1000.0;"
             "command=$p.CommandLine}|ConvertTo-Json -Compress")
        identity = json.loads(subprocess.check_output([shutil.which('pwsh'), '-NoProfile', '-Command', query],
                                                 text=True, encoding='utf-8'))
        command = identity.pop('command')
        if ('collect_broader_train_teacher.py' not in command or TEACHER_EXECUTION.name not in command
            or '--output' not in command or '--run' not in command):
            raise ValueError('Dependency is not the registered TRAIN-only teacher observer')
        owner = OwnedProcess(identity['pid'], identity['creation_epoch'], identity['executable'], allow_terminate=False)
        try:
            if owner.poll() is not None or dependency['phase'] != 'WAITING_FOR_EXACT_BROADER_B1_COHORT_EXIT':
                raise ValueError('Register against the actual waiting TRAIN-only teacher observer')
        finally:
            owner.kernel.CloseHandle(owner.handle)
    parents = read(COHORT / 'registration.json')['rows']
    parents = sorted([dict(seed=r['seed'], batch=str(ROOT / 'runs' / r['batch'])) for r in parents]
                     + [dict(seed=22, batch=str(PILOT))], key=lambda r: r['seed'])
    if [r['seed'] for r in parents] != [11, 22, 33, 44, 55]:
        raise ValueError('Keep every original independent parent, including failed basic gates')
    output.mkdir()
    source = output / 'fit_source'
    source.mkdir()
    shutil.copy2(ROOT / 'scripts/fit_broader_measured_actor.py', source / 'fit_broader_measured_actor.py')
    shutil.copy2(__file__, output / 'executed_observer.py')
    atomic_json(output / 'registration.json', dict(version='broader_measured_actor_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), workspace=str(ROOT), parents=parents,
        teacher_execution=str(TEACHER_EXECUTION), dependency=identity,
        completed_teacher_adoption=completed_teacher, teacher_status_sha256=sha(TEACHER_EXECUTION / 'status.json') if completed_teacher else None,
        teacher_batch=read(TEACHER_EXECUTION / 'registration.json')['batch'],
        script_sha256=sha(source / 'fit_broader_measured_actor.py'),
        steps=1024, learning_rate=1e-4, maximum_fit_wall_seconds_each=600,
        maximum_total_wall_seconds=43200, automatic_retries=0,
        supervision='Only successful TRAIN teacher episodes; all36case outcomes retained. No VAL/final labels or teacher value targets.',
        holdout='Original every-fifth-frame correlated diagnostic; no gradient or checkpoint selection.',
        frozen='Each fixed final independent51200-control model; retain critic/log_std/PPOAdam/counters/RMS/RNG. Actor-only extra supervision,not PPO.',
        native_rollouts_started=0, model_fitting_started=False,
        followup='No automatic native promotion. Review allfive fits before a separately registered learned-policy driving experiment.',
        research_acceptance='UNMET'))


def run(output):
    output = Path(output).resolve()
    if (output / 'started.json').exists():
        raise FileExistsError('Never repeat a started teaching observer')
    registration = read(output / 'registration.json')
    teacher_execution = Path(registration['teacher_execution'])
    identity = registration['dependency']
    dependency = None if registration.get('completed_teacher_adoption') else OwnedProcess(identity['pid'], identity['creation_epoch'], identity['executable'], allow_terminate=False)
    if dependency is not None and dependency.poll() is not None:
        dependency.kernel.CloseHandle(dependency.handle)
        raise ValueError('Teacher observer ended before adoption; inspect before continuing')
    atomic_json(output / 'started.json', dict(pid=os.getpid(), epoch=time.time(), dependency=identity))
    deadline = time.monotonic() + 43200
    def status(phase, **extra):
        atomic_json(output / 'status.json', dict(phase=phase, pid=os.getpid(), epoch=time.time(), **extra))
    try:
        with ProcessLease(output / 'observer.lock'):
            while dependency is not None and dependency.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('Finite teacher dependency wait exhausted')
                status('WAITING_FOR_EXACT_TRAIN_TEACHER_EXIT')
                time.sleep(10)
            if (registration.get('completed_teacher_adoption') and sha(teacher_execution / 'status.json') != registration['teacher_status_sha256']):
                raise ValueError('Completed teacher status changed after adoption')
            if ((teacher_execution / 'failure.json').exists()
                    or read(teacher_execution / 'status.json')['phase'] != 'ALL36_TRAIN_TEACHER_AND_DIRECT_VIEWS_COMPLETE'):
                raise ValueError('Require fully audited TRAIN teacher and exact measured views')
            plan = {**registration, 'teacher': str(ROOT / 'runs' / registration['teacher_batch']),
                'views': str(teacher_execution / 'direct_views'),
                'reference_config': str(PILOT / 'B1/config.json'),
                'manifest_sha256': sha(teacher_execution / 'direct_views/manifest.json'),
                'strict_delivery_sha256': sha(teacher_execution / 'strict_delivery.json')}
            for row in plan['parents']:
                batch = Path(row['batch'])
                endpoint = batch / 'B1' / read(batch / 'B1/latest_checkpoint.json')['folder']
                row['model_sha256'] = read(endpoint / 'metadata.json')['model_sha256']
            atomic_json(output / 'bound_plan.json', plan)
            script = output / 'fit_source/fit_broader_measured_actor.py'
            completed = []
            for row in plan['parents']:
                okay, reason = resources_ok(ROOT)
                if not okay or time.monotonic() > deadline or sha(script) != registration['script_sha256']:
                    raise RuntimeError('Frozen fit resource/source/deadline guard: ' + str(reason))
                seed = row['seed']
                status('TRAIN_ONLY_ACTOR_FIT_RUNNING', seed=seed, completed=completed)
                with (output / f'seed{seed}.log').open('xb') as log:
                    child = subprocess.Popen([sys.executable, str(script), '--plan', str(output / 'bound_plan.json'),
                        '--seed', str(seed), '--output', str(output / f'seed{seed}')], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT)
                    atomic_json(output / 'last_child.json', dict(pid=child.pid, seed=seed, epoch=time.time()))
                    try:
                        code = child.wait(timeout=min(660, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        # Popen owns this exact CPU-only process handle, never the CARLA dependency.
                        child.kill()
                        child.wait(timeout=30)
                        raise
                if code:
                    raise RuntimeError(f'Actor fit failed for seed{seed}; no retry or partial promotion')
                completed.append(dict(seed=seed, result_sha256=sha(output / f'seed{seed}/complete.json')))
            status('ALL_FIVE_TRAIN_ONLY_ACTOR_FITS_COMPLETE', completed=completed, new_PPO_updates=0, native_rollouts=0)
    except BaseException as error:
        atomic_json(output / 'failure.json', dict(error=repr(error), epoch=time.time()))
        status('STOPPED_WITH_FAILURE_NO_RETRY')
        raise
    finally:
        if dependency is not None:
            dependency.kernel.CloseHandle(dependency.handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--teacher-execution', type=Path)
    parser.add_argument('--cohort-execution', type=Path)
    parser.add_argument('--completed-teacher', action='store_true', help='Explicit adoption of the completed audited data;no waiting-process restart')
    args = parser.parse_args()
    if args.teacher_execution is not None:
        TEACHER_EXECUTION = args.teacher_execution.resolve()
    if args.cohort_execution is not None:
        COHORT = args.cohort_execution.resolve()
    run(args.output) if args.run else prepare(args.output, args.completed_teacher)

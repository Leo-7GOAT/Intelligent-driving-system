"""Register four equal-budget development courses and wait for the exact running seed22 pilot."""
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
from navstack.owned_process import OwnedProcess
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata

PILOT = ROOT / 'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO'
DEPENDENCY = ROOT / 'runs/B1_basic_seed55_recovery_execution_20261102'


def prepare(output):
    output = Path(output).resolve()
    if output.exists() or (DEPENDENCY / 'broader_learning_audit.json').exists():
        raise FileExistsError('Register before observing pilot validation; preserve any previous preparation')
    cohort = read(ROOT / 'reports/native_B1_shared_parking_complete_basic_cohort_20261102.json')
    template = read(PILOT / 'protocol.json')
    if read(DEPENDENCY / 'status.json')['phase'] != 'BROADER_PILOT_ACTUAL_PPO_RUNNING':
        raise ValueError('Require the running, not completed, original development pilot')
    for name, digest in template['source_hashes'].items():
        if sha(PILOT / 'source' / name) != digest:
            raise ValueError('Pilot frozen source changed')
    okay, reason = resources_ok(ROOT)
    if not okay:
        raise RuntimeError(str(reason))
    output.mkdir()
    rows = []
    for seed in (11, 33, 44, 55):
        record = next(r for r in cohort['seeds'] if r['seed'] == seed)
        parent = ROOT / 'runs' / record['endpoint_batch']
        require_finished_parent(parent)
        old = read(parent / 'protocol.json')
        origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
        metadata = verified_checkpoint_metadata(origin)
        if (metadata['model_sha256'] != record['model_sha256'] or metadata['lineage_control_steps'] != 30720
                or read(origin / 'config.json')['signature'] != read(PILOT / 'resume/B1/config.json')['signature']):
            raise ValueError('Require each fixed final independent endpoint and identical environment semantics')
        for name, digest in template['broader_B1_PPO']['preserved_low_level_source_hashes'].items():
            if old['source_hashes'][name] != digest:
                raise ValueError('Common physical/low-level implementation differs')
        batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + f'_native_broader_B1_seed{seed}')
        batch.mkdir()
        (batch / 'infrastructure').mkdir()
        for name in ('source', 'calibration'):
            shutil.copytree(PILOT / name, batch / name, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(origin, batch / 'resume/B1')
        shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
        p = deepcopy(template)
        p.update(seed=seed, parent_batch=parent.name,
            scope='Prospectively registered additional development seed, same36TRAIN/24VAL and20480newPPOcontrols as original22pilot. All seeds retained; not matched H/B1 or final evidence.')
        p['extra_supervision'] = deepcopy(old['extra_supervision'])
        summary, raw = read(parent / 'B1/summary.json'), read(parent / 'B1/endpoint_raw_audit.json')
        p['recovery']['B1'] = dict(checkpoint='resume/B1', parent_batch=parent.name,
            parent_checkpoint='B1/' + origin.name, lineage_controls=30720,
            consumed_controls_lower_bound=summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'],
            retained_optimizer_steps=old['recovery']['B1']['retained_optimizer_steps'] + summary['optimizer_steps'],
            unknown_unflushed_parent_tail=bool(old['recovery']['B1'].get('unknown_unflushed_parent_tail', False)),
            recovery_chain=old['recovery']['B1'].get('recovery_chain', []) + [parent.name],
            model_sha256=metadata['model_sha256'], normalization_sha256=metadata['normalization_sha256'])
        spec = p['broader_B1_PPO']
        spec.update(source_model_sha256=metadata['model_sha256'], normalization_sha256=metadata['normalization_sha256'],
            parent_basic_audit_sha256=record['model_sha256'],
            selection='Every original seed fixed final endpoint, including failed basic gates11/55. No checkpoint/seed selection.',
            limits='Additional development course. Original22pilot remains separately registered and unmodified;four other seeds committed before its validation. Independent warm starts and55interruption retained. No matched H/B1 budget or final confirmation claim.')
        # This field now explicitly names its actual provenance, not an audit hash.
        spec.pop('parent_basic_audit_sha256')
        spec['parent_basic_model_sha256'] = record['model_sha256']
        now = time.time()
        p.update(created_epoch=now, deadline_epoch=now + 7200)
        atomic_json(batch / 'protocol.json', p)
        atomic_json(batch / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(batch / 'protocol.json')))
        rows.append(dict(seed=seed, batch=batch.name, protocol_sha256=sha(batch / 'protocol.json'), initial_model_sha256=metadata['model_sha256']))
    atomic_json(output / 'registration.json', dict(rows=rows, original_pilot=PILOT.name,
        original_pilot_source_model_sha256=template['broader_B1_PPO']['source_model_sha256'],
        all_seeds=[11, 22, 33, 44, 55], requested_additional_controls=81920,
        controls_each=20480, training_conditions=36, known_validation_conditions=24,
        original_basic_cohort_sha256=sha(ROOT / 'reports/native_B1_shared_parking_complete_basic_cohort_20261102.json'),
        purpose='Complete five development policies using unchanged common training after independent basic courses;retain basic failures.',
        limits='Prepared before22pilot validation;not a fromscratch or matched H/B1 comparison. No final conditions.',
        new_controls_so_far=0, automatic_retries=0))
    print('Four courses prepared; zero new controls', flush=True)


def run(output):
    output = Path(output).resolve()
    if (output / 'started.json').exists():
        raise FileExistsError('Never relaunch an existing observer')
    registration = read(output / 'registration.json')
    validation = read(output / 'validation.json')
    if ([r['seed'] for r in validation] != [11, 33, 44, 55]
            or any(not r['frozen_import_and_config_signature'] or not r['saved_PPO_hyperparameters_match_original22']
                   or not r['launcher_inputs_validated_not_server_probed'] for r in validation)):
        raise ValueError('Every frozen course must pass its actual import/config/PPO/launcher preflight')
    identity = read(output / 'dependency_identity.json')
    owner = OwnedProcess(identity['pid'], identity['creation_epoch'], identity['executable'], allow_terminate=False)
    if owner.poll() is not None:
        raise ValueError('Dependency already ended; inspect completed evidence before recovery')
    source = output / 'source'
    for directory in ('scripts', 'navstack'):
        shutil.copytree(ROOT / directory, source / directory, ignore=shutil.ignore_patterns('__pycache__'))
    hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    atomic_json(output / 'source_manifest.json', hashes)
    atomic_json(output / 'started.json', dict(pid=os.getpid(), epoch=time.time(), dependency=identity, automatic_retries=0))
    deadline = time.monotonic() + 43200
    completed = []
    def status(phase, **extra):
        atomic_json(output / 'status.json', dict(phase=phase, pid=os.getpid(), epoch=time.time(), completed=completed, **extra))
    def call(args, label, cap, env=None):
        for name, digest in hashes.items():
            if sha(source / name) != digest:
                raise ValueError('Observer source changed')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Overall observer deadline exhausted before child launch')
        with (output / (label + '.log')).open('xb') as log:
            child = subprocess.Popen(list(map(str, args)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(output / 'last_child.json', dict(pid=child.pid, label=label, epoch=time.time()))
            code = child.wait(timeout=min(cap, remaining))
        if code:
            raise RuntimeError('Child failed without retry: ' + label)
    try:
        with ProcessLease(output / 'observer.lock'):
            while owner.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('Finite dependency wait exhausted')
                status('WAITING_FOR_EXACT_ORIGINAL_PILOT_OBSERVER')
                time.sleep(10)
            if (read(DEPENDENCY / 'status.json')['phase'] != 'RECOVERY_AND_BROADER_PILOT_COMPLETE'
                    or (DEPENDENCY / 'failure.json').exists()):
                raise ValueError('Original pilot must complete its full audit; no infrastructure retry')
            for row in registration['rows']:
                okay, reason = resources_ok(ROOT)
                if not okay:
                    raise RuntimeError(str(reason))
                batch = ROOT / 'runs' / row['batch']
                if sha(batch / 'protocol.json') != row['protocol_sha256']:
                    raise ValueError('Queued protocol changed')
                prefix = 'seed' + str(row['seed'])
                status('ADDITIONAL_BROADER_ACTUAL_PPO_RUNNING', seed=row['seed'], batch=batch.name)
                call([shutil.which('pwsh'), '-NoProfile', '-File', source / 'scripts/run_registered_native.ps1',
                    '-Batch', batch, '-Python', ROOT / '.venv/Scripts/python.exe', '-CarlaPython', ROOT / '.carla-python37/runtime/python.exe',
                    '-ActivateQueued', '-UseCurrentGuard'], prefix + '_lifecycle', 7800)
                require_finished_parent(batch)
                env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'), PYTHONIOENCODING='utf-8')
                call([sys.executable, source / 'scripts/audit_native_method_endpoint.py', '--batch', batch, '--method', 'B1'], prefix + '_raw', 600, env)
                call([sys.executable, source / 'scripts/audit_native_policy_inference.py', '--batch', batch, '--method', 'B1',
                    '--output', batch / 'B1/measured_start_inference.json'], prefix + '_starts', 600, env)
                audit = output / (prefix + '_learning_audit.json')
                call([sys.executable, source / 'scripts/analyze_native_broader_b1_course.py', '--batch', batch,
                    '--output', audit], prefix + '_learning', 600)
                completed.append(dict(seed=row['seed'], batch=batch.name, audit_sha256=sha(audit)))
            status('ALL_FOUR_ADDITIONAL_BROADER_COURSES_AUDITED', original_pilot_audit_sha256=sha(DEPENDENCY / 'broader_learning_audit.json'))
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

"""Resume disk-interrupted seed44, then execute the untouched registered seed55."""
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
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.prepare_native_evaluation_recovery import verified_interruption
from scripts.prepare_native_recovery import verified_checkpoint_metadata, claim_recovery
from scripts.evaluate_native_rule import require_finished_parent

PARENT = ROOT / 'runs/20261102T044004788200Z_native_broader_B1_seed44'
ORIGINAL = ROOT / 'runs/B1_additional_broader_cohort_20261102'


def prepare(output):
    output = Path(output).resolve()
    if output.exists() or (PARENT / 'recovery_registration.json').exists():
        raise FileExistsError('Preserve the registered recovery')
    if verified_interruption(PARENT) != 'worker_disk_below_12GiB':
        raise ValueError('Require the actual owned disk-interrupted lifecycle')
    prefix = read(PARENT / 'verified_saved_prefix.json')
    old = read(PARENT / 'protocol.json')
    checkpoint = PARENT / 'B1' / read(PARENT / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(checkpoint)
    if (prefix['verified_controls'] != 11520 or prefix['saved_controls'] != 11520
            or prefix['all_engine_controls_acknowledged'] is not True
            or prefix['checkpoint_sha256'] != metadata['model_sha256']
            or metadata['physics_steps'] != 11520 or metadata['optimizer_steps'] != 360
            or metadata['lineage_control_steps'] != 42240 or old['seed'] != 44):
        raise ValueError('Saved physical/learning prefix differs')
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('Restore at least20GiB free before this explicit recovery')
    for name, digest in old['source_hashes'].items():
        if sha(PARENT / 'source' / name) != digest:
            raise ValueError('Frozen learning/physical source changed')
    original = read(ORIGINAL / 'registration.json')
    completed = read(ORIGINAL / 'status.json')['completed']
    if [r['seed'] for r in completed] != [11, 33]:
        raise ValueError('Never replay either completed broader course')
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_broader_B1_seed44_recovery')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    for name in ('source', 'calibration'):
        shutil.copytree(PARENT / name, batch / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(checkpoint, batch / 'resume/B1')
    shutil.copy2(PARENT / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    p = deepcopy(old)
    prior = p['recovery']['B1']
    now = time.time()
    p.update(parent_batch=PARENT.name, created_epoch=now, deadline_epoch=now + 7200,
        method_budgets={'B1': 8960}, controlled_budget_per_method=8960,
        checkpoint_thresholds=list(range(1280, 8961, 1280)), lineage_targets={'B1': 51200})
    p['recovery']['B1'] = dict(checkpoint='resume/B1', parent_batch=PARENT.name,
        parent_checkpoint='B1/' + checkpoint.name, lineage_controls=42240,
        consumed_controls_lower_bound=max(metadata['charged_control_steps_lower_bound'],
            prior['consumed_controls_lower_bound'] + 11520), unknown_unflushed_parent_tail=True,
        retained_optimizer_steps=prior['retained_optimizer_steps'] + 360,
        model_sha256=metadata['model_sha256'], normalization_sha256=metadata['normalization_sha256'],
        recovery_chain=prior.get('recovery_chain', []) + [PARENT.name])
    p['broader_B1_PPO'].update(new_actual_PPO_controls=8960, source_model_sha256=metadata['model_sha256'],
        limits='Explicit resource recovery:11520saved original controls plus8960remaining retain the original20480course.360savedAdamsteps retained;new measured episode,not an uninterrupted replicate. No new labels or final conditions.')
    p['broader_course_recovery'] = dict(original_protocol_sha256=sha(PARENT / 'protocol.json'),
        saved_prefix_sha256=sha(PARENT / 'verified_saved_prefix.json'), saved_original_controls=11520,
        saved_original_Adam_steps=360, original_target_controls=20480, resumed_controls=8960,
        source_and_physical_bytes_unchanged=True, interruption='worker_and_guard_disk_below_12GiB',
        unknown_unflushed_tail_retained=True)
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(batch / 'protocol.json')))
    claim_recovery(PARENT, batch.name)
    rows = deepcopy(original['rows'])
    row = next(r for r in rows if r['seed'] == 44)
    row.update(batch=batch.name, protocol_sha256=sha(batch / 'protocol.json'), initial_model_sha256=metadata['model_sha256'], recovery_disclosed=True)
    output.mkdir()
    atomic_json(output / 'registration.json', dict(rows=rows, retained_completed=completed,
        original_execution=ORIGINAL.name, recovered_seed44=batch.name, pending_seed55=rows[-1]['batch'],
        saved_controls=11520, remaining_controls=8960, initial_C_free_bytes=shutil.disk_usage(ROOT).free,
        automatic_retries=0, new_PPO_controls_requested=8960 + 20480, final_conditions=0))
    print(batch, flush=True)


def run(output):
    output = Path(output).resolve()
    if (output / 'started.json').exists():
        raise FileExistsError('Never relaunch this started recovery observer')
    registration = read(output / 'registration.json')
    source = output / 'source'
    for directory in ('scripts', 'navstack'):
        shutil.copytree(ROOT / directory, source / directory, ignore=shutil.ignore_patterns('__pycache__'))
    hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    atomic_json(output / 'source_manifest.json', hashes)
    atomic_json(output / 'started.json', dict(pid=os.getpid(), epoch=time.time(), automatic_retries=0))
    completed = registration['retained_completed'][:]
    deadline = time.monotonic() + 21600
    def status(phase, **extra):
        atomic_json(output / 'status.json', dict(phase=phase, pid=os.getpid(), epoch=time.time(), completed=completed, **extra))
    def call(args, label, cap, env=None):
        if any(sha(source / name) != digest for name, digest in hashes.items()):
            raise ValueError('Frozen recovery observer changed')
        with (output / (label + '.log')).open('xb') as log:
            child = subprocess.Popen(list(map(str, args)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            atomic_json(output / 'last_child.json', dict(pid=child.pid, label=label, epoch=time.time()))
            code = child.wait(timeout=min(cap, max(0, deadline - time.monotonic())))
        if code:
            raise RuntimeError('Child failed without retry: ' + label)
    try:
        for row in registration['rows']:
            if row['seed'] in (11, 33):
                continue
            okay, reason = resources_ok(ROOT)
            if not okay or shutil.disk_usage(ROOT).free < 18 * 1024**3:
                raise RuntimeError('Recovery requires18GiB free at dispatch: ' + str(reason))
            batch = ROOT / 'runs' / row['batch']
            if sha(batch / 'protocol.json') != row['protocol_sha256']:
                raise ValueError('Registered recovery/untouched55protocol changed')
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
            call([sys.executable, source / 'scripts/analyze_native_broader_b1_course.py', '--batch', batch, '--output', audit], prefix + '_learning', 600)
            completed.append(dict(seed=row['seed'], batch=batch.name, audit_sha256=sha(audit), recovery_disclosed=row.get('recovery_disclosed', False)))
        status('ALL_FOUR_ADDITIONAL_BROADER_COURSES_AUDITED', seed44_interruption_disclosed=True)
    except BaseException as error:
        atomic_json(output / 'failure.json', dict(error=repr(error), epoch=time.time()))
        status('STOPPED_WITH_FAILURE_NO_RETRY')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    (run if args.run else prepare)(args.output)

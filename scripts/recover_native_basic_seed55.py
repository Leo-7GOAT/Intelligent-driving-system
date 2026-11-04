"""Recover the externally interrupted fifth basic course without replaying four completed seeds."""
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

PARENT = ROOT / 'runs/20261101T121900945085Z_native_B1_shared_parking_cohort_seed55'
QUEUE = ROOT / 'runs/B1_shared_parking_five_seed_execution_20261101'


def prepare(output):
    output = Path(output).resolve()
    if output.exists() or (PARENT / 'recovery_registration.json').exists():
        raise FileExistsError('Preserve existing recovery; never register a duplicate')
    verified_interruption(PARENT)
    okay, reason = resources_ok(ROOT)
    if not okay:
        raise RuntimeError(str(reason))
    old = read(PARENT / 'protocol.json')
    audit = read(PARENT / 'interruption_audit.json')['methods']['B1']
    prior = old['recovery']['B1']
    origin = PARENT / 'B1' / read(PARENT / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    if (old['seed'] != 55 or metadata['physics_steps'] != 3840 or metadata['optimizer_steps'] != 120
            or metadata['lineage_control_steps'] != 24320 or audit['state'] != 'interrupted'
            or audit['consumed_controls_lower_bound'] < 4498):
        raise ValueError('Expected interrupted course/checkpoint changed')
    state = read(QUEUE / 'status.json')
    if [r['seed'] for r in state['completed']] != [11, 22, 33, 44]:
        raise ValueError('Retain the four already completed independent courses')
    for name, digest in old['source_hashes'].items():
        if sha(PARENT / 'source' / name) != digest:
            raise ValueError('Frozen learning source changed')
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_B1_basic_seed55_recovery')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    for name in ('source', 'calibration'):
        shutil.copytree(PARENT / name, batch / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(origin, batch / 'resume/B1')
    shutil.copy2(PARENT / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    p = deepcopy(old)
    now = time.time()
    p.update(created_epoch=now, deadline_epoch=now + 7200, method_budgets={'B1': 6400},
        controlled_budget_per_method=6400, checkpoint_thresholds=list(range(1280, 6401, 1280)),
        parent_batch=PARENT.name, lineage_targets={'B1': 30720})
    p['recovery']['B1'] = dict(checkpoint='resume/B1', parent_batch=PARENT.name,
        parent_checkpoint='B1/' + origin.name, lineage_controls=24320,
        consumed_controls_lower_bound=max(metadata['charged_control_steps_lower_bound'],
            prior['consumed_controls_lower_bound'] + audit['consumed_controls_lower_bound']),
        unknown_unflushed_parent_tail=True, retained_optimizer_steps=prior['retained_optimizer_steps'] + 120,
        model_sha256=metadata['model_sha256'], normalization_sha256=metadata['normalization_sha256'],
        recovery_chain=[PARENT.name])
    p['shared_parking_PPO'].update(source_model_sha256=metadata['model_sha256'],
        normalization_sha256=metadata['normalization_sha256'], new_PPO_controls=6400,
        limits='Recovery segment only:3840 saved original updates plus6400 resumed controls complete the original10240 budget. Discarded658+ controls and unknown tail charged separately. Not an uninterrupted replicate.')
    p['basic_course_recovery'] = dict(original_protocol_sha256=sha(PARENT / 'protocol.json'),
        interruption_audit_sha256=sha(PARENT / 'interruption_audit.json'), saved_original_controls=3840,
        original_consumed_controls_lower_bound=audit['consumed_controls_lower_bound'],
        original_target_controls=10240, resumed_controls=6400, new_supervised_updates=0,
        source_and_low_level_bytes_unchanged=True)
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(batch / 'protocol.json')))
    claim_recovery(PARENT, batch.name)
    output.mkdir()
    atomic_json(output / 'registration.json', dict(batch=batch.name, protocol_sha256=sha(batch / 'protocol.json'),
        parent=PARENT.name, saved_controls=3840, remaining_controls=6400, automatic_retries=0))
    print(batch, flush=True)


def run(output):
    output = Path(output).resolve()
    registration = read(output / 'registration.json')
    if (output / 'started.json').exists():
        raise FileExistsError('Inspect the existing recovery; never relaunch it')
    batch = ROOT / 'runs' / registration['batch']
    if sha(batch / 'protocol.json') != registration['protocol_sha256']:
        raise ValueError('Registered protocol changed')
    source = output / 'source'
    for name in ('scripts', 'navstack'):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__'))
    hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    atomic_json(output / 'source_manifest.json', hashes)
    atomic_json(output / 'started.json', dict(pid=os.getpid(), epoch=time.time(), automatic_retries=0))
    def status(phase):
        atomic_json(output / 'status.json', dict(phase=phase, pid=os.getpid(), epoch=time.time(), batch=batch.name))
    def call(args, label, timeout, env=None):
        for name, digest in hashes.items():
            if sha(source / name) != digest:
                raise ValueError('Frozen recovery observer changed')
        with (output / (label + '.log')).open('xb') as log:
            child = subprocess.Popen(list(map(str, args)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            atomic_json(output / 'last_child.json', dict(pid=child.pid, label=label, epoch=time.time()))
            code = child.wait(timeout=timeout)
        if code:
            raise RuntimeError('Child failed without retry: ' + label)
    try:
        status('RECOVERING_SEED55_ACTUAL_PPO')
        call([shutil.which('pwsh'), '-NoProfile', '-File', source / 'scripts/run_registered_native.ps1',
            '-Batch', batch, '-Python', ROOT / '.venv/Scripts/python.exe',
            '-CarlaPython', ROOT / '.carla-python37/runtime/python.exe', '-ActivateQueued', '-UseCurrentGuard'], 'lifecycle', 7800)
        require_finished_parent(batch)
        env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'), PYTHONIOENCODING='utf-8')
        status('VERIFYING_RECOVERED_SEED55')
        call([sys.executable, source / 'scripts/audit_native_method_endpoint.py', '--batch', batch, '--method', 'B1'], 'raw', 600, env)
        call([sys.executable, source / 'scripts/audit_native_policy_inference.py', '--batch', batch, '--method', 'B1',
            '--output', batch / 'B1/measured_start_inference.json'], 'starts', 600, env)
        call([sys.executable, source / 'scripts/analyze_native_shared_parking_ppo.py', '--batch', batch,
            '--output', output / 'learning_audit.json'], 'learning', 600)
        call([sys.executable, source / 'scripts/audit_native_flat_sequence.py', '--batch', batch,
            '--output', batch / 'B1/whole_sequence_inference.json'], 'whole_sequence', 600)
        atomic_json(output / 'completed_five_courses.json', dict(original_four=read(QUEUE / 'status.json')['completed'],
            recovered_seed55=read(output / 'learning_audit.json'), interruption_disclosed=True))
        status('ALL_FIVE_COURSES_RECORDED_WITH_SEED55_RECOVERY')
        prep = ROOT / 'runs/B1_broader_shared_parking_preparation_v2_20261101'
        broader = ROOT / 'runs' / read(prep / 'registration.json')['batch']
        if sha(broader / 'protocol.json') != read(prep / 'registration.json')['protocol_sha256']:
            raise ValueError('Unstarted broader protocol changed')
        status('BROADER_PILOT_ACTUAL_PPO_RUNNING')
        call([shutil.which('pwsh'), '-NoProfile', '-File', source / 'scripts/run_registered_native.ps1',
            '-Batch', broader, '-Python', ROOT / '.venv/Scripts/python.exe',
            '-CarlaPython', ROOT / '.carla-python37/runtime/python.exe', '-ActivateQueued', '-UseCurrentGuard'], 'broader_lifecycle', 7800)
        require_finished_parent(broader)
        env['NAVSTACK_AUDIT_SOURCE_ROOT'] = str(broader / 'source')
        call([sys.executable, source / 'scripts/audit_native_method_endpoint.py', '--batch', broader, '--method', 'B1'], 'broader_raw', 600, env)
        call([sys.executable, source / 'scripts/audit_native_policy_inference.py', '--batch', broader, '--method', 'B1',
            '--output', broader / 'B1/measured_start_inference.json'], 'broader_starts', 600, env)
        call([sys.executable, source / 'scripts/analyze_native_broader_b1_course.py', '--batch', broader,
            '--output', output / 'broader_learning_audit.json'], 'broader_learning', 600)
        status('RECOVERY_AND_BROADER_PILOT_COMPLETE')
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

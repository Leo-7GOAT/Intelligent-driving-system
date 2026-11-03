"""Finish retained static development comparisons after one exact coordinator exits."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.native_cases import registered_native_cases
from navstack.owned_process import OwnedProcess
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read, read_prefix, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare(parent, output, kind):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every physical comparison attempt')
    old = read(parent / 'protocol.json')
    prefix_path = parent / 'H0_PPO/completed_prefix_audit.json'
    prefix = read(prefix_path)
    if (prefix['protocol_sha256'] != sha(parent / 'protocol.json')
            or prefix['evaluation_trace_sha256'] != sha(parent / 'H0_PPO/evaluation_trace.jsonl.gz')
            or prefix['model_sha256'] != sha(parent / 'H0_PPO/model.zip')
            or prefix['full_evaluation_complete'] is not False):
        raise ValueError('Require immutable audited interrupted prefix')
    if kind not in ('H0_remaining', 'H_teacher'):
        raise ValueError('Unknown remaining comparison')
    name = 'H0_PPO' if kind == 'H0_remaining' else 'H_teacher'
    ids = prefix['remaining_case_ids'] if kind == 'H0_remaining' else list(range(12))
    if kind == 'H_teacher' and (parent / 'H_teacher').exists():
        raise ValueError('Teacher already started; audit its prefix first')
    if not ids or any(i in prefix['completed_case_ids'] for i in ids) and kind == 'H0_remaining':
        raise ValueError('Do not repeat authenticated complete cases')
    for relative, digest in old['source_hashes'].items():
        if sha(parent / 'source' / relative) != digest:
            raise ValueError('Parent source changed')
    source = parent / old['recovery'][name]['checkpoint']
    verified_checkpoint_metadata(source)
    p = deepcopy(old)
    cases = [deepcopy(old['native_case_manifest']['cases'][i]) for i in ids]
    route_ids = sorted({case['route_index'] for case in cases})
    p['routes'] = [deepcopy(old['routes'][i]) for i in route_ids]
    for case in cases:
        case['route_index'] = route_ids.index(case['route_index'])
    p['native_case_manifest']['cases'] = cases
    previous, selected = registered_native_cases(old), registered_native_cases(p)
    for original, case in zip(ids, selected):
        if {k: v for k, v in previous[original].items() if k != 'condition'} != {k: v for k, v in case.items() if k != 'condition'}:
            raise ValueError('Physical route or hazard changed during case reindexing')
    if output.drive.lower() != 'd:' or shutil.disk_usage(output.parent).free < 16 * 1024**3:
        raise RuntimeError('Use D staging with at least16GiB headroom')
    output.mkdir()
    (output / 'infrastructure').mkdir()
    for directory in ('source', 'calibration'):
        shutil.copytree(parent / directory, output / directory, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(source, output / 'resume' / name)
    dataset = ROOT / 'runs/20261103T043242Z_native_recovery/measured_demonstrations.npz'
    if sha(dataset) != old['extra_supervision']['dataset_sha256']:
        raise ValueError('Original supervision provenance changed')
    shutil.copy2(dataset, output / dataset.name)
    inherited = deepcopy(old['recovery'][name])
    inherited['checkpoint'] = 'resume/' + name
    consumed_prefix = 0
    if name == 'H0_PPO':
        rows, _ = read_prefix(parent / name / 'evaluation_trace.jsonl.gz')
        episodes = {r['info']['trace'][0]['backend_evidence']['episode_id'] for r in rows}
        for path in (parent / 'infrastructure').glob('*_responses.jsonl.gz'):
            wrappers, _ = read_prefix(path)
            if any(w['response'].get('episode_id') in episodes for w in wrappers):
                consumed_prefix += sum('vehicle_control' in w['response'] for w in wrappers)
        inherited['consumed_controls_lower_bound'] += consumed_prefix
        inherited['unknown_unflushed_parent_tail'] = True
    now = time.time()
    p.update(methods=[name], jobs=[job for job in old['jobs'] if job['name'] == name],
             method_budgets={name: 0}, recovery={name: inherited}, created_epoch=now,
             deadline_epoch=now + 7200,
             resource_limits={**old['resource_limits'], 'wall_seconds': 7200, 'automatic_retries': 0},
             static_remainder_registration=dict(parent=parent.name, original_parent_case_ids=ids,
                 original_teacher_case_ids=[12 + i for i in ids], prefix_audit_sha256=sha(prefix_path),
                 prefix_consumed_controls_lower_bound=consumed_prefix, new_PPO_updates=0,
                 physical_route_and_hazard_exact=True, new_final_conditions=0),
             scope='Retained known static development comparison remainder. Local case IDs map explicitly to original IDs; no successful completed H or H0 case is rerun.')
    p['frozen_policy_comparison']['original_teacher_case_ids'] = [12 + i for i in ids]
    for key in ('original_B3_successes', 'original_B3_cases'):
        p['frozen_policy_comparison'].pop(key, None)
    atomic_json(output / 'protocol.json', p)
    atomic_json(output / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(output / 'protocol.json')))
    return output, name


def run(parent, execution, after_identity):
    execution = Path(execution).resolve()
    execution.mkdir()
    identity = read(after_identity)
    owner = OwnedProcess(identity['pid'], identity['creation_epoch'], identity['executable'], allow_terminate=False)
    atomic_json(execution / 'registration.json', dict(after_identity=identity, automatic_retries=0,
        scope='Wait for exact current reference coordinator; then only missing old static comparisons'))
    def call(args, label, timeout=900, env=None):
        atomic_json(execution / 'status.json', dict(phase=label, pid=os.getpid(), epoch=time.time()))
        with (execution / (label + '.log')).open('xb') as log:
            subprocess.run(list(map(str, args)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env, check=True)
    try:
        deadline = time.monotonic() + 28800
        atomic_json(execution / 'status.json', dict(phase='WAITING_FOR_EXACT_REFERENCE_COORDINATOR', pid=os.getpid(), epoch=time.time()))
        while owner.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded predecessor wait exhausted')
            time.sleep(10)
        atomic_json(execution / 'predecessor_exit.json', dict(exit_code=owner.poll()))
        results = {}
        for kind in ('H0_remaining', 'H_teacher'):
            batch, name = prepare(parent, Path('D:/ThesisNativeArchive_20261103') / ('20261103_static_' + kind + '_remainder'), kind)
            call([shutil.which('pwsh'), '-NoProfile', '-File', ROOT / 'scripts/run_registered_native.ps1',
                  '-Batch', batch, '-UseCurrentGuard'], kind + '_native', 7800)
            require_finished_parent(batch)
            env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'), PYTHONIOENCODING='utf-8')
            call([sys.executable, ROOT / 'scripts/audit_native_method_endpoint.py', '--batch', batch, '--method', name], kind + '_raw', env=env)
            call([sys.executable, ROOT / 'scripts/audit_native_policy_inference.py', '--batch', batch, '--method', name,
                  '--output', batch / name / 'measured_start_inference.json'], kind + '_starts', env=env)
            call([sys.executable, ROOT / 'scripts/audit_native_control_delivery.py', '--batch', batch,
                  '--output', batch / 'strict_delivery.json'], kind + '_delivery')
            if not read(batch / 'strict_delivery.json')['strict_delivery_gate_passed']:
                raise ValueError('Engine control acknowledgement failed')
            results[kind] = dict(batch=str(batch), episodes=read(batch / name / 'evaluation.json')['episodes'],
                                original_parent_case_ids=read(batch / 'protocol.json')['static_remainder_registration']['original_parent_case_ids'])
            atomic_json(execution / (kind + '_complete.json'), results[kind])
        atomic_json(execution / 'complete.json', dict(results=results, research_acceptance='UNMET'))
    except BaseException as error:
        atomic_json(execution / 'failure.json', dict(error=repr(error), epoch=time.time()))
        raise
    finally:
        owner.kernel.CloseHandle(owner.handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--after-identity', required=True)
    args = parser.parse_args()
    run(args.parent, args.execution, args.after_identity)

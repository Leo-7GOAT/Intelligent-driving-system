"""Freeze outcome-feedback learners on the complete existing development lists."""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.outcome_context_policy import VERSION
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.run_fixed_lowlevel_adaptation import endpoint


def prepare(parent, template, output, phase):
    parent, template, output = [Path(x).resolve() for x in (parent, template, output)]
    if phase not in ('placements30', 'known24'):
        raise ValueError('Use the complete existing development inventories')
    known_template = ('20261104_option_adapted_all30_placements_H' if phase == 'placements30'
                      else '20261104_option_adapted_known24_H_exact_compute')
    if template.name != known_template:
        raise ValueError('This entry point is restricted to the already used development lists')
    require_finished_parent(parent)
    p = read(parent / 'protocol.json')
    task = read(template / 'protocol.json')
    proof = read(parent / 'H/saved_learning_proof.json')
    training_raw = read(parent / 'training_only_raw_audit.json')
    if (p.get('reference_context') != {'version': VERSION}
            or proof['parameter_count'] != 34
            or not all(proof[k] for k in ('actual_initialized_parameters_and_Adam_exact',
                                         'initial_torch_RNG_exact', 'RMS_exact'))
            or not read(parent / 'strict_delivery.json')['strict_delivery_gate_passed']
            or proof['model_sha256'] != sha(parent / 'H/model.zip')
            or training_raw['protocol_sha256'] != sha(parent / 'protocol.json')
            or training_raw['methods']['H']['model_sha256'] != proof['model_sha256']
            or task['calibration_sha256'] != p['calibration_sha256']):
        raise ValueError('Require closed, audited learning with matching physical calibration')
    expected = 30 if phase == 'placements30' else 24
    manifest = task['native_case_manifest'] if phase == 'placements30' else task['native_validation']['manifest']
    if len(manifest['cases']) != expected:
        raise ValueError('Do not omit known conditions')
    for name, digest in p['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Frozen parent source changed:' + name)
    if output.drive.lower() != 'd:' or shutil.disk_usage(output.parent).free < 16 * 1024**3:
        raise ValueError('Use D-drive storage with16GiB headroom')
    endpoint(parent, output)
    p = read(output / 'protocol.json')
    for name in ('fixed_lowlevel_endpoint', 'outcome_initialization', 'native_validation'):
        p.pop(name, None)
    for name in ('routes', 'native_case_manifest', 'native_validation'):
        if name in task:
            p[name] = deepcopy(task[name])
    p['fixed_outcome_development_validation'] = dict(
        version='frozen_outcome_known_development_v1', phase=phase, expected_cases=expected,
        parent_training=parent.name, template=template.name,
        template_protocol_sha256=sha(template / 'protocol.json'),
        parent_learning_proof_sha256=sha(parent / 'H/saved_learning_proof.json'),
        frozen_model_sha256=sha(parent / 'H/model.zip'),
        comparison_label=read(parent / 'protocol.json')['fixed_lowlevel_adaptation_registration']['comparison_label'],
        new_PPO_updates=0, new_final_conditions=0,
        selection='Final fixed-budget learner;all known conditions retained;no checkpoint selection')
    p['scope'] = 'Frozen outcome-feedback policy on complete known development conditions;not final confirmation'
    now = time.time()
    p.update(created_epoch=now, deadline_epoch=now + 14400,
             resource_limits={**p['resource_limits'], 'wall_seconds': 14400, 'automatic_retries': 0})
    atomic_json(output / 'protocol.json', p)
    atomic_json(output / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(output / 'protocol.json')))
    return output


def run(batch, execution):
    batch, execution = [Path(x).resolve() for x in (batch, execution)]
    p = read(batch / 'protocol.json')
    registration = p['fixed_outcome_development_validation']
    execution.mkdir()
    atomic_json(execution / 'registration.json', dict(batch=str(batch), protocol_sha256=sha(batch / 'protocol.json'),
                script_sha256=sha(Path(__file__)), new_PPO_updates=0, new_final_conditions=0, automatic_retries=0))

    def call(args, phase, timeout=1800, env=None):
        atomic_json(execution / 'status.json', dict(phase=phase, pid=os.getpid(), epoch=time.time()))
        with (execution / (phase + '.log')).open('xb') as log:
            subprocess.run([str(x) for x in args], cwd=ROOT, check=True, stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout, env=env)
    try:
        call([shutil.which('pwsh'), '-NoProfile', '-File', ROOT / 'scripts/run_registered_native.ps1',
              '-Batch', batch, '-UseCurrentGuard'], 'native', 15000)
        require_finished_parent(batch)
        env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'), PYTHONIOENCODING='utf-8')
        call([sys.executable, ROOT / 'scripts/audit_native_method_endpoint.py', '--batch', batch, '--method', 'H'], 'raw', env=env)
        for flags, name in (([], 'measured_start_inference.json'), (['--all-static-actions'], 'static_action_inference.json')):
            call([sys.executable, ROOT / 'scripts/audit_native_reference_inference.py', '--batch', batch,
                  '--method', 'H', *flags, '--output', batch / 'H' / name], name, env=env)
        call([sys.executable, ROOT / 'scripts/audit_native_control_delivery.py', '--batch', batch,
              '--output', batch / 'strict_delivery.json'], 'engine')
        evaluation = read(batch / 'H/evaluation.json')
        episodes = evaluation['episodes']
        raw = read(batch / 'H/endpoint_raw_audit.json')
        if (len(episodes) != registration['expected_cases']
                or evaluation['model_sha256'] != registration['frozen_model_sha256']
                or sha(batch / 'H/model.zip') != registration['frozen_model_sha256']
                or not read(batch / 'strict_delivery.json')['strict_delivery_gate_passed']):
            raise ValueError('Complete frozen inventory/model/delivery verification failed')
        atomic_json(execution / 'complete.json', dict(
            batch=str(batch), comparison_label=registration['comparison_label'], cases=len(episodes),
            successes=sum(x['success'] for x in episodes),
            termination_counts=dict(Counter(x['termination_reason'] for x in episodes)),
            actual_controls=raw['evaluation_control_frames_verified'],
            raw_audit_sha256=sha(batch / 'H/endpoint_raw_audit.json'),
            model_sha256=evaluation['model_sha256'], new_PPO_updates=0, new_final_conditions=0,
            research_acceptance='UNMET', limits='Complete known development inventory,one continued policy;not independent final confirmation.'))
    except BaseException as error:
        atomic_json(execution / 'failure.json', dict(error=repr(error), epoch=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'template', 'output', 'phase'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--execution')
    args = parser.parse_args()
    batch = prepare(args.parent, args.template, args.output, args.phase)
    if args.execution:
        run(batch, args.execution)

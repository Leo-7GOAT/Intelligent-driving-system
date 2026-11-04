"""Finite actual H/H0 adaptation on the already collected diverse TRAIN roads."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import VehicleConfig
from navstack.experiments import signature
from navstack.owned_process import OwnedProcess
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha, read_prefix
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def learner_proof(batch, method):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from scripts.analyze_native_feedback_ppo import adam_step_increments
    folder = batch / method
    initial = load_ppo(batch / 'resume' / method / 'model.zip', device='cpu')
    final = load_ppo(folder / 'model.zip', device='cpu')
    summary = read(folder / 'summary.json')
    if (not 10240 <= summary['physics_steps'] < 10560 or summary['stopped_reason'] is not None
            or final.num_timesteps - initial.num_timesteps != summary['actions']):
        raise ValueError('Incomplete native control budget or saved decision count')
    proof = adam_step_increments(initial.policy.optimizer.state_dict()['state'],
                                final.policy.optimizer.state_dict()['state'], summary['optimizer_steps'])
    changes = {'actor': 0, 'critic': 0}
    for key, value in initial.policy.state_dict().items():
        if not torch.equal(value, final.policy.state_dict()[key]):
            changes['critic' if 'value_net' in key else 'actor'] += 1
    if not all(changes.values()):
        raise ValueError('Require actual actor and critic updates')
    norms = []
    for path in (batch / 'resume' / method / 'normalization.pkl', folder / 'normalization.pkl'):
        with path.open('rb') as stream:
            norms.append(pickle.load(stream).obs_rms)
    np.testing.assert_array_equal(norms[0].mean, norms[1].mean)
    np.testing.assert_array_equal(norms[0].var, norms[1].var)
    if norms[0].count != norms[1].count:
        raise ValueError('Frozen observation statistics changed')
    return dict(actual_controls=summary['physics_steps'], actual_decisions=summary['actions'],
                Adam=proof, changed_network_tensors=changes, observation_statistics_exact=True,
                model_sha256=sha(folder / 'model.zip'), new_auxiliary_updates=0)


def continuation_protocol(old, inventory, validation):
    if (old.get('methods') != ['H_PPO', 'H0_PPO', 'H_teacher']
            or old.get('evaluation_only') is not True
            or len(inventory['routes']) != 16 or len(inventory['native_case_manifest']['cases']) != 48
            or len(validation['manifest']['cases']) != 24):
        raise ValueError('Require the exact transferred pair and complete registered TRAIN/known-VAL inventories')
    p = deepcopy(old)
    for key in ('jobs', 'frozen_policy_comparison'):
        p.pop(key, None)
    p.update(methods=['H', 'H0'], evaluation_only=False,
             controlled_budget_per_method=10240, method_budgets={'H': 10240, 'H0': 10240},
             routes=deepcopy(inventory['routes']), native_case_manifest=deepcopy(inventory['native_case_manifest']),
             native_validation=deepcopy(validation), freeze_observation_statistics=True,
             deterministic_training_route_evaluation=True, record_initial_parameters=True,
             checkpoint_thresholds=[2560, 5120, 7680, 10240],
             scope='One canonical seed matched H/H0 actual adaptation on diverse TRAIN48,known24development endpoint. Fixed current synchronous delivery/parking throughout new learning. No new supervision or final conditions;not five-seed or whole-budget matched to B1.')
    return p


def prepare(execution, predecessor):
    """Reuse saved transfer bytes; freeze the low level before any new PPO."""
    old = read(predecessor / 'protocol.json')
    reference = ROOT / 'runs/20261102T130625263733Z_native_canonical_parent_TRAIN48_PPO10240'
    require_finished_parent(reference)
    inventory = read(reference / 'protocol.json')
    validation = read(ROOT / 'runs/20261102T132547635941Z_native_route_feedback_PPO_known_validation/protocol.json')['native_validation']
    p = continuation_protocol(old, inventory, validation)
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 16 * 1024**3:
        raise RuntimeError('Resources unavailable: ' + str(reason))
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_diverse_H_H0_PPO10240')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    for name in ('source', 'calibration'):
        shutil.copytree(predecessor / name, batch / name, ignore=shutil.ignore_patterns('__pycache__'))
    vehicle = VehicleConfig(**read(batch / 'calibration/calibration.json')['vehicle'])
    p['recovery'] = {}
    for method, source_name in (('H', 'H_PPO'), ('H0', 'H0_PPO')):
        origin = predecessor / 'resume' / source_name
        meta = verified_checkpoint_metadata(origin)
        if signature(training_config(p, method, vehicle)) != read(origin / 'config.json')['signature']:
            raise ValueError('Training changed the fixed deployment config')
        shutil.copytree(origin, batch / 'resume' / method)
        prior = deepcopy(old['recovery'][source_name])
        prior.update(checkpoint='resume/' + method, parent_batch=predecessor.name,
                     parent_checkpoint='resume/' + source_name)
        trace = predecessor / source_name / 'evaluation_trace.jsonl.gz'
        rows, error = read_prefix(trace) if trace.exists() else ([], None)
        diagnostic_controls = sum(len(row['info']['trace']) for row in rows)
        prior.update(diagnostic_recorded_controls=diagnostic_controls,
                     diagnostic_trace_incomplete=error is not None,
                     diagnostic_unknown_unflushed_tail=error is not None)
        prior['consumed_controls_lower_bound'] += diagnostic_controls
        raw_path = predecessor / source_name / 'endpoint_raw_audit.json'
        if raw_path.exists():
            if read(raw_path)['evaluation_control_frames_verified'] != diagnostic_controls:
                raise ValueError('Prior diagnostic cost differs from its completed raw audit')
        p['recovery'][method] = prior
        if sha(batch / 'resume' / method / 'model.zip') != meta['model_sha256']:
            raise ValueError('Saved source model changed')
    now = time.time()
    p.update(created_epoch=now, deadline_epoch=now + 14400,
             resource_limits={**p['resource_limits'], 'wall_seconds': 14400, 'automatic_retries': 0},
             diverse_native_adaptation=dict(version='diverse_TRAIN48_pair_v1', seed=22,
                 requested_new_controls_each=10240, maximum_rollout_overshoot_each=319,
                 new_auxiliary_updates=0, final_conditions=0,
                 all_training_conditions_retained_including_model_disconnected_case19=True,
                 predecessor_summary=(read(predecessor / 'summary.json')
                     if (predecessor / 'summary.json').exists() else None),
                 selection='One fixed final requested budget;retain every failure,actual overshoot and prior transfer cost.'))
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(batch / 'protocol.json')))
    atomic_json(execution / 'batch.json', dict(batch=batch.name, protocol_sha256=sha(batch / 'protocol.json')))
    return batch


def run(execution, dependency):
    execution, dependency = Path(execution).resolve(), Path(dependency).resolve()
    if execution.exists():
        raise FileExistsError('Preserve every finite adaptation attempt')
    state = read(dependency / 'status.json')
    predecessor = ROOT / 'runs' / state['batch']
    pid = int(state['pid'])
    command = (f'$taskProcess=Get-Process -Id {pid} -ErrorAction Stop; '
               '@{pid=$taskProcess.Id;creation_epoch=([DateTimeOffset]$taskProcess.StartTime).ToUnixTimeMilliseconds()/1000;executable=$taskProcess.Path} | ConvertTo-Json -Compress')
    identity = json.loads(subprocess.check_output([shutil.which('pwsh'), '-NoProfile', '-Command', command], encoding='utf-8-sig'))
    owner = OwnedProcess(**identity, allow_terminate=False)
    execution.mkdir()
    hashes = {name: sha(ROOT / name) for name in ('scripts/continue_native_hierarchical_pair.py',
        'scripts/audit_native_method_endpoint.py', 'scripts/audit_native_policy_inference.py',
        'scripts/audit_native_control_delivery.py')}
    atomic_json(execution / 'registration.json', dict(dependency=identity, source_hashes=hashes,
        requested_controls_per_method=10240, methods=['H', 'H0'], seed=22, automatic_retries=0,
        new_auxiliary_updates=0, new_final_conditions=0,
        rationale='Old hierarchical policies were never adapted to the twelve diverse base roads. Learn from actual measured execution on those already collected TRAIN roads,with common fixed low level.'))
    def status(phase, **extra):
        atomic_json(execution / 'status.json', dict(phase=phase, pid=os.getpid(), epoch=time.time(), **extra))
    def call(args, label, timeout=900, env=None):
        if any(sha(ROOT / name) != value for name, value in hashes.items()):
            raise ValueError('Registered adaptation/audit source changed')
        status(label)
        with (execution / (label + '.log')).open('xb') as stream:
            subprocess.run(list(map(str, args)), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True, env=env)
    try:
        deadline = time.monotonic() + 7200
        while owner.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('Finite preceding diagnostic wait exhausted')
            status('WAITING_FOR_EXACT_STATIC_DIAGNOSTIC_EXIT')
            time.sleep(10)
        atomic_json(execution / 'predecessor_exit.json', dict(exit_code=owner.poll(),
            complete=(dependency / 'complete.json').exists(), failure=(dependency / 'failure.json').exists(),
            note='An incomplete bounded diagnostic stays incomplete;source checkpoint bytes are independently verified.'))
        batch = prepare(execution, predecessor)
        call([shutil.which('pwsh'), '-NoProfile', '-File', ROOT / 'scripts/run_registered_native.ps1',
              '-Batch', batch, '-UseCurrentGuard'], 'actual_PPO_and_known24', 15000)
        require_finished_parent(batch)
        env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'), PYTHONIOENCODING='utf-8')
        results = {}
        for method in ('H', 'H0'):
            call([sys.executable, ROOT / 'scripts/audit_native_method_endpoint.py', '--batch', batch,
                  '--method', method], method + '_raw', env=env)
            call([sys.executable, ROOT / 'scripts/audit_native_policy_inference.py', '--batch', batch,
                  '--method', method, '--output', batch / method / 'measured_start_inference.json'], method + '_starts', env=env)
            proof = learner_proof(batch, method)
            episodes = read(batch / method / 'evaluation.json')['episodes']
            proof.update(successes=sum(e['success'] for e in episodes), cases=len(episodes))
            atomic_json(batch / method / 'learning_proof.json', proof)
            results[method] = proof
        call([sys.executable, ROOT / 'scripts/audit_native_control_delivery.py', '--batch', batch,
              '--output', batch / 'strict_delivery.json'], 'strict_delivery')
        if not read(batch / 'strict_delivery.json')['strict_delivery_gate_passed']:
            raise ValueError('Strict engine delivery failed')
        atomic_json(execution / 'complete.json', dict(batch=batch.name, results=results, research_acceptance='UNMET'))
        status('DIVERSE_NATIVE_ADAPTATION_COMPLETE', batch=batch.name, results=results)
    except BaseException as error:
        atomic_json(execution / 'failure.json', dict(error=repr(error), epoch=time.time()))
        raise
    finally:
        owner.kernel.CloseHandle(owner.handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--dependency', required=True)
    args = parser.parse_args()
    run(args.execution, args.dependency)

"""Run one registered reference course, verify saved learning, then development endpoints."""
import argparse
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent


def run(batch, execution):
    batch, execution = Path(batch).resolve(), Path(execution).resolve()
    execution.mkdir()  # Every attempt has its own durable evidence.
    protocol = read(batch / 'protocol.json')
    digest = sha(batch / 'protocol.json')
    atomic_json(execution / 'registration.json', dict(batch=str(batch), protocol_sha256=digest,
        automatic_retries=0, new_final_conditions=0, scope='One seed development only'))

    def call(arguments, label, timeout=900, env=None):
        atomic_json(execution / 'status.json', dict(phase=label, pid=os.getpid(), epoch=time.time()))
        with (execution / (label + '.log')).open('xb') as log:
            subprocess.run(list(map(str, arguments)), cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout, env=env, check=True)

    def native(target, label):
        call([shutil.which('pwsh'), '-NoProfile', '-File', ROOT / 'scripts/run_registered_native.ps1',
              '-Batch', target, '-UseCurrentGuard'], label,
             read(target / 'protocol.json')['resource_limits']['wall_seconds'] + 600)
        require_finished_parent(target)

    def delivery(target, label):
        call([sys.executable, ROOT / 'scripts/audit_native_control_delivery.py', '--batch', target,
              '--output', target / 'strict_delivery.json'], label)
        if not read(target / 'strict_delivery.json')['strict_delivery_gate_passed']:
            raise ValueError('Engine control delivery failed')

    try:
        native(batch, 'native_training')
        if sha(batch / 'protocol.json') != digest:
            raise ValueError('Registered protocol changed')
        delivery(batch, 'training_delivery')
        env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'), PYTHONIOENCODING='utf-8')
        call([sys.executable, ROOT / 'scripts/audit_native_training_only.py', '--batch', batch,
              '--output', batch / 'training_only_raw_audit.json'], 'training_raw', env=env)
        import numpy as np
        from navstack.independent_clip_ppo import load_ppo
        proofs = {}
        for method in protocol['methods']:
            folder = batch / method
            summary = read(folder / 'summary.json')
            budget = protocol['method_budgets'][method]
            if not budget <= summary['physics_steps'] < budget + 320 or summary['stopped_reason'] is not None:
                raise ValueError('Registered training budget incomplete')
            before = load_ppo(batch / 'resume' / method / 'model.zip', device='cpu')
            after = load_ppo(folder / 'model.zip', device='cpu')
            if after.num_timesteps - before.num_timesteps != summary['actions']:
                raise ValueError('Saved decision count differs')
            old, new = dict(before.policy.named_parameters()), dict(after.policy.named_parameters())
            empty = []
            if set(old) != set(new) or len(new) != 32:
                raise ValueError('Saved parameter inventory differs')
            for name, parameter in new.items():
                start = before.policy.optimizer.state.get(old[name], {})
                end = after.policy.optimizer.state.get(parameter, {})
                if not start:
                    empty.append(name)
                if float(end.get('step', -1)) - float(start.get('step', 0)) != summary['optimizer_steps']:
                    raise ValueError('Adam advancement differs: ' + name)
            if set(empty) != {'action_net.reference_bias.gain', 'value_net.context_weight'}:
                raise ValueError('Unexpected empty original Adam states')
            with (batch / 'resume' / method / 'normalization.pkl').open('rb') as stream:
                a = pickle.load(stream).obs_rms
            with (folder / 'normalization.pkl').open('rb') as stream:
                z = pickle.load(stream).obs_rms
            np.testing.assert_array_equal(a.mean, z.mean)
            np.testing.assert_array_equal(a.var, z.var)
            if a.count != z.count:
                raise ValueError('Frozen RMS count changed')
            proofs[method] = dict(summary=summary, model_sha256=sha(folder / 'model.zip'),
                all32Adamstates_advance=summary['optimizer_steps'], RMS_exact=True,
                learned_reference_gain=float(after.policy.action_net.reference_bias.gain.detach()),
                reference_gain_scale=protocol['reference_learning_registration'].get('gain_scale', 1.0))
            atomic_json(folder / 'saved_learning_proof.json', proofs[method])
        atomic_json(execution / 'training_complete.json', proofs)
        results = {}
        for phase in ('static', 'known_validation'):
            target = batch.parent / (batch.name + '_' + phase)
            call([sys.executable, ROOT / 'scripts/prepare_native_reference_evaluation.py', '--parent', batch,
                  '--output', target, '--phase', phase], phase + '_prepare')
            native(target, phase + '_native')
            env = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(target / 'source'), PYTHONIOENCODING='utf-8')
            phase_results = {}
            for method in protocol['methods']:
                call([sys.executable, ROOT / 'scripts/audit_native_method_endpoint.py', '--batch', target,
                      '--method', method], phase + '_' + method + '_raw', env=env)
                call([sys.executable, ROOT / 'scripts/audit_native_reference_inference.py', '--batch', target,
                      '--method', method, '--output', target / method / 'measured_start_inference.json'],
                     phase + '_' + method + '_starts', env=env)
                call([sys.executable, ROOT / 'scripts/audit_native_reference_inference.py', '--batch', target,
                      '--method', method, '--all-static-actions', '--output', target / method / 'static_action_inference.json'],
                     phase + '_' + method + '_static_actions', env=env)
                episodes = read(target / method / 'evaluation.json')['episodes']
                if len(episodes) != (12 if phase == 'static' else 24):
                    raise ValueError('Incomplete registered development inventory')
                phase_results[method] = dict(cases=len(episodes), successes=sum(e['success'] for e in episodes))
            delivery(target, phase + '_delivery')
            results[phase] = phase_results
            atomic_json(execution / (phase + '_complete.json'), dict(batch=str(target), results=phase_results))
        atomic_json(execution / 'complete.json', dict(results=results, methods=protocol['methods'],research_acceptance='UNMET',
            scope='Selected one-seed known development only; no score inferred for unselected methods. H0 when selected removes both old feedback and reference context.'))
    except BaseException as error:
        atomic_json(execution / 'failure.json', dict(error=repr(error), epoch=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--execution', required=True)
    args = parser.parse_args()
    run(args.batch, args.execution)

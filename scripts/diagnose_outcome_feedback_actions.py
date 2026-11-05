"""Measure the frozen failure term's action influence;never execute alternatives."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def diagnose(batch, output, *, all_actions=False):
    batch, output = Path(batch).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every completed mechanism diagnostic')
    os.environ['NAVSTACK_AUDIT_SOURCE_ROOT'] = str(batch / 'source')
    from scripts.audit_native_reference_inference import audit
    from scripts.audit_native_method_endpoint import read, read_prefix, sha
    from scripts.collect_native_recovery import tensor_digest
    from navstack import independent_clip_ppo as loader
    from navstack.research import atomic_json
    import numpy as np
    import torch

    protocol = read(batch / 'protocol.json')
    if (protocol.get('reference_context') != {'version': 'action_aligned_outcome_context_v1'}
            or protocol['outcome_context_registration']['failure_feedback_ablated'] is not False):
        raise ValueError('Require the frozen feedback-enabled policy,not the trained H0 control')
    raw = read(batch / 'H/endpoint_raw_audit.json')
    trace = batch / 'H/evaluation_trace.jsonl.gz'
    if sha(trace) != raw['evaluation_trace_sha256']:
        raise ValueError('Raw-bound recorded observations are mandatory')
    rows, error = read_prefix(trace)
    if error:
        raise ValueError(error)
    ordered = [r for case in sorted({r['case_id'] for r in rows}) for r in rows
               if r['case_id'] == case and (all_actions or not r['info']['initial_scenario']['dynamic'])]
    original_load = loader.load_ppo
    loaded = []
    examined = []
    exposed = []

    def observed_load(*args, **kwargs):
        if loaded:
            raise ValueError('Only one fixed evaluated model may be diagnosed')
        model = original_load(*args, **kwargs)
        head = model.policy.action_net.reference_bias
        if model.policy.outcome_ablated or head.failure_ablated:
            raise ValueError('The actual model must retain its learned failure term')
        before = (tensor_digest(model.policy.state_dict()), tensor_digest(model.policy.optimizer.state_dict()),
                  torch.get_rng_state().clone(), model.num_timesteps)
        original_predict = model.predict

        def observed_predict(observation, *args, **kwargs):
            if kwargs.get('deterministic') is not True:
                raise ValueError('Use the actual deterministic frozen endpoint convention')
            result = original_predict(observation, *args, **kwargs)
            row = ordered[len(examined)]
            action = int(result[0][0])
            if action != int(row['policy_action']):
                raise ValueError('Observed policy action differs from the raw record')
            context = np.asarray(observation)[0, -31:]
            examined.append(row['case_id'])
            if np.any(context < 0):
                tensor = torch.as_tensor(observation)
                with torch.no_grad():
                    active = model.policy.get_distribution(tensor).distribution.probs.clone()
                    try:
                        head.failure_ablated = True
                        hidden = model.policy.get_distribution(tensor).distribution.probs.clone()
                    finally:
                        head.failure_ablated = False
                if int(active.argmax()) != action:
                    raise ValueError('Actual categorical mode differs from actual predict')
                exposed.append(dict(case_id=row['case_id'], decision_index=len(examined) - 1,
                    failed_labels=np.flatnonzero(context < 0).tolist(), actual_action=action,
                    same_weights_failure_hidden_action=int(hidden.argmax()),
                    actual_probabilities=active[0].tolist(), failure_hidden_probabilities=hidden[0].tolist()))
            return result

        model.predict = observed_predict
        loaded.append((model, before))
        return model

    try:
        loader.load_ppo = observed_load
        proof = audit(batch, 'H', **({'all_actions':True} if all_actions else {'all_static':True}))
    finally:
        loader.load_ppo = original_load
    if len(examined) != len(ordered) or not loaded:
        raise ValueError('Require every selected recorded decision,not an exposed-only selection')
    model, before = loaded[0]
    if (tensor_digest(model.policy.state_dict()) != before[0]
            or tensor_digest(model.policy.optimizer.state_dict()) != before[1]
            or not torch.equal(torch.get_rng_state(), before[2])
            or model.num_timesteps != before[3]
            or model.policy.action_net.reference_bias.failure_ablated):
        raise ValueError('Read-only mechanism analysis changed model/Adam/RNG/counters')
    atomic_json(output, dict(scope=__doc__, batch=batch.name, script_sha256=sha(Path(__file__)),
        protocol_sha256=sha(batch / 'protocol.json'), model_sha256=sha(batch / 'H/model.zip'),
        trace_sha256=sha(trace), reconstruction=proof, decisions=len(examined),
        negative_context_decisions=len(exposed),
        all_recorded_actions=all_actions,
        learned_failure_gain=float(model.policy.action_net.reference_bias.failure_gain.detach()),
        failure_gain_scale=float(model.policy.action_net.reference_bias.failure_scale),
        deterministic_modes_changed=sum(r['actual_action'] != r['same_weights_failure_hidden_action'] for r in exposed),
        exposed_decisions=exposed, weights_and_full_Adam_RNG_counters_unchanged=True,
        new_controls=0, new_updates=0,
        limits='Same-weight actor counterfactuals are not the independently trained H0 policy.No alternative action is driven;no counterfactual success/failure or causal driving gain is inferred.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--all-actions',action='store_true')
    args = parser.parse_args()
    diagnose(args.batch, args.output,all_actions=args.all_actions)

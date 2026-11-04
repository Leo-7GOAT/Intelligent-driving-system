"""Rebuild the fixed-budget, single-seed failure-feedback development pair."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from navstack.independent_clip_ppo import load_ppo
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read, read_prefix, sha
from scripts.collect_native_recovery import tensor_digest
from scripts.summarize_fixed_lowlevel_adaptation import summarize


def summarize_pair(h_execution, h0_execution):
    executions = [Path(x).resolve() for x in (h_execution, h0_execution)]
    results = [summarize(x) for x in executions]
    batches = [x.parent / r['training_batch'] for x, r in zip(executions, results)]
    protocols = [read(x / 'protocol.json') for x in batches]
    for key in ('source_hashes', 'routes', 'native_case_manifest', 'seed',
                'calibration_sha256', 'controlled_budget_per_method',
                'fixed_lowlevel_training', 'freeze_observation_statistics'):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError('Paired protocol differs:' + key)
    if results[0]['baseline_batch'] != results[1]['baseline_batch']:
        raise ValueError('Different pre-learning baselines')
    initializers = [load_ppo(x / 'resume/H/model.zip', device='cpu') for x in batches]
    for index, (p, model) in enumerate(zip(protocols, initializers)):
        ablated = bool(index)
        if (p['outcome_context_registration']['failure_feedback_ablated'] is not ablated
                or model.policy_kwargs['outcome_ablated'] is not ablated):
            raise ValueError('Wrong failure-only ablation')
    a, b = [x.policy for x in initializers]
    if (a.state_dict().keys() != b.state_dict().keys()
            or any(not torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
            or tensor_digest(a.optimizer.state_dict()) != tensor_digest(b.optimizer.state_dict())):
        raise ValueError('Paired initial weights/full Adam differ')
    for name in ('normalization.pkl', 'resume_state.pkl'):
        if sha(batches[0] / 'resume/H' / name) != sha(batches[1] / 'resume/H' / name):
            raise ValueError('Paired normalization/RNG differ')
    for batch, result, initializer, protocol in zip(batches, results, initializers, protocols):
        summary = read(batch / 'H/summary.json')
        if summary != read(batch / 'H/saved_learning_proof.json')['summary']:
            raise ValueError('Training cost summary differs from the saved learning proof')
        architecture = read(batch / 'H/policy_architecture.json')
        if architecture['parameters'] != sum(p.numel() for p in initializer.policy.parameters()):
            raise ValueError('Recorded network size differs from actual parameters')
        result['training'].update(wall_seconds=summary['wall_seconds'], world_ticks=summary['world_ticks'],
                                  requested_controls=protocol['controlled_budget_per_method'],
                                  rollout_overshoot=summary['physics_steps'] - protocol['controlled_budget_per_method'])
        result['architecture'] = architecture
        target = batch.parent / result['endpoint_batch']
        episodes = read(target / 'H/evaluation.json')['episodes']
        for case, episode in zip(result['paired_cases'], episodes):
            case['after'].update({k: episode[k] for k in ('simulation_seconds', 'progress', 'plan_failures',
                'mpc_failures', 'control_cycle_times', 'decision_cycle_times', 'policy_prediction_times')})
        result['frozen_endpoint_costs'] = dict(
            physical_controls=sum(e['physics_steps'] for e in episodes),
            simulation_seconds=sum(e['simulation_seconds'] for e in episodes),
            interventions=sum(e['interventions'] for e in episodes),
            plan_failures=sum(e['plan_failures'] for e in episodes),
            mpc_failures=sum(e['mpc_failures'] for e in episodes))
        trace = target / 'H/evaluation_trace.jsonl.gz'
        if sha(trace) != read(target / 'H/endpoint_raw_audit.json')['evaluation_trace_sha256']:
            raise ValueError('Feedback history trace differs from the raw audit')
        rows, error = read_prefix(trace)
        if error:
            raise ValueError(error)
        histories = {}
        exposed = 0
        for row in rows:
            history = histories.setdefault(row['case_id'], set())
            exposed += bool(history)
            if row['info']['mode'] == 'reference':
                history.clear()
            elif row['info']['mode'] == 'plan_failed' and int(row['policy_action']):
                history.add(int(row['policy_action']))
        result['failure_history_nonempty_predecision_count'] = exposed
    control = load_ppo(batches[1] / 'H/model.zip', device='cpu')
    if (not control.policy.outcome_ablated
            or torch.count_nonzero(control.policy.action_net.reference_bias.failure_gain)
            or torch.count_nonzero(control.policy.value_net.failure_weight)):
        raise ValueError('Disabled new failure terms changed in H0')
    cases = []
    for h, h0 in zip(results[0]['paired_cases'], results[1]['paired_cases']):
        if h['original_case_id'] != h0['original_case_id'] or h['before'] != h0['before']:
            raise ValueError('Endpoint case pairing differs')
        cases.append(dict(original_known_case_id=h['original_case_id'],
                          before=h['before'], H=h['after'], H0=h0['after']))
    if len(cases) != 2:
        raise ValueError('Require all two registered known endpoint conditions')
    new_controls = sum(r['training']['actual_controls'] +
                       sum(c[label]['physics_steps'] for c in cases)
                       for r, label in zip(results, ('H', 'H0')))
    return dict(scope='One continued-seed, known-condition development ablation',
                research_acceptance='UNMET', independent_seeds=1, continued_seed=protocols[0]['seed'],
                new_final_conditions=0, initial_weights_Adam_RNG_RMS_exact=True,
                H0_new_failure_terms_remained_zero=True,
                intervention='Only new failed-action actor/critic branches hidden in H0;old feedback retained',
                H=results[0], H0=results[1], paired_cases=cases,
                contribution_limits='Equal completion counts do not show a feedback success-rate benefit. Nonempty failure history is counted from raw modes;zero H exposure means its new failure terms did not directly change endpoint logits/values. Training-path effects and observed intervention/time differences remain exploratory.',
                safe_completion_count_difference_H_minus_H0=results[0]['after_successes'] - results[1]['after_successes'],
                unsafe_count_difference_H_minus_H0=results[0]['after_unsafe_episodes'] - results[1]['after_unsafe_episodes'],
                new_training_and_endpoint_controls=new_controls,
                registration_protocol_sha256=[sha(x / 'protocol.json') for x in batches],
                script_sha256=sha(Path(__file__)),
                limits='One seed and two previously used conditions;no independent-seed confidence interval or general benefit claim. Actual rollout overshoot is disclosed separately for each arm. Baseline controls are shared historical costs,not charged twice as new interactions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('h-execution', 'h0-execution', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve each completed comparison')
    atomic_json(output, summarize_pair(args.h_execution, args.h0_execution))

"""Build a closed-course learning comparison from raw-bound frozen endpoints."""
import argparse
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,read_prefix,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_fixed_lowlevel_adaptation import without_settings


def evaluation(batch):
    require_finished_parent(batch);raw_path=batch/'H/endpoint_raw_audit.json';raw=read(raw_path)
    result=read(batch/'H/evaluation.json')
    if (raw['protocol_sha256']!=sha(batch/'protocol.json') or raw['evaluation_sha256']!=sha(batch/'H/evaluation.json')
            or result['model_sha256']!=sha(batch/'H/model.zip')
            or not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']):
        raise ValueError('Frozen endpoint evidence changed')
    for name in ['measured_start_inference.json','static_action_inference.json']:
        if read(batch/'H'/name)['raw_audit_sha256']!=sha(raw_path):raise ValueError('Inference lacks raw binding')
    return result,raw,sha(raw_path)


def summarize(execution):
    execution=Path(execution).resolve();done=read(execution/'complete.json');archive=execution.parent
    batch=archive/done['training'];target=archive/done['endpoint'];require_finished_parent(batch)
    p=read(batch/'protocol.json');registration=p['fixed_lowlevel_adaptation_registration'];baseline=archive/registration['parent_pilot']
    ids=registration['original_known_development_case_ids'];before,br,bh=evaluation(baseline);after,ar,ah=evaluation(target)
    if len(before['episodes'])!=len(ids) or len(after['episodes'])!=len(ids):raise ValueError('Incomplete matched inventory')
    old=read(baseline/'protocol.json');endpoint=read(target/'protocol.json')
    initializer_matches=sha(batch/'resume/H/model.zip')==before['model_sha256']
    if p.get('outcome_initialization'):
        receipt=read(batch/p['outcome_initialization']['receipt']);equivalence=read(batch/p['outcome_initialization']['equivalence'])
        initializer_matches=(receipt['old_model_sha256']==before['model_sha256']
            and receipt['new_model_sha256']==sha(batch/'resume/H/model.zip')
            and equivalence['model_sha256']==receipt['new_model_sha256']
            and equivalence['raw_audit_sha256']==bh and equivalence['outcome_migration_probabilities_and_values_exact'])
    if (p['routes']!=old['routes'] or p['native_case_manifest']!=old['native_case_manifest']
            or endpoint['source_hashes']!=p['source_hashes']
            or sha(batch/'H/model.zip')!=after['model_sha256']
            or not initializer_matches):
        raise ValueError('Fixed task/source/model comparison violated')
    allowed={'navstack/reference_road_refinement.py','navstack/steering_transition.py','navstack/reference_option_commitment.py',
        'navstack/steering_retiming.py','navstack/fixed_lowlevel_training.py','navstack/mpc_reference_effort.py'}
    expected_modules={n for n in old['source_hashes'] if n.startswith('navstack/')}
    if p.get('outcome_initialization'):expected_modules.add('navstack/outcome_context_policy.py')
    if {n for n in p['source_hashes'] if n.startswith('navstack/')}!=expected_modules:
        raise ValueError('Runtime module inventory differs')
    for name,digest in old['source_hashes'].items():
        if name.startswith('navstack/') and digest!=p['source_hashes'][name]:
            if name not in allowed or without_settings(baseline/'source'/name)!=without_settings(batch/'source'/name):
                raise ValueError('Lowlevel changed across learning:'+name)
    training_path=batch/'training_only_raw_audit.json';raw=read(training_path);proof_path=batch/'H/saved_learning_proof.json';proof=read(proof_path)
    if (raw['protocol_sha256']!=sha(batch/'protocol.json') or raw['methods']['H']['model_sha256']!=after['model_sha256']
            or not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']
            or not all(proof[k] for k in ['actual_initialized_parameters_and_Adam_exact','initial_torch_RNG_exact','RMS_exact'])):
        raise ValueError('Actual learning provenance incomplete')
    terminations=Counter()
    for path in (batch/'H').glob('transitions*.gz'):
        rows,error=read_prefix(path)
        if error or sha(path)!=raw['methods']['H']['transition_hashes'][path.name]:raise ValueError('Training transitions changed')
        terminations.update(r['termination_reason'] for r in rows if r.get('termination_reason'))
    summaries=[]
    for original,a,z in zip(ids,before['episodes'],after['episodes']):
        summaries.append(dict(original_case_id=original,before={k:a[k] for k in ['success','termination_reason','physics_steps','interventions','distance']},
            after={k:z[k] for k in ['success','termination_reason','physics_steps','interventions','distance']}))
    return dict(scope='Closed known-condition learning comparison with fixed lowlevel;single continued policy',research_acceptance='UNMET',
        training_batch=batch.name,endpoint_batch=target.name,baseline_batch=baseline.name,source_git_commit=registration['source_git_commit'],
        training=dict(actual_controls=raw['training_control_frames_verified'],reset_ticks=raw['reset_ticks_verified'],
            decisions=raw['methods']['H']['actions'],Adam_updates_each=proof.get('all_parameter_Adam_states_advance',proof.get('all32Adamstates_advance')),
            parameter_states=proof.get('parameter_count',32),learned_failure_gain=proof.get('learned_failure_gain'),
            actual_initial_weights_Adam_RNG_equal=True,RMS_exact=True,completed_episode_termination_counts=dict(terminations),
            unfinished_episode_not_counted_as_success=True,new_supervision=0,learned_reference_gain=proof['learned_reference_gain']),
        before_successes=sum(e['success'] for e in before['episodes']),after_successes=sum(e['success'] for e in after['episodes']),
        before_unsafe_episodes=sum(e['termination_reason'] in ('collision','out_of_bounds') for e in before['episodes']),
        after_unsafe_episodes=sum(e['termination_reason'] in ('collision','out_of_bounds') for e in after['episodes']),
        unique_known_conditions=len(ids),paired_cases=summaries,
        before_model_sha256=before['model_sha256'],after_model_sha256=after['model_sha256'],
        initializer_model_sha256=sha(batch/'resume/H/model.zip'),zero_effect_schema_migration=bool(p.get('outcome_initialization')),
        raw_audit_sha256=dict(training=sha(training_path),before=bh,after=ah),saved_learning_proof_sha256=sha(proof_path),
        actual_controls_in_comparison=raw['training_control_frames_verified']+br['evaluation_control_frames_verified']+ar['evaluation_control_frames_verified'],
        script_sha256=sha(Path(__file__)),
        limits='Repeated known conditions and one continued seed,not independent-seed statistical evidence or final confirmation. Stochastic training success is separate from frozen-policy performance. Formal five-seed,120unique-condition,baseline and ablation requirements remain unmet.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve completed reports')
    result=summarize(a.execution);atomic_json(output,result)
    print({k:result[k] for k in ['before_successes','after_successes','unique_known_conditions']})

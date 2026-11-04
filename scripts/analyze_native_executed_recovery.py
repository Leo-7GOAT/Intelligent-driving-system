"""Combine the audited TRAIN mixture, fixed actor fit and pure-policy probe."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.run_native_corrective_probes import summarize_probe
from scripts.fit_native_executed_recovery import require_probe_fit


def analyze(fit):
    fit=Path(fit);fitted=read(fit/'complete.json');registered=read(fit/'probe_registration.json')
    collection=ROOT/'runs'/fitted['collection_batch'];probe=ROOT/'runs'/registered['batch']
    require_finished_parent(collection);require_finished_parent(probe)
    data=read(collection/'B1/collection.json');audit=read(collection/'B1/collection_raw_audit.json')
    unprojected=require_probe_fit(fitted)
    if (sha(collection/'B1/collection_raw_audit.json')!=fitted['collection_audit_sha256']
            or sha(collection/'B1/collection.json')!=audit['collection_result_sha256']
            or sha(collection/'B1/collection.npz')!=audit['dataset_sha256']
            or audit['not_policy_evaluation'] is not True
            or read(probe/'protocol.json')['executed_native_recovery_actor_fit']['complete_sha256']!=sha(fit/'complete.json')):
        raise ValueError('Collection,fit or pure-probe provenance differs')
    evaluation=read(probe/'B1/evaluation.json');raw=read(probe/'B1/endpoint_raw_audit.json')
    if any(sha(probe/'B1'/name)!=raw[key] for name,key in [('evaluation.json','evaluation_sha256'),
        ('evaluation_trace.jsonl.gz','evaluation_trace_sha256'),('summary.json','summary_sha256')]):
        raise ValueError('Pure-policy evaluation differs from raw audit')
    inference=read(probe/'B1/measured_start_inference.json')
    if (len(inference['results'])!=4 or inference['model_sha256']!=fitted['model_sha256']
            or inference['trace_sha256']!=raw['evaluation_trace_sha256']):raise ValueError('Require all measured initial actions')
    performance=summarize_probe(evaluation,read(probe/'B1/summary.json'),raw,
        {'batch':probe.name,'model_sha256':fitted['model_sha256'],'normalization_sha256':fitted['normalization_sha256']})
    if (data['collection_controls']!=audit['collection_controls']
            or data['teacher_controls']+data['learner_controls']!=data['collection_controls']
            or fitted['new_executed_teacher_TRAIN_rows']!=audit['eligible_executed_teacher_labels']):
        raise ValueError('Actual collection and fitting costs differ')
    result={'scope':__doc__,'seed':22,'collection':{
        'batch':collection.name,'original_TRAIN_route_indices':read(collection/'protocol.json')['native_recovery_collection']['original_TRAIN_route_indices'],
        **{key:data[key] for key in ('collection_controls','collection_world_ticks','teacher_controls','learner_controls',
            'eligible_executed_teacher_labels','dataset_sha256','trace_sha256','teacher_query_seconds','wall_seconds')},
        'successes':audit['training_mixture_successes'],'unsafe_episodes':audit['training_mixture_unsafe_episodes'],
        'case_outcomes':[{'case_id':e['case_id'],'reason':e['termination_reason'],'controls':e['physics_steps']} for e in data['episodes']],
        'PPO_updates':0,'supervised_updates':0,'not_policy_evaluation':True,
        'all_measured_observations_and_learner_actions_bit_exact':audit['all_measured_observations_bit_exact'] and audit['all_learner_proposals_bit_exact'],
        'audit_sha256':sha(collection/'B1/collection_raw_audit.json')},
        'fit':{'path':fit.relative_to(ROOT).as_posix(),'complete_sha256':sha(fit/'complete.json'),
            **{key:fitted[key] for key in ('original_executed_teacher_TRAIN_rows','new_executed_teacher_TRAIN_rows',
                'excluded_counterfactual_collection_rows','new_supervised_Adam_proposals','retention_gate',
                'new_TRAIN_teacher_MSE_before','new_TRAIN_teacher_MSE_after')},
            'critic_PPO_optimizer_counters_RMS_retained':all(fitted['fit'][key] for key in
                ('critic_exact','retained_PPO_optimizer_exact','retained_PPO_counters_exact','normalization_statistics_exact')),
            'new_PPO_updates':0,'new_controls':0},
        'pure_policy_probe':{**performance,'raw_audit_sha256':sha(probe/'B1/endpoint_raw_audit.json'),
            'measured_initial_actions_verified':4,'teacher_used_at_runtime':False},
        'new_controls_total':data['collection_controls']+raw['total_applied_controls_verified'],
        'new_supervised_Adam_proposals_total':512,'new_PPO_controls_total':0,'new_PPO_updates_total':0,
        'retained_PPO_lineage_controls':20480,
        'basic_gate':'At least3/4successesANDzero collision/bounds;unchanged.',
        'selection':'All registered episodes,one fixed final fit endpoint,no retries,no outcome-based label exclusion.',
        'new_final_conditions_used':0,'raw_assets_status':'Local audited evidence;not yet a publicly reconstructed capsule.',
        'research_acceptance':'UNMET;one known development seed,not a healthy full-task five-seed baseline.'}
    if fitted.get('route_profile')=='original_TRAIN4_v1':
        previous=read(ROOT/'reports/native_B1_executed_recovery_collection_and_probe_20261101.json')
        result.update(route_profile='original_TRAIN4_v1',
            unchanged_retention_reference_model_sha256=fitted['retention_reference_model_sha256'],
            both_recovery_attempts_controls=previous['new_controls_total']+result['new_controls_total'],
            both_recovery_attempts_auxiliary_Adam_proposals=previous['new_supervised_Adam_proposals_total']+512,
            cumulative_actor_intervention_steps=read(fit/'checkpoint/metadata.json')['cumulative_actor_fit_steps'])
    if unprojected:
        import numpy as np
        projected_path=ROOT/fitted['matched_projected_fit'];projected_fit=read(projected_path/'complete.json')
        projected=read(ROOT/'reports/native_B1_original_TRAIN_recovery_collection_and_probe_20261101.json')
        for key in ('source_model_sha256','source_normalization_sha256','steps','learning_rate','seed','sampling'):
            if fitted[key]!=projected_fit[key]:raise ValueError('Projection ablation changes another fit input:'+key)
        with np.load(fit/'fit_inputs.npz',allow_pickle=False) as a,np.load(projected_path/'fit_inputs.npz',allow_pickle=False) as b:
            if set(a.files)!=set(b.files):raise ValueError('Projection ablation changes dataset schema')
            for name in a.files:np.testing.assert_array_equal(a[name],b[name])
        result['new_controls_total']=raw['total_applied_controls_verified']
        result.pop('both_recovery_attempts_controls',None);result.pop('both_recovery_attempts_auxiliary_Adam_proposals',None)
        result.update(fit_profile=fitted['fit_profile'],TRAIN_retention_screen_applied=False,
            matched_projection_ablation={'projected_report':'reports/native_B1_original_TRAIN_recovery_collection_and_probe_20261101.json',
                'projected_report_sha256':sha(ROOT/'reports/native_B1_original_TRAIN_recovery_collection_and_probe_20261101.json'),
                'same_parent_data_weights_seed_and_budget':True,'projected_probe':projected['pure_policy_probe'],
                'reused_collection_controls':data['collection_controls'],'new_collection_controls':0,
                'driving_gate_unchanged':True,'only_intended_change':'Remove auxiliary TRAINMSEdisplacement projection;no change to physical controls,policy architecture or driving qualification.'},
            all_recovery_profiles_controls=projected['both_recovery_attempts_controls']+raw['total_applied_controls_verified'],
            all_recovery_profiles_auxiliary_Adam_proposals=projected['both_recovery_attempts_auxiliary_Adam_proposals']+512)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve completed report')
    out.write_text(json.dumps(analyze(Path(args.fit).resolve()),indent=2),encoding='utf-8')

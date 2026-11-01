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


def analyze(fit):
    fit=Path(fit);fitted=read(fit/'complete.json');registered=read(fit/'probe_registration.json')
    collection=ROOT/'runs'/fitted['collection_batch'];probe=ROOT/'runs'/registered['batch']
    require_finished_parent(collection);require_finished_parent(probe)
    data=read(collection/'B1/collection.json');audit=read(collection/'B1/collection_raw_audit.json')
    if (sha(collection/'B1/collection_raw_audit.json')!=fitted['collection_audit_sha256']
            or sha(collection/'B1/collection.json')!=audit['collection_result_sha256']
            or sha(collection/'B1/collection.npz')!=audit['dataset_sha256']
            or audit['not_policy_evaluation'] is not True or not fitted['retention_gate']['passed']
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
    return {'scope':__doc__,'seed':22,'collection':{
        'batch':collection.name,'original_TRAIN_route_indices':[4,5,6,7],
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve completed report')
    out.write_text(json.dumps(analyze(Path(args.fit).resolve()),indent=2),encoding='utf-8')

"""Stage both matched trained-actor fits on all four original basic cases."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def require_fit_result(result):
    constrained=result.get('retained_correction_profile')=='basic_teacher_group_backtracking_v1'
    clean=result.get('executed_teacher_only_profile')=='verified_measured_groups0to11_v1'
    arms=['teacher_only_cleaning'] if clean else ['trajectory_correction'] if constrained else ['replay','trajectory_correction']
    updates=512 if constrained or clean else 1024
    if (result['steps_each']!=512 or result['additional_supervised_steps_total']!=updates
            or result['new_controls'] or result['PPO_updates']
            or [r['arm'] for r in result['results']]!=arms
            or result.get('require_basic_retention') and result.get('retention_gate',{}).get('passed') is not True
            or (constrained or clean) and (result.get('retention_gate',{}).get('passed') is not True
                or result.get('new_supervised_updates')!=512 or result.get('reused_supervised_updates')!=0)):
        raise ValueError('Require every declared fit arm and passed constrained-TRAINgate')
    return result


def prepare(fit):
    fit=Path(fit).resolve();result=read(fit/'complete.json');parent=ROOT/'runs'/result['parent']
    require_finished_parent(parent)
    require_fit_result(result)
    if (fit/'native_probe_registration.json').exists():
        raise ValueError('Preserve previous probe registration')
    for name,key in [('fit_inputs.npz','inputs_sha256'),('executed_probe.py','script_sha256'),('executed_fit.py','fit_source_sha256')]:
        if sha(fit/name)!=result[key]:raise ValueError('Executed fit inputs/code changed')
    old=read(parent/'protocol.json');raw=read(parent/'B1/endpoint_raw_audit.json');summary=read(parent/'B1/summary.json')
    if (old['methods']!=['B1'] or len(old['native_case_manifest']['cases'])!=4
            or old.get('native_validation') or sha(parent/'B1/endpoint_raw_audit.json')!=result['parent_raw_audit_sha256']):
        raise ValueError('Require original four known basic roads and audited trained parent')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent frozen source changed')
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    rows=[]
    for arm in result['results']:
        origin=fit/arm['arm'];metadata=verified_checkpoint_metadata(origin)
        if (metadata['model_sha256']!=arm['model_sha256'] or metadata['normalization_sha256']!=arm['normalization_sha256']
                or not arm['fit']['retained_PPO_optimizer_exact'] or not arm['fit']['critic_exact']
                or arm['fit']['PPO_updates'] or arm['fit']['additional_supervised_steps']!=512):
            raise ValueError('Matched actor fit artifact changed')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_trained_actor_'+arm['arm'])
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(origin,batch/'resume/B1')
        shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
        shutil.copytree(fit/'oracle',batch/'corrective_oracle')
        for name in ('complete.json','registration.json','fit_inputs.npz','executed_probe.py','executed_fit.py'):
            target=batch/'actor_fit_evidence'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(fit/name,target)
        for name in ('executed_retention_gate.py','fixed_retention_gate_before_correction.json'):
            if (fit/name).exists():shutil.copy2(fit/name,batch/'actor_fit_evidence'/name)
        p=deepcopy(old)
        for name in ('jobs','freeze_observation_statistics','own_success_replay','native_validation','native_camera_evaluation'):
            p.pop(name,None)
        now=time.time();p.update(evaluation_only=True,controlled_budget_per_method=0,method_budgets={'B1':0},
            checkpoint_thresholds=[],created_epoch=now,deadline_epoch=now+1800,parent_batch=parent.name,
            scope='Matched trained-PPO actor correction diagnostic on all4known basic roads;zero new PPO;not independent final tests or healthy-baseline qualification',
            trained_actor_correction={'version':'trained_actor_separate_Adam_v1','arm':arm['arm'],
                'fit_result_sha256':sha(fit/'complete.json'),'fit':arm['fit'],
                'new_training_controls':0,'additional_supervised_steps':512,'source_model_sha256':result['parent_model_sha256'],
                'counterfactual_queries_not_executed_controls':True,'selection':'All4original cases,all outcomes;no best checkpoint choice',
                'extra_query_rows_used':result['solved_reference_rows'] if arm['arm']=='trajectory_correction' else 0},
            recovery={'B1':{'checkpoint':'resume/B1','parent_batch':parent.name,'parent_checkpoint':'B1/'+read(parent/'B1/latest_checkpoint.json')['folder'],
                'lineage_controls':summary['lineage_control_steps'],
                'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
                'retained_optimizer_steps':old['recovery']['B1']['retained_optimizer_steps']+summary['optimizer_steps'],
                'unknown_unflushed_parent_tail':old['recovery']['B1'].get('unknown_unflushed_parent_tail',False)}},
            resource_limits={**old['resource_limits'],'wall_seconds':1800,'automatic_retries':0})
        p['extra_supervision']['supervised_optimizer_steps']+=512
        p['extra_supervision']['additional_trained_actor_intervention']={
            'arm':arm['arm'],'fit_evidence':'actor_fit_evidence/complete.json',
            'inputs_sha256':result['inputs_sha256'],'steps':512,
            'scope':'Original teacher TRAIN rows plus solved-reference on-policy TRAIN labels in correction arm;no new driving or value targets'}
        if result.get('executed_teacher_only_profile')=='verified_measured_groups0to11_v1':
            p['scope']='Frozen trained B1after executed-teacher-only TRAINrepair;all4known basic roads,zero new PPO,not a qualified five-seed baseline'
            p['executed_teacher_only_training_repair']={
                'profile':result['executed_teacher_only_profile'],'teacher_TRAIN_rows':result['teacher_TRAIN_rows'],
                'excluded_counterfactual_rows':result['excluded_counterfactual_rows'],
                'excluded_counterfactual_weight':result['excluded_counterfactual_weight'],
                'new_auxiliary_Adam_updates':512,'new_query_rows':0,
                'fit_mask':'actor_fit_evidence/fit_inputs.npz:fit_train_mask',
                'retention_gate':result['retention_gate'],
                'ancestry':'Earlier supervision and actual PPOremain in the parent lineage;only this new512step fit excludes counterfactual rows.'}
            p['extra_supervision']['additional_trained_actor_intervention']['scope']='Only original physically executed teacher TRAINgroups0..11. Exact fit mask retained;29counterfactual rows excluded from this new fit,zero new queries. Parent ancestry unchanged.'
        # Physical worker and its complete frozen source are identical across arms.
        p['source_hashes']=deepcopy(old['source_hashes'])
        atomic_json(batch/'protocol.json',p)
        atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        rows.append({'arm':arm['arm'],'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json'),
                     'model_sha256':arm['model_sha256'],'normalization_sha256':arm['normalization_sha256']})
    atomic_json(fit/'native_probe_registration.json',{'scope':__doc__,'batches':rows,'new_controls_so_far':0,'PPO_updates':0})
    return rows


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fit',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.fit)),flush=True)

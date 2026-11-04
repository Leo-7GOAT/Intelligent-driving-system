"""One fixed all4-TRAIN deployment after the reflection fit's unchanged screen."""
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


def prepare(fit):
    fit=Path(fit).resolve()
    if (fit/'probe_registration.json').exists():raise FileExistsError('Preserve registered mirror probe')
    result=read(fit/'complete.json');metadata=verified_checkpoint_metadata(fit/'checkpoint')
    if (not result['offline_screen_passed'] or result['model_sha256']!=metadata['model_sha256']
            or sha(fit/'fit_inputs.npz')!=result['fit_inputs_sha256'] or sha(fit/'executed_fit.py')!=result['script_sha256']):
        raise ValueError('Require fixed completed,hash-bound reflection fit before deployment')
    parent=ROOT/'runs/20261101T094700825994Z_native_B1_feedback_PPO';require_finished_parent(parent)
    old=read(parent/'protocol.json');summary=read(parent/'B1/summary.json');raw=read(parent/'B1/endpoint_raw_audit.json')
    if (read(parent/'control_delivery_audit.json')['strict_delivery_gate_passed'] is not True
            or metadata['normalization_sha256']!=sha(parent/'B1/normalization.pkl')):raise ValueError('Fixed sync lowlevel and normalization required')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Frozen native source changed:'+name)
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_reflection_feedback_probe')
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(fit/'checkpoint',batch/'resume/B1')
    shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    evidence=batch/'fit_evidence';evidence.mkdir()
    for name in ('registration.json','complete.json','fit_inputs.npz','executed_fit.py','executed_reflection.py','auxiliary_Adam.pt'):
        shutil.copy2(fit/name,evidence/name)
    shutil.copy2(__file__,batch/'source/scripts/prepare_native_reflection_probe.py')
    p=deepcopy(old);p.pop('native_feedback_PPO');p.pop('freeze_observation_statistics',None);now=time.time()
    p.update(scope='Direct B1NNdeployment after explicit virtual reflection augmentation. Same4original TRAINroads and fixed sync interface;no runtime reflection,teacher,planner or new PPO. Not final confirmation.',
        evaluation_only=True,controlled_budget_per_method=0,method_budgets={'B1':0},checkpoint_thresholds=[],
        created_epoch=now,deadline_epoch=now+1800,resource_limits={**old['resource_limits'],'wall_seconds':1800},
        native_reflection_feedback={'version':'virtual_reflection_pose_feedback_v1','fit_complete_sha256':sha(fit/'complete.json'),
            'new_auxiliary_Adam_steps':512+result.get('inherited_reflection_Adam_steps',0),
            'virtual_rows_are_executed':False,'new_PPO_updates':0,'qualification':'All4original TRAINroads;success>=3/4ANDzero collision/bounds;unchanged.',
            'policy_runtime':'Original direct architecture;no reflection/averaging/rules at runtime.',
            'limits':'Extra virtual supervised data,not new physical demonstrations,independent final cases or isolated PPOvalue.'})
    p['recovery']['B1'].update(checkpoint='resume/B1',parent_batch=parent.name,
        parent_checkpoint='B1/'+read(parent/'B1/latest_checkpoint.json')['folder'],
        lineage_controls=summary['lineage_control_steps'],
        consumed_controls_lower_bound=summary['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
        retained_optimizer_steps=old['recovery']['B1']['retained_optimizer_steps']+summary['optimizer_steps'])
    p['extra_supervision']['supervised_optimizer_steps']+=p['native_reflection_feedback']['new_auxiliary_Adam_steps']
    p['extra_supervision']['reflection_augmentation']={'acknowledged_teacher_TRAINrows':3424,'virtual_reflected_rows':3424,
        'new_actual_teacher_controls':0,'hypothetical_pose_triplets':120,'new_native_training_controls':0}
    p['source_hashes']['scripts/prepare_native_reflection_probe.py']=sha(batch/'source/scripts/prepare_native_reflection_probe.py')
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
    atomic_json(fit/'probe_registration.json',{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')});print(batch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);a=p.parse_args();prepare(a.fit)

"""Register one all4-TRAIN-road deployment probe after the pose-response screen."""
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
    if (fit/'probe_registration.json').exists():raise FileExistsError('Preserve registered deployment probe')
    result=read(fit/'complete.json')
    if not result['offline_screen_passed'] or result['new_PPO_updates']!=0:
        raise ValueError('Offline response and retention screen must pass before driving')
    if sha(fit/'fit_inputs.npz')!=result['fit_inputs_sha256'] or sha(fit/'executed_fit.py')!=result['script_sha256']:
        raise ValueError('Executed fit evidence changed')
    checkpoint=verified_checkpoint_metadata(fit/'checkpoint')
    if checkpoint['model_sha256']!=result['model_sha256']:raise ValueError('Fit model changed')
    parent=ROOT/'runs/20261101T084713744547Z_native_sync_delivery_transfer';require_finished_parent(parent)
    old=read(parent/'protocol.json')
    if read(parent/'control_delivery_audit.json')['strict_delivery_gate_passed'] is not True:
        raise ValueError('Require the validated explicit sync low-level version')
    if checkpoint['normalization_sha256']!=read(parent/'B1/evaluation.json')['normalization_sha256']:
        raise ValueError('Normalizer changed')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Sync source changed:'+name)
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_feedback_contrast_probe')
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(fit/'checkpoint',batch/'resume/B1')
    shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    evidence=batch/'fit_evidence';evidence.mkdir()
    for name in ('registration.json','complete.json','executed_fit.py','fit_inputs.npz','query_diagnostic.json','auxiliary_Adam.pt'):
        shutil.copy2(fit/name,evidence/name)
    shutil.copy2(__file__,batch/'source/scripts/prepare_native_feedback_contrast_probe.py')
    p=deepcopy(old);p.pop('control_delivery_transfer');now=time.time()
    p.update(scope='All4original TRAINroads after explicit pose-response auxiliary fit. Direct B1NNonly,no runtime teacher. Frozen policy deployment on fixed batch_sync_v1;not new RL or final confirmation.',
        created_epoch=now,deadline_epoch=now+1800,resource_limits={**old['resource_limits'],'wall_seconds':1800},
        native_feedback_contrast={'version':'TRAINpose_response_contrast_v1','fit_complete_sha256':sha(fit/'complete.json'),
            'fit_steps':result['new_auxiliary_Adam_updates'],'inherited_fit_steps':result.get('inherited_contrastive_Adam_steps',0),
            'new_PPO_updates':0,'counterfactual_pose_triplets':result['pose_triplets'],
            'qualification':'All4original TRAINroads;success>=3/4ANDzero collision/bounds. No automatic full-task or five-seed promotion.',
            'low_level':'batch_sync_v1;same gains,20Hzphysics,goals,reward,normalizer and direct architecture as sync control.',
            'limits':'Additional counterfactual supervision;not an isolated PPOcontribution or training under the new lowlevel.'})
    p['recovery']['B1'].update(parent_batch=parent.name,parent_checkpoint='B1',checkpoint='resume/B1',
        consumed_controls_lower_bound=old['recovery']['B1']['consumed_controls_lower_bound']+read(parent/'B1/endpoint_raw_audit.json')['total_applied_controls_verified'])
    p['extra_supervision']['supervised_optimizer_steps']+=result['new_auxiliary_Adam_updates']+result.get('inherited_contrastive_Adam_steps',0)
    p['extra_supervision']['pose_response_fit']={'new_Adam_steps':result['new_auxiliary_Adam_updates'],
        'prior_contrastive_steps':result.get('inherited_contrastive_Adam_steps',0),'triplets':result['pose_triplets'],
        'discarded_ridge_solves':1,'new_native_teacher_controls':0,'new_PPO_updates':0}
    p['source_hashes']['scripts/prepare_native_feedback_contrast_probe.py']=sha(batch/'source/scripts/prepare_native_feedback_contrast_probe.py')
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
    atomic_json(fit/'probe_registration.json',{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')})
    print(batch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);a=p.parse_args();prepare(a.fit)

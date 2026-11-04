"""One real PPO basic-course continuation after the rejected pose-response probe."""
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


def prepare():
    pointer=ROOT/'runs/native_B1_feedback_PPO_registration_20261101.json'
    if pointer.exists():raise FileExistsError('Preserve the registered actual PPOcourse')
    parent=ROOT/'runs/20261101T094244452999Z_native_feedback_contrast_probe';require_finished_parent(parent)
    result=read(ROOT/'runs/native_B1_feedback_contrast_20261101.json');old=read(parent/'protocol.json')
    if result['probe']['basic_gate_passed'] or result['probe']['successes']!=2:
        raise ValueError('This is a basic-course learning continuation after the preserved failed probe,no full-task promotion')
    raw=read(parent/'B1/endpoint_raw_audit.json');summary=read(parent/'B1/summary.json')
    if (raw['protocol_sha256']!=sha(parent/'protocol.json') or not result['probe']['all_engine_controls_acknowledged']
            or raw['total_applied_controls_verified']!=1782):raise ValueError('Require complete actual failed-probe accounting')
    checkpoint=parent/'resume/B1';metadata=verified_checkpoint_metadata(checkpoint)
    if metadata['model_sha256']!='2e5925184e4f6eb560d96724526d51dec8cdf4f5c37314e15af3f7d53e46d53b':
        raise ValueError('Use the declared single fixed pose-response endpoint')
    if read(parent/'calibration/calibration.json')['recommended_adapter']['control_delivery']!='batch_sync_v1':
        raise ValueError('Require the validated fixed sync lowlevel throughout learning')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed:'+name)
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_B1_feedback_PPO')
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(checkpoint,batch/'resume/B1')
    shutil.copytree(parent/'fit_evidence',batch/'fit_evidence')
    shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    shutil.copy2(__file__,batch/'source/scripts/prepare_native_feedback_ppo.py')
    p=deepcopy(old);p.pop('native_feedback_contrast');now=time.time()
    p.update(scope='Actual PPO basic-road continuation after explicit pose-response supervision,with fixed validated batch_sync_v1. One development seed;not a qualified five-seed/full-task baseline or final confirmation.',
        evaluation_only=False,controlled_budget_per_method=10240,method_budgets={'B1':10240},
        checkpoint_thresholds=list(range(1280,10241,1280)),created_epoch=now,deadline_epoch=now+7200,
        deterministic_training_route_evaluation=True,freeze_observation_statistics=True,
        resource_limits={**old['resource_limits'],'wall_seconds':7200,'automatic_retries':0},
        native_feedback_PPO={'version':'pose_response_initialized_sync_PPO10240_v1',
            'source_model_sha256':metadata['model_sha256'],'normalization_sha256':metadata['normalization_sha256'],
            'prior_pose_response_Adam_steps':1536,'new_supervised_updates':0,'new_PPO_controls':10240,
            'exploration':'Retain trained Gaussian log_std;no reset or std search.',
            'low_level_fixed':'batch_sync_v1;no gains,reward,goals,observations or controller changes during training.',
            'source_probe_report_sha256':sha(ROOT/'runs/native_B1_feedback_contrast_20261101.json'),
            'selection':'One final10240control endpoint,all4known TRAINroad evaluations retained,no outcome-based checkpoint choice.',
            'basic_gate':'At least3/4successANDzero collision/bounds;unchanged. No automatic full-task promotion.',
            'limits':'Warm-started real PPOwith separately disclosed extra supervision,not from-scratch or isolated PPObenefit.'})
    p['recovery']['B1'].update(parent_batch=parent.name,parent_checkpoint='B1',checkpoint='resume/B1',
        lineage_controls=summary['lineage_control_steps'],
        consumed_controls_lower_bound=summary['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'])
    p['source_hashes']['scripts/prepare_native_feedback_ppo.py']=sha(batch/'source/scripts/prepare_native_feedback_ppo.py')
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
    atomic_json(pointer,{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json'),'new_controls_so_far':0})
    print(batch,flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();prepare()

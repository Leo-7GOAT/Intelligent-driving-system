"""Continue qualified basic PPO policies on the original twelve-condition DEVELOPMENT task."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from navstack.independent_clip_ppo import load_ppo
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def require_all_courses(state):
    rows=state['completed']
    if (state['phase']!='ALL_FIVE_BASIC_PPO_COURSES_QUALIFIED_FOR_FULL_TASK_TRAINING'
            or [r['seed'] for r in rows]!=[11,22,33,44,55]
            or any(r.get('basic_qualified_after_learning') is not True for r in rows)):
        raise ValueError('All five actually trained basic courses must qualify;no seed omission or gate waiver')


def full_protocol(old,parent_name,summary,evaluation_controls):
    course=old['basic_motion_learning'];manifest=course['original_full_case_manifest']
    if (old['methods']!=['B1'] or old.get('evaluation_only') or old['method_budgets']!={'B1':10240}
            or len(old['native_case_manifest']['cases'])!=4 or len(manifest['cases'])!=12
            or summary['physics_steps']!=10240 or summary['optimizer_steps']<=0
            or summary['stopped_reason'] is not None or summary['reload_exact'] is not True
            or type(evaluation_controls) is not int or evaluation_controls<=0):
        raise ValueError('Complete actual PPO course and every development probe are required')
    p=deepcopy(old);p['native_case_manifest']=deepcopy(manifest)
    p.update(controlled_budget_per_method=20480,method_budgets={'B1':20480},
             checkpoint_thresholds=list(range(1280,20481,1280)),
             scope='Actual trained basic-PPO continuation on all12original DEVELOPMENT conditions;all curriculum and prior failed costs disclosed;not unseen final confirmation')
    p['full_task_after_basic_course']={'version':'all_five_full_task_PPO20480_after_basic10240_v1',
        'parent':parent_name,'basic_training_controls':summary['physics_steps'],
        'basic_actual_Adam_steps':summary['optimizer_steps'],'basic_evaluation_controls':evaluation_controls,
        'new_full_training_controls':20480,'policy_lineage_target_controls':30720,
        'additional_supervised_updates':0,'optimizer_weights_normalizer_and_RNG_retained':True,
        'limits':'Additional10240basic-course interactions and failed/lost branches must enter total costs. No equal-total-budget H/B1superiority claim. No new final conditions.'}
    p['resource_limits']['wall_seconds']=7200;p['resource_limits']['automatic_retries']=0
    return p


def prepare(parent):
    parent=Path(parent).resolve();require_finished_parent(parent)
    registration=parent/'full_task_after_basic_registration.json'
    if registration.exists():raise FileExistsError('Preserve the registered full-task continuation')
    healthy,reason=resources_ok(ROOT)
    if not healthy:raise RuntimeError(str(reason))
    old=read(parent/'protocol.json');folder=parent/'B1';summary=read(folder/'summary.json');raw=read(folder/'endpoint_raw_audit.json')
    if sha(folder/'evaluation_trace.jsonl.gz')!=raw['evaluation_trace_sha256'] or sha(parent/'protocol.json')!=raw['protocol_sha256']:
        raise ValueError('Qualified parent raw evidence changed')
    from scripts.run_native_basic_motion_ppo_cohort import basic_qualification
    if not basic_qualification(read(folder/'evaluation.json')['episodes']):raise ValueError('Actual parent basic gate failed')
    name=read(folder/'latest_checkpoint.json')['folder']
    if Path(name).name!=name:raise ValueError('Invalid saved checkpoint name')
    origin=folder/name;metadata=verified_checkpoint_metadata(origin)
    if metadata['physics_steps']!=summary['physics_steps'] or metadata['optimizer_steps']!=summary['optimizer_steps']:
        raise ValueError('Latest checkpoint did not retain the completed course')
    torch.set_num_threads(1)
    saved=load_ppo(origin/'model.zip',device='cpu');final=load_ppo(folder/'model.zip',device='cpu')
    if (saved.num_timesteps!=10240 or saved._n_updates<=0 or not saved.policy.optimizer.state
            or any(not torch.equal(v,final.policy.state_dict()[k]) for k,v in saved.policy.state_dict().items())):
        raise ValueError('Final trained policy and retained learning checkpoint differ')
    for relative,digest in old['source_hashes'].items():
        if sha(parent/'source'/relative)!=digest:raise ValueError('Parent frozen source changed')
    p=full_protocol(old,parent.name,summary,raw['evaluation_control_frames_verified'])
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_B1_full_after_basic')
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',batch/'calibration');shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    shutil.copytree(origin,batch/'resume/B1');shutil.copy2(Path(__file__),batch/'source/scripts'/Path(__file__).name)
    p['recovery']={'B1':{'checkpoint':'resume/B1','parent_batch':parent.name,'parent_checkpoint':'B1/'+name,
        'lineage_controls':metadata['lineage_control_steps'],
        'consumed_controls_lower_bound':metadata['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
        'retained_optimizer_steps':metadata['optimizer_steps'],
        'unknown_unflushed_parent_tail':old['recovery']['B1'].get('unknown_unflushed_parent_tail',False)}}
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+7200,parent_batch=parent.name)
    p['source_hashes']={f.relative_to(batch/'source').as_posix():sha(f) for f in (batch/'source').rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    atomic_json(batch/'protocol.json',p);record={'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json'),'status':'PREPARED_NOT_STARTED'}
    atomic_json(batch/'prepared.json',{'status':record['status'],'protocol_sha256':record['protocol_sha256']})
    with registration.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    return batch


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);a=p.parse_args();print(prepare(a.parent))

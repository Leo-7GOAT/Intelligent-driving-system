"""Create a new owned single-case reproduction from a verified restored capsule."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json,resources_ok
from navstack.native_cases import registered_native_cases
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare(parent,method,output,case_id=0):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Never reuse an existing reproduction lifecycle')
    require_finished_parent(parent);old=read(parent/'protocol.json')
    if method not in old['methods'] or old.get('evaluation_only') is not True:
        raise ValueError('Require the declared method in a completed frozen-policy capsule')
    cases=registered_native_cases(old)
    if type(case_id) is not int or not 0<=case_id<len(cases):raise ValueError('Invalid original development case')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Restored frozen source differs:'+name)
    latest=read(parent/method/'latest_checkpoint.json');origin=parent/method/latest['folder']
    metadata=verified_checkpoint_metadata(origin)
    if metadata['model_sha256']!=read(parent/method/'evaluation.json')['model_sha256']:raise ValueError('Restored model differs')
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    output.mkdir(parents=True);(output/'infrastructure').mkdir()
    shutil.copytree(parent/'source',output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',output/'calibration');shutil.copytree(origin,output/'resume'/method)
    if (parent/'measured_demonstrations.npz').exists():shutil.copy2(parent/'measured_demonstrations.npz',output/'measured_demonstrations.npz')
    p=deepcopy(old);selected=deepcopy(old['native_case_manifest']['cases'][case_id]);route=old['routes'][selected['route_index']];selected['route_index']=0
    p.update(methods=[method],method_budgets={method:0},controlled_budget_per_method=0,checkpoint_thresholds=[],
        routes=[deepcopy(route)],native_case_manifest={**old['native_case_manifest'],'cases':[selected]},
        scope='New physical representative reproduction of one prospectively selected known TRAINcondition from a published frozen model. Not a new independent test or training run.',
        created_epoch=time.time(),deadline_epoch=time.time()+1800,
        resource_limits={**old['resource_limits'],'wall_seconds':1800,'automatic_retries':0},
        public_native_reproduction={'parent_protocol_sha256':sha(parent/'protocol.json'),
            'original_case_id':case_id,'local_case_id':0,'model_sha256':metadata['model_sha256'],
            'new_PPO_updates':0,'new_final_conditions':0,'case_selection':'Original case0 by default;fixed before the new lifecycle.'})
    p.pop('freeze_observation_statistics',None);p.pop('native_validation',None)
    summary=read(parent/method/'summary.json');raw=read(parent/method/'endpoint_raw_audit.json')
    previous=old.get('recovery',{}).get(method,{})
    p['recovery']={method:{'checkpoint':'resume/'+method,'parent_batch':parent.name,'parent_checkpoint':method+'/'+latest['folder'],
        'lineage_controls':summary['lineage_control_steps'],
        'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified'],
        'retained_optimizer_steps':previous.get('retained_optimizer_steps',0)+summary['optimizer_steps'],
        'model_sha256':metadata['model_sha256'],'normalization_sha256':metadata['normalization_sha256']}}
    registered_native_cases(p)
    atomic_json(output/'protocol.json',p);atomic_json(output/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(output/'protocol.json')})
    print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--method',required=True)
    p.add_argument('--output',required=True);p.add_argument('--case',type=int,default=0);a=p.parse_args();prepare(a.batch,a.method,a.output,a.case)

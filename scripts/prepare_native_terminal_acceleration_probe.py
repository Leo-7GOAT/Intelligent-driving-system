"""Register a direct learned-acceleration deployment with the steering path fixed."""
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
    if (fit/'probe_registration.json').exists():raise FileExistsError('Preserve registered acceleration probe')
    r=read(fit/'complete.json');m=verified_checkpoint_metadata(fit/'checkpoint')
    if not r['offline_screen_passed'] or m['model_sha256']!=r['model_sha256'] or sha(fit/'fit_inputs.npz')!=r['fit_inputs_sha256']:
        raise ValueError('Require the complete predeclared terminal fit screen')
    if sha(fit/'executed_policy.py')!=r['policy_source_sha256'] or sha(ROOT/'navstack/terminal_acceleration_policy.py')!=r['policy_source_sha256']:
        raise ValueError('Executed177parameter policy implementation changed')
    parent=ROOT/'runs/20261101T100704463570Z_native_reflection_feedback_probe';require_finished_parent(parent)
    old=read(parent/'protocol.json');raw=read(parent/'B1/endpoint_raw_audit.json')
    if m['normalization_sha256']!=sha(parent/'B1/normalization.pkl'):raise ValueError('Normalizer changed')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Frozen source changed:'+name)
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_terminal_acceleration_probe')
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(fit/'checkpoint',batch/'resume/B1')
    shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    evidence=batch/'fit_evidence';evidence.mkdir()
    for name in ('registration.json','complete.json','fit_inputs.npz','executed_fit.py','executed_policy.py','auxiliary_Adam.pt'):
        shutil.copy2(fit/name,evidence/name)
    for source,name in ((ROOT/'navstack/terminal_acceleration_policy.py','navstack/terminal_acceleration_policy.py'),
            (Path(__file__),'scripts/prepare_native_terminal_acceleration_probe.py')):
        shutil.copy2(source,batch/'source'/name)
    p=deepcopy(old);p.pop('native_reflection_feedback');now=time.time()
    p.update(scope='One fixed direct B1NNdeployment with learned177parameter acceleration residual. Original steering NNweights unchanged;no runtime teacher,reflection,planner,goal-distance rule or new PPO. All4original TRAINroads.',
        created_epoch=now,deadline_epoch=now+1800,resource_limits={**old['resource_limits'],'wall_seconds':1800},
        native_terminal_acceleration={'version':'learned_terminal_acceleration_residual_v1','fit_complete_sha256':sha(fit/'complete.json'),
            'new_auxiliary_Adam_steps':512+r.get('inherited_terminal_head_Adam_steps',0),'new_parameters':177,
            'all_original_steering_weights_fixed':True,'new_PPO_updates':0,
            'hypothetical_velocity_labels_are_executed':False,'qualification':'All4known TRAINroads,success>=3/4ANDzero collision/bounds;unchanged.',
            'limits':'Additional supervised NNcapacity and counterfactual velocity labels;not independent final conditions or isolated PPOvalue.'})
    p['recovery']['B1'].update(parent_batch=parent.name,parent_checkpoint='B1',checkpoint='resume/B1',
        consumed_controls_lower_bound=old['recovery']['B1']['consumed_controls_lower_bound']+raw['total_applied_controls_verified'])
    p['extra_supervision']['supervised_optimizer_steps']+=p['native_terminal_acceleration']['new_auxiliary_Adam_steps']
    query_manifest=read(ROOT/'runs/B1_terminal_velocity_queries_20261101/manifest.json')
    if sha(ROOT/'runs/B1_terminal_velocity_queries_20261101/manifest.json')!=r['terminal_query_manifest_sha256']:
        raise ValueError('Terminal query provenance changed')
    p['extra_supervision']['terminal_velocity_queries']={
        'measured_teacher_states':len({(v['case_id'],v['control_index']) for v in query_manifest['records']}),
        'hypothetical_velocity_rows':query_manifest['available_labels'],
        'virtual_reflected_velocity_rows':query_manifest['available_labels'],'new_native_teacher_controls':0,'new_PPO_updates':0}
    for name in ('navstack/terminal_acceleration_policy.py','scripts/prepare_native_terminal_acceleration_probe.py'):
        p['source_hashes'][name]=sha(batch/'source'/name)
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
    atomic_json(fit/'probe_registration.json',{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')});print(batch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);a=p.parse_args();prepare(a.fit)

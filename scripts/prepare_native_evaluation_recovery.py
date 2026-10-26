"""Resume an interrupted endpoint's full development evaluation without learning."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_interrupted_native import audit as interrupted_audit
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.evaluate_native_rule import require_finished_parent
from scripts.audit_native_method_endpoint import audit as endpoint_audit


def prepare(parent,method='H',original_goals=False):
    parent=Path(parent).resolve();old=read(parent/'protocol.json')
    guard=read(parent/'guard_final.json')
    if original_goals:
        require_finished_parent(parent)
        if 'curriculum' not in old or endpoint_audit(parent,method)!=read(parent/method/'endpoint_raw_audit.json'):
            raise ValueError('Require a complete audited curriculum endpoint')
    elif guard.get('reason')!='wall_deadline' or type(guard.get('exit_code')) is not int:
        raise ValueError('Require a verified wall-interrupted parent, never a live task')
    if read(parent/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped':
        raise ValueError('Parent owned-server cleanup is incomplete')
    folder=parent/method;summary=read(folder/'summary.json')
    if summary['stopped_reason'] is not None or not summary['reload_exact'] or (not original_goals and (folder/'evaluation.json').exists()):
        raise ValueError('Require completed training but incomplete endpoint evaluation')
    name=read(folder/'latest_checkpoint.json')['folder']
    if Path(name).name!=name:raise ValueError('Checkpoint must be a direct child')
    origin=folder/name;metadata=verified_checkpoint_metadata(origin)
    if any(metadata[k]!=summary[k] for k in ('physics_steps','actions','optimizer_steps')):
        raise ValueError('Checkpoint is not the final trained endpoint')
    if read(origin/'config.json')!=read(folder/'config.json'):
        raise ValueError('Saved task semantics changed')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed')
    if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:
        raise ValueError('Calibration changed')
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    with ProcessLease(parent/'evaluation_recovery_prepare.lock'):
        registration='original_goals_registration.json' if original_goals else 'evaluation_recovery_registration.json'
        if (parent/registration).exists():
            raise FileExistsError('Preserve the already registered evaluation recovery')
        if original_goals:
            complete=read(parent/method/'endpoint_raw_audit.json')
            prefix={'scope':'Complete endpoint audit, not an interrupted prefix','endpoint_raw_audit':complete,
                'bridge_streams':{complete['bridge_response_stream']:{'applied_controls':complete['total_applied_controls_verified']}}}
        else:
            if not (parent/'interruption_audit.json').exists():interrupted_audit(parent)
            prefix=read(parent/'interruption_audit.json')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_evaluation_recovery')
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent/'calibration',batch/'calibration')
        destination=batch/'resume'/method;destination.parent.mkdir()
        shutil.copytree(origin,destination)
        if verified_checkpoint_metadata(destination)!=metadata:raise ValueError('Copied checkpoint changed')
        overrides=['native_road_learning.py','audit_native_method_endpoint.py','watch_existing_pair.py','prepare_native_evaluation_recovery.py','run_carla_owned_foreground.ps1']
        for name in overrides:shutil.copy2(ROOT/'scripts'/name,batch/'source/scripts'/name)
        write_json(batch/'parent_interruption_audit.json',prefix)
        write_json(batch/'parent_training_summary.json',summary)
        write_json(batch/'parent_protocol.json',old)
        protocol=deepcopy(old);now=time.time()
        for key in ('jobs','actor_head_experiment','gradient_clipping_experiment','rollout_length_experiment'):
            protocol.pop(key,None)
        protocol.pop('freeze_observation_statistics',None);protocol.pop('own_success_replay',None)
        prior=old.get('recovery',{}).get(method,{})
        protocol.update(scope='Frozen saved-endpoint reevaluation after wall interruption; zero new training/optimization; old partial attempts retained and charged, not independent conditions',
            evaluation_only=True,methods=[method],controlled_budget_per_method=0,method_budgets={method:0},checkpoint_thresholds=[],
            created_epoch=now,deadline_epoch=now+1800,parent_batch=parent.name,
            parent_interruption_audit_sha256=sha(batch/'parent_interruption_audit.json'),
            prior_applied_controls_readable_lower_bound=sum(v['applied_controls'] for v in prefix['bridge_streams'].values()),
            prior_attempt_scope='Original partial evaluation/streams remain immutable; all4cases reevaluated because original batch is not a complete audited endpoint',
            recovery={method:{'checkpoint':'resume/'+method,'parent_batch':parent.name,'parent_checkpoint':origin.name,
                'lineage_controls':summary['lineage_control_steps'],'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound'],
                'retained_optimizer_steps':prior.get('retained_optimizer_steps',0)+summary['optimizer_steps']}},
            harness_overrides=['scripts/'+n for n in overrides],resource_limits={**old['resource_limits'],'wall_seconds':1800})
        if original_goals:
            protocol.update(routes=deepcopy(old['curriculum']['original_routes']),original_goal_transfer=True,
                scope='Frozen curriculum endpoint transferred back to original known six-metre goals; identical model/normalizer bytes, zero new training or auxiliary optimization, not unseen confirmation',
                prior_attempt_scope='Complete curriculum endpoint and all failures retained; this new phase changes goals only')
        protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}
        write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        write_json(parent/registration,{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')})
        write_json(ROOT/'runs/native_evaluation_recovery.json',{'batch':str(batch)})
    print(batch,flush=True);return batch


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--method',default='H')
    p.add_argument('--original-goals',action='store_true')
    args=p.parse_args();prepare(args.parent,args.method,args.original_goals)

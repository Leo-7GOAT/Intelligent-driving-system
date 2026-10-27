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


def evaluation_lineage(prior, summary, parent_name, checkpoint_name, method):
    """A frozen replay keeps uncertainty and consumed costs from every ancestor."""
    chain=list(prior.get('recovery_chain',[]))
    if not chain or chain[-1]!=parent_name:
        chain.append(parent_name)
    return {'checkpoint':'resume/'+method,'parent_batch':parent_name,'parent_checkpoint':checkpoint_name,
            'lineage_controls':summary['lineage_control_steps'],
            'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound'],
            'retained_optimizer_steps':prior.get('retained_optimizer_steps',0)+summary['optimizer_steps'],
            'unknown_unflushed_parent_tail':bool(prior.get('unknown_unflushed_parent_tail',False)),
            'recovery_chain':chain}


def verified_interruption(parent):
    """Accept normal wall receipts or fresh, separately verified process loss."""
    final=parent/'guard_final.json'
    if final.exists():
        guard=read(final)
        disk_stop = (guard.get('reason') is None and type(guard.get('exit_code')) is int
                     and guard['exit_code'] == 1 and (parent/'failure.json').is_file()
                     and read(parent/'failure.json').get('error') == "RuntimeError('Disk below12GiB')")
        if not disk_stop and (guard.get('reason')!='wall_deadline' or type(guard.get('exit_code')) is not int):
            raise ValueError('Require a verified resource-interrupted parent, never a live task')
        if read(parent/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped':
            raise ValueError('Parent owned-server cleanup is incomplete')
        if disk_stop:
            from navstack.process_lease import pid_alive
            worker=read(parent/'worker_process.json');watch=read(parent/'guard.json')
            server=read(parent/'infrastructure/server_start.json')
            stop=read(parent/'infrastructure/server_stop.json')
            identities=(worker['pid'],watch['pid'],server['shipping_pid'],server['launcher_pid'])
            if (guard.get('worker_pid')!=worker['pid'] or not server.get('owned_by_task')
                    or stop.get('shipping_pid')!=server['shipping_pid']
                    or stop.get('post_stop_tcp')!=[] or stop.get('post_stop_udp')!=[]
                    or any(type(pid) is not int or pid<=0 or pid_alive(pid) for pid in identities)):
                raise ValueError('Resource-interrupted processes or cleanup are live, ambiguous or mismatched')
            return 'worker_disk_below_12GiB'
        return 'wall_deadline'
    from navstack.process_lease import pid_alive
    receipt=read(parent/'process_absence_receipt.json')
    if (receipt.get('status')!='verified_worker_guard_server_absent'
            or receipt.get('reason')!='external_process_loss_cause_unknown'
            or not 0<=time.time()-receipt['observed_epoch']<=300
            or any(receipt.get(k) for k in ('tcp_endpoints','udp_endpoints','related_bridges'))):
        raise ValueError('External-loss recovery requires a fresh empty-process/port receipt')
    expected={'protocol.json','worker_process.json','guard.json','infrastructure/server_start.json'}
    if set(receipt.get('source_hashes',{}))!=expected or any(sha(parent/name)!=digest for name,digest in receipt['source_hashes'].items()):
        raise ValueError('External-loss source identity records changed')
    worker=read(parent/'worker_process.json');guard=read(parent/'guard.json');server=read(parent/'infrastructure/server_start.json')
    identities={'worker_pid':worker['pid'],'guard_pid':guard['pid'],'shipping_pid':server['shipping_pid'],'launcher_pid':server['launcher_pid']}
    port=read(parent/'protocol.json')['carla_rpc_port']
    if (not server.get('owned_by_task') or receipt.get('process_ids')!=identities
            or receipt.get('protocol_port_span')!=list(range(port,port+3))
            or any(type(pid) is not int or pid<=0 or pid_alive(pid) for pid in identities.values())):
        raise ValueError('Recorded processes are live, ambiguous or differ from absence receipt')
    return 'external_process_loss_cause_unknown'


def prepare(parent,method='H',original_goals=False):
    parent=Path(parent).resolve();old=read(parent/'protocol.json')
    interruption=None
    if original_goals:
        require_finished_parent(parent)
        if 'curriculum' not in old or endpoint_audit(parent,method)!=read(parent/method/'endpoint_raw_audit.json'):
            raise ValueError('Require a complete audited curriculum endpoint')
    else:
        interruption=verified_interruption(parent)
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
        for key in ('jobs','actor_head_experiment','gradient_clipping_experiment','rollout_length_experiment','fresh_cadence_experiment'):
            protocol.pop(key,None)
        protocol.pop('freeze_observation_statistics',None);protocol.pop('own_success_replay',None)
        prior=old.get('recovery',{}).get(method,{})
        protocol.update(scope='Frozen saved-endpoint reevaluation after verified interruption; zero new training/optimization; old partial attempts retained and charged, not independent conditions',
            parent_interruption_kind=interruption,
            evaluation_only=True,methods=[method],controlled_budget_per_method=0,method_budgets={method:0},checkpoint_thresholds=[],
            created_epoch=now,deadline_epoch=now+1800,parent_batch=parent.name,
            parent_interruption_audit_sha256=sha(batch/'parent_interruption_audit.json'),
            prior_applied_controls_readable_lower_bound=sum(v['applied_controls'] for v in prefix['bridge_streams'].values()),
            prior_attempt_scope='Original partial evaluation/streams remain immutable; all4cases reevaluated because original batch is not a complete audited endpoint',
            recovery={method:evaluation_lineage(prior,summary,parent.name,origin.name,method)},
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

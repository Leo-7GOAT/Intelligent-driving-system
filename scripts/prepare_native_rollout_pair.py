"""Stage a single-factor 32-versus-256-decision native PPO development study.

Use the global-clipping endpoints for every method, regardless of observed
clipping-study performance. All six preceding arms must finish and be audited.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from navstack.experiments import write_json
from scripts.continue_native_learning import (read,sha,verified_endpoint,write_json_atomic,
    assert_continuation_pointer_available,assert_parent_endpoints_unclaimed,_record_abort)
from scripts.combine_native_endpoints import _claim_endpoints_atomic
from scripts.audit_native_method_endpoint import audit
from scripts.native_road_learning import registered_jobs

REFRESH_FILES=['navstack/independent_clip_ppo.py']+['scripts/'+name for name in (
    'native_road_learning.py','prepare_native_recovery.py','prepare_native_rollout_pair.py',
    'watch_existing_pair.py','run_carla_owned_foreground.ps1','stop_owned_carla.ps1')]


def parent_manifest(parent,additional):
    protocol=read(parent/'protocol.json')
    expected=[m+'_'+arm for m in ('H0','H','B1') for arm in ('global','independent')]
    if protocol.get('methods')!=expected or not protocol.get('gradient_clipping_experiment'):
        raise ValueError('Require the complete registered six-arm clipping study')
    if protocol.get('rollout_actions')!=32 or protocol.get('ppo_epochs')!=2:
        raise ValueError('Expected the unchanged short-rollout parent schedule')
    if (parent/'failure.json').exists() or read(parent/'summary.json')['status']!='complete':
        raise ValueError('All parent arms must finish normally')
    guard=read(parent/'guard_final.json')
    if type(guard.get('exit_code')) is not int or guard['exit_code']!=0 or guard.get('reason') is not None:
        raise ValueError('Parent lifecycle is not normally terminal')
    if read(parent/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped':
        raise ValueError('Parent owned-server cleanup is incomplete')
    for name,digest in protocol['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent frozen source changed')
    if sha(parent/'calibration/calibration.json')!=protocol['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    audits={}
    for name in expected:
        audits[name]=audit(parent,name)
        if audits[name]!=read(parent/name/'endpoint_raw_audit.json'):
            raise ValueError('Missing or stale parent raw audit: '+name)
    records={}
    for method in ('H0','H','B1'):
        name=method+'_global'
        if next(j for j in protocol['jobs'] if j['name']==name)['gradient_clip_mode']!='global':
            raise ValueError('Parent selection must retain global clipping')
        old=read(parent/name/'summary.json')
        checkpoint,summary,metadata=verified_endpoint(parent/name,old['lineage_control_steps']+additional)
        import zipfile
        with zipfile.ZipFile(checkpoint/'model.zip') as archive:
            learner=json.loads(archive.read('data'))
        if any(learner[key]!=value for key,value in {'n_steps':32,'batch_size':32,'n_epochs':2}.items()):
            raise ValueError('Saved learner schedule differs from the registered parent')
        prior=protocol['recovery'][name]
        records[method]={'method':name,'source_batch':{'root_relative':parent.relative_to(ROOT).as_posix()},
            'checkpoint':checkpoint.name,'source_method_folder':name,
            'checkpoint_hashes':{p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()},
            'lineage_controls':summary['lineage_control_steps'],
            'charged_controls_lower_bound':summary['charged_control_steps_lower_bound'],
            'optimizer_steps_total_lower_bound':audits[name]['optimizer_steps_total_lower_bound'],
            'unknown_unflushed_ancestor_tail':audits[name]['unknown_unflushed_ancestor_tail'],
            'recovery_chain':prior['recovery_chain'],
            'raw_audit_sha256':sha(parent/name/'endpoint_raw_audit.json')}
    manifest={'kind':'native_rollout_global_parent_selection','sources':records,
        'target_controls':max(r['lineage_controls'] for r in records.values())+additional,
        'selection':'Global-clipping parents for all three methods; no selection by outcome',
        'all_six_raw_audit_hashes':{name:sha(parent/name/'endpoint_raw_audit.json') for name in expected}}
    manifest['manifest_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return manifest


def prepare(parent,additional_controls=102400,wall_seconds=21600):
    parent=Path(parent).resolve()
    if not 51200<=additional_controls<=204800 or additional_controls%2560 or not 7200<=wall_seconds<=21600:
        raise ValueError('Rollout study budget outside registered bounds')
    ok,reason=resources_ok(ROOT)
    if not ok or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError('Resource guard: '+str(reason))
    runs=ROOT/'runs'
    with ProcessLease(runs/'native_continuation.prepare.lock'):
        assert_continuation_pointer_available(runs)
        pointer=runs/'native_rollout_pair.json'
        if pointer.exists():
            previous=Path(read(pointer)['batch'])
            if not (previous/'guard_final.json').is_file() or type(read(previous/'guard_final.json').get('exit_code')) is not int:
                raise RuntimeError('Previous rollout study has no terminal guard receipt')
        assert_parent_endpoints_unclaimed(runs,parent)
        manifest=parent_manifest(parent,additional_controls)
        batch=runs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_rollout_pair')
        batch.mkdir()
        try:
            (batch/'infrastructure').mkdir()
            shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(parent/'calibration',batch/'calibration')
            for name in REFRESH_FILES:shutil.copy2(ROOT/name,batch/'source'/name)
            jobs,recovery,budgets,targets=[],{},{},{}
            for method,record in manifest['sources'].items():
                for size in (32,256):
                    name=method+'_rollout'+str(size)
                    jobs.append({'name':name,'method':method,'gradient_clip_mode':'global','rollout_actions':size})
                    destination=batch/'resume'/name
                    shutil.copytree(parent/record['source_method_folder']/record['checkpoint'],destination)
                    if {p.name:sha(p) for p in destination.iterdir() if p.is_file()}!=record['checkpoint_hashes']:
                        raise ValueError('Staged checkpoint changed')
                    recovery[name]={'checkpoint':'resume/'+name,'parent_batch':parent.name,
                        'parent_checkpoint':record['checkpoint'],'parent_job':record['source_method_folder'],
                        'lineage_controls':record['lineage_controls'],
                        'consumed_controls_lower_bound':record['charged_controls_lower_bound'],
                        'retained_optimizer_steps':record['optimizer_steps_total_lower_bound'],
                        'unknown_unflushed_parent_tail':record['unknown_unflushed_ancestor_tail'],
                        'recovery_chain':record['recovery_chain']+[parent.name],
                        'model_sha256':record['checkpoint_hashes']['model.zip'],
                        'normalization_sha256':record['checkpoint_hashes']['normalization.pkl']}
                    budgets[name]=additional_controls
                    targets[name]=record['lineage_controls']+additional_controls
            protocol=deepcopy(read(parent/'protocol.json'))
            protocol.pop('gradient_clipping_experiment',None)
            now=time.time()
            protocol.update(rollout_length_experiment=True,methods=[j['name'] for j in jobs],jobs=jobs,
                parent_batch=parent.name,parent_records=manifest['sources'],
                parent_endpoint_manifest_id=manifest['manifest_id'],created_epoch=now,deadline_epoch=now+wall_seconds,
                recovery=recovery,method_budgets=budgets,lineage_targets=targets,controlled_budget_per_method=max(targets.values()),
                checkpoint_thresholds=[25600,51200,additional_controls],
                scope='Single-factor rollout-length development study; same-parent correlated forks, not independent seeds or final confirmation',
                hypothesis='Longer on-policy buffers reduce endpoint-value dependence and improve task completion/progress without worse safety',
                primary_development_measure='All four frozen training-road outcomes, progress, safety and intervention; not KL alone',
                rollout_budget_note='Same nominal applied-control budgets; complete-rollout overshoot retained and reported. Longer groups change update timing and GAE, while minibatch32/epochs2/global clipping remain fixed.',
                continuation_reason='Registered32-versus256decision on-policy collection; no reward, observation, controller, epoch or learning-rate change',
                source_hashes={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*')
                               if p.is_file() and p.suffix in ('.py','.ps1')},disk_free_bytes=shutil.disk_usage(ROOT).free)
            protocol['resource_limits']['wall_seconds']=wall_seconds
            registered_jobs(protocol)
            write_json(batch/'protocol.json',protocol)
            write_json(batch/'parent_endpoint_manifest.json',manifest)
            write_json(batch/'prepared.json',{'status':'PREPARED_UNREGISTERED','protocol_sha256':sha(batch/'protocol.json')})
        except BaseException as exc:
            _record_abort(batch,parent.name,'rollout_pair_staging',exc)
            raise
        try:claims=_claim_endpoints_atomic(runs,manifest,batch)
        except BaseException as exc:
            _record_abort(batch,parent.name,'rollout_pair_registration',exc)
            raise
        write_json_atomic(batch/'registered.json',{'status':'REGISTERED_READY','epoch':time.time(),**claims})
        try:write_json_atomic(pointer,{'batch':str(batch)})
        except BaseException as exc:
            write_json(batch/'pointer_publication_failure.json',{'status':'REGISTERED_CHILD_POINTER_NOT_PUBLISHED',
                'claims':claims,'epoch':time.time(),'error_type':type(exc).__name__,'error':str(exc)})
            raise
        print(batch,flush=True)
        return batch


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True)
    parser.add_argument('--additional-controls',type=int,default=102400)
    parser.add_argument('--wall-seconds',type=int,default=21600)
    args=parser.parse_args()
    prepare(args.parent,args.additional_controls,args.wall_seconds)

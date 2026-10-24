"""Register only unfinished native methods, retaining wasted interaction costs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.process_lease import pid_alive
from navstack.research import resources_ok
from navstack.experiments import write_json


def claim_recovery(parent, child_name):
    """One explicit recovery per parent; another interruption continues its child."""
    with (Path(parent)/'recovery_registration.json').open('x',encoding='utf-8') as stream:
        json.dump({'child_batch':child_name,'epoch':time.time(),
                   'note':'Do not replay this parent again; inspect or recover the registered child'},stream,indent=2)


def write_json_atomic(path, value):
    """Replace a small pointer only after its target batch is complete."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    write_json(temporary, value)
    temporary.replace(path)


def prepare(parent, wall_seconds, rpc_port=None):
    parent=Path(parent).resolve()
    if not 900<=wall_seconds<=7200:raise ValueError('Recovery wall budget outside bounds')
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    old=json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
    if rpc_port is None:
        rpc_port=old.get('carla_rpc_port',2000)
    if isinstance(rpc_port,bool) or not isinstance(rpc_port,int) or not 1024<=rpc_port<=65533:
        raise ValueError('CARLA RPC base port must be in 1024..65533')
    old['carla_rpc_host']='127.0.0.1'
    old['carla_rpc_port']=rpc_port
    old['carla_rpc_port_span']=list(range(rpc_port,rpc_port+3))
    audit=json.loads((parent/'interruption_audit.json').read_text(encoding='utf-8'))
    for path in parent.glob('*/heartbeat.json'):
        heartbeat=json.loads(path.read_text(encoding='utf-8'))
        if pid_alive(heartbeat['pid']):raise RuntimeError('Parent PID live or ambiguous; inspect identity before recovery')
    for relative,digest in old['source_hashes'].items():
        if hashlib.sha256((parent/'source'/relative).read_bytes()).hexdigest()!=digest:raise ValueError('Parent source changed')
    if (parent/'recovery_registration.json').exists():
        raise FileExistsError('Parent already has a registered recovery; continue that child instead')
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_recovery')
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'calibration',batch/'calibration')
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    # Vehicle/reward/geometry/PPO semantics stay at the parent frozen version.
    # Only orchestration/checkpointing is replaced and fully rehashed. The
    # lifecycle files are frozen too because a successful worker exit alone is
    # not evidence that the owned simulator was safely released.
    shutil.copy2(ROOT/'navstack/carla_bridge.py',batch/'source/navstack/carla_bridge.py')
    shutil.copy2(ROOT/'navstack/independent_clip_ppo.py',batch/'source/navstack/independent_clip_ppo.py')
    for name in ['native_road_learning.py','prepare_native_recovery.py','watch_existing_pair.py','carla_bridge37.py']:
        shutil.copy2(ROOT/'scripts'/name,batch/'source/scripts'/name)
    for name in ['start_carla.ps1','diagnose_carla_start.ps1','run_carla_owned_foreground.ps1','stop_owned_carla.ps1']:
        shutil.copy2(ROOT/'scripts'/name,batch/'source/scripts'/name)
    shutil.copy2(parent/'interruption_audit.json',batch/'parent_interruption_audit.json')
    recovery,methods,budgets={},{},{}
    for method in old['methods']:
        target=old.get('lineage_targets',{}).get(method,old['controlled_budget_per_method'])
        status=audit['methods'][method]
        if status['state']=='complete':continue
        methods[method]=status['state']
        budgets[method]=target
        prior=old.get('recovery',{}).get(method)
        if status['state']=='not_started' and prior:
            source=parent/prior['checkpoint']
            destination=batch/'resume'/method
            shutil.copytree(source,destination)
            metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
            for filename,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),
                                 ('resume_state.pkl','resume_state_sha256')]:
                if hashlib.sha256((source/filename).read_bytes()).hexdigest()!=metadata[key]:
                    raise ValueError('Carried recovery checkpoint hash mismatch')
            lineage=prior['lineage_controls']
            if lineage>=target:raise ValueError('Carried policy already reached target')
            budgets[method]=target-lineage
            chain=list(prior.get('recovery_chain',[]))
            if not chain and old.get('parent_batch'):
                chain.append(old['parent_batch'])
            if not chain or chain[-1]!=parent.name:
                chain.append(parent.name)
            recovery[method]={**prior,'checkpoint':'resume/'+method,'parent_batch':parent.name,
                'recovery_chain':chain}
            continue
        if status['state']=='interrupted':
            if not status['checkpoints']:raise ValueError('No verified checkpoint; explicit new experiment required')
            saved=max(status['checkpoints'],key=lambda c:c['physics_steps'])
            source=parent/method/saved['folder']
            destination=batch/'resume'/method
            shutil.copytree(source,destination)
            shutil.copy2(parent/method/'config.json',destination/'config.json')
            metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
            lineage=metadata.get('lineage_control_steps',metadata['physics_steps'])
            if lineage>=target:raise ValueError('Training already at target; recover evaluation separately')
            budgets[method]=target-lineage
            inherited_consumed=prior.get('consumed_controls_lower_bound',0) if prior else 0
            inherited_optimizer=prior.get('retained_optimizer_steps',0) if prior else 0
            chain=list(prior.get('recovery_chain',[])) if prior else []
            if not chain and old.get('parent_batch'):
                chain.append(old['parent_batch'])
            if not chain or chain[-1]!=parent.name:
                chain.append(parent.name)
            recovery[method]={'checkpoint':'resume/'+method,'parent_batch':parent.name,'parent_checkpoint':saved['folder'],
                'lineage_controls':lineage,
                'consumed_controls_lower_bound':max(metadata.get('charged_control_steps_lower_bound',0),
                    inherited_consumed+status['consumed_controls_lower_bound']),
                'unknown_unflushed_parent_tail':status['unknown_unflushed_tail'] or bool(prior and prior.get('unknown_unflushed_parent_tail')),
                'retained_optimizer_steps':inherited_optimizer+metadata['optimizer_steps'],
                'model_sha256':metadata['model_sha256'],'normalization_sha256':metadata['normalization_sha256'],
                'recovery_chain':chain}
    if not methods:raise ValueError('No unfinished methods')
    hashes={p.relative_to(batch/'source').as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (batch/'source').rglob('*') if p.is_file() and p.suffix in ('.py','.ps1')}
    now=time.time()
    old.update(created_epoch=now,deadline_epoch=now+wall_seconds,methods=list(methods),method_budgets=budgets,
        source_hashes=hashes,recovery=recovery,parent_batch=parent.name,
        scope='Interrupted development recovery; retained policy lineage and all consumed interactions are separate. Not a budget-matched superiority comparison.',
        disk_free_bytes=shutil.disk_usage(ROOT).free)
    old['resource_limits']['wall_seconds']=wall_seconds
    if 'jobs' in old:
        old['jobs']=[job for job in old['jobs'] if job['name'] in methods]
    write_json(batch/'protocol.json',old)
    # Register only after every immutable input, checkpoint and protocol file
    # exists.  A preparation failure therefore cannot strand the parent with
    # a registration pointing at a partial or absent child.
    try:
        claim_recovery(parent,batch.name)
    except BaseException as exc:
        write_json(batch/'registration_failure.json',{
            'epoch':time.time(),'parent_batch':parent.name,'child_batch':batch.name,
            'error_type':type(exc).__name__,'error':str(exc),
            'status':'UNREGISTERED_COMPLETE_CHILD_DO_NOT_RUN'})
        raise
    write_json_atomic(ROOT/'runs/native_recovery.json',{'batch':str(batch)})
    print(batch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--wall-seconds',type=int,default=7200)
    p.add_argument('--rpc-port',type=int)
    a=p.parse_args();prepare(a.parent,a.wall_seconds,a.rpc_port)

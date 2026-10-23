"""Register only unfinished native methods, retaining wasted interaction costs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
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


def prepare(parent, wall_seconds):
    parent=Path(parent).resolve()
    if not 900<=wall_seconds<=7200:raise ValueError('Recovery wall budget outside bounds')
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    old=json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
    audit=json.loads((parent/'interruption_audit.json').read_text(encoding='utf-8'))
    for path in parent.glob('*/heartbeat.json'):
        heartbeat=json.loads(path.read_text(encoding='utf-8'))
        if pid_alive(heartbeat['pid']):raise RuntimeError('Parent PID live or ambiguous; inspect identity before recovery')
    for relative,digest in old['source_hashes'].items():
        if hashlib.sha256((parent/'source'/relative).read_bytes()).hexdigest()!=digest:raise ValueError('Parent source changed')
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_recovery')
    claim_recovery(parent,batch.name)
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'calibration',batch/'calibration')
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    # Vehicle/reward/geometry/PPO semantics stay at the parent frozen version.
    # Only orchestration/checkpointing is replaced and fully rehashed.
    for name in ['native_road_learning.py','prepare_native_recovery.py']:
        shutil.copy2(ROOT/'scripts'/name,batch/'source/scripts'/name)
    shutil.copy2(parent/'interruption_audit.json',batch/'parent_interruption_audit.json')
    recovery,methods,budgets={},{},{}
    target=old['controlled_budget_per_method']
    for method in old['methods']:
        status=audit['methods'][method]
        if status['state']=='complete':continue
        methods[method]=status['state']
        budgets[method]=target
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
            recovery[method]={'checkpoint':'resume/'+method,'parent_batch':parent.name,'parent_checkpoint':saved['folder'],
                'lineage_controls':lineage,'consumed_controls_lower_bound':status['consumed_controls_lower_bound'],
                'unknown_unflushed_parent_tail':status['unknown_unflushed_tail'],
                'retained_optimizer_steps':metadata['optimizer_steps'],
                'model_sha256':metadata['model_sha256'],'normalization_sha256':metadata['normalization_sha256']}
    if not methods:raise ValueError('No unfinished methods')
    hashes={p.relative_to(batch/'source').as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (batch/'source').rglob('*.py')}
    now=time.time()
    old.update(created_epoch=now,deadline_epoch=now+wall_seconds,methods=list(methods),method_budgets=budgets,
        source_hashes=hashes,recovery=recovery,parent_batch=parent.name,
        scope='Interrupted development recovery; retained policy lineage and all consumed interactions are separate. Not a budget-matched superiority comparison.',
        disk_free_bytes=shutil.disk_usage(ROOT).free)
    old['resource_limits']['wall_seconds']=wall_seconds
    write_json(batch/'protocol.json',old)
    write_json(ROOT/'runs/native_recovery.json',{'batch':str(batch)})
    print(batch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--wall-seconds',type=int,default=7200)
    a=p.parse_args();prepare(a.parent,a.wall_seconds)

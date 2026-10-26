"""One fresh dense-progress B1 endpoint against the completed sparse control."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import audit, read, sha
from scripts.evaluate_native_rule import require_finished_parent


def prepare(parent, longitudinal_mode='acceleration'):
    if longitudinal_mode not in ('acceleration','speed_target_v1'):
        raise ValueError('Unknown registered longitudinal variant')
    speed_trial=longitudinal_mode=='speed_target_v1'
    parent=Path(parent).resolve()
    require_finished_parent(parent)
    old=read(parent/'protocol.json')
    if old.get('recovery') or (not speed_trial and not old.get('flat_distribution_experiment')) or (speed_trial and old.get('methods')!=['B1']):
        raise ValueError('Require the complete fresh distribution sanity pair')
    name='B1' if speed_trial else 'B1_clipped_gaussian'
    if old.get('controlled_budget_per_method')!=4096 or old.get('task_progress_weight',0)!=(1 if speed_trial else 0):
        raise ValueError('Require the fixed 4096-control zero-progress parent')
    if audit(parent,name)!=read(parent/name/'endpoint_raw_audit.json'):
        raise ValueError('Parent raw audit absent or stale')
    for relative,digest in old['source_hashes'].items():
        if sha(parent/'source'/relative)!=digest:raise ValueError('Parent source changed')
    if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<12*1024**3:
        raise RuntimeError('Resource guard: '+str(reason))
    with ProcessLease(ROOT/'runs/native_progress_sanity_prepare.lock'):
        label='native_speed_sanity' if speed_trial else 'native_progress_sanity'
        pointer=ROOT/'runs'/(label+'.json')
        if pointer.exists():raise FileExistsError('Preserve existing progress sanity registration')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+label)
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'calibration',batch/'calibration')
        for folder in ('navstack','scripts','tests'):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and (
                        source.suffix in ('.py','.ps1') or 'fixtures' in source.parts):
                    target=batch/'source'/source.relative_to(ROOT)
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        protocol=deepcopy(old)
        protocol.pop('jobs',None);protocol.pop('flat_distribution_experiment',None)
        now=time.time()
        protocol.update(scope='Fresh six-metre B1 sanity: measured task progress weight1 versus completed weight0 control; not full-task competence',
            methods=['B1'],task_progress_weight=1.,parent_batch=parent.name,
            parent_control_method=name,parent_control_raw_audit_sha256=sha(parent/name/'endpoint_raw_audit.json'),
            created_epoch=now,deadline_epoch=now+900,
            hypothesis='Terminal-zero potential cancels distance credit on failed episodes; explicit shared task progress can supply learning credit before first success',
            comparison='Fresh seed11 clipped-Gaussian, same tasks/network/normalization/PPO/budget as completed clipped-Gaussian parent; only task reward progress_weight changes0to1',
            differences_from_parent=['task reward progress_weight0to1; shared option available to H/H0/B1, no architecture comparison claimed'],
            stopping_rule='One fixed4096-control endpoint and all4sanity probes; no automatic retry. Do not infer full-route competence or expand without observed learning.',
            resource_limits={**old['resource_limits'],'wall_seconds':900})
        if speed_trial:
            protocol.update(flat_longitudinal_mode=longitudinal_mode,
                scope='Fresh six-metre B1 speed/steering sanity versus completed acceleration/steering control; no planner or MPC, not full-task competence',
                hypothesis='Speed-target actions supply coherent forward exploration without requiring a zero-centred Gaussian to first learn sustained positive acceleration',
                comparison='Same fresh seed11 clipped-Gaussian/network/normalization/reward/PPO/4096control budget/short roads; only longitudinal action parameterization and its disclosed proportional servo differ',
                differences_from_parent=['acceleration to speed_target_v1; target_speed=(action0+1)*vmax/2; acceleration=clip(target_speed-measured_speed,-max_decel,max_accel); no goal or route logic'],
                stopping_rule='One4096-control endpoint and all4probes; all failures retained, zero automatic retries. No full-task health or hierarchy claim.')
        protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p)
            for p in (batch/'source').rglob('*') if p.is_file()}
        write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        write_json(pointer,{'batch':str(batch),'protocol_sha256':sha(batch/'protocol.json')})
    print(batch,flush=True)
    return batch


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True)
    parser.add_argument('--longitudinal-mode',choices=['acceleration','speed_target_v1'],default='acceleration')
    args=parser.parse_args();prepare(args.parent,args.longitudinal_mode)

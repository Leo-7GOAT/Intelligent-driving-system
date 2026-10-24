"""Register a paired, single-factor native gradient-clipping development study.

Each completed H0/H/B1 endpoint seeds a global-clip and independent-clip arm.
Every arm retains the same weights, Adam/RNG/normalization state and inherited
costs. This correlated continuation cannot count as an independent training seed.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.experiments import write_json
from navstack.independent_clip_ppo import GLOBAL_CLIP, INDEPENDENT_CLIP
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.combine_native_endpoints import build_manifest, _claim_endpoints_atomic
from scripts.continue_native_learning import (read, sha, write_json_atomic,
    assert_continuation_pointer_available, assert_parent_endpoints_unclaimed, _record_abort)
from scripts.native_road_learning import registered_jobs


def prepare(parent, additional_controls=20000, wall_seconds=21600):
    parent = Path(parent).resolve()
    if not 10000 <= additional_controls <= 50000 or not 1800 <= wall_seconds <= 21600:
        raise ValueError('Gradient-pair budget outside registered bounds')
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError('Resource guard: '+str(reason))
    runs = ROOT/'runs'
    with ProcessLease(runs/'native_continuation.prepare.lock'):
        assert_continuation_pointer_available(runs)
        pointer = runs/'native_gradient_pair.json'
        if pointer.exists():
            previous = Path(read(pointer)['batch'])
            if not (previous/'guard_final.json').exists():
                raise RuntimeError('Previous gradient pair has no terminal guard receipt')
            if type(read(previous/'guard_final.json').get('exit_code')) is not int:
                raise RuntimeError('Previous gradient pair is not verified terminal')
        protocol = read(parent/'protocol.json')
        if protocol['methods'] != ['H0','H','B1'] or protocol.get('jobs'):
            raise ValueError('Expected the completed three-method structured native study')
        if (parent/'failure.json').exists() or read(parent/'summary.json')['status'] != 'complete':
            raise ValueError('Parent must finish normally before branching')
        guard = read(parent/'guard_final.json')
        if type(guard.get('exit_code')) is not int or guard['exit_code'] != 0 or guard.get('reason') is not None:
            raise ValueError('Parent guard did not finish normally')
        if read(parent/'infrastructure/server_stop.json')['status'] != 'owned_empty_server_stopped':
            raise ValueError('Parent server cleanup is not verified')
        assert_parent_endpoints_unclaimed(runs, parent)
        maximum = max(read(parent/m/'summary.json')['lineage_control_steps'] for m in protocol['methods'])
        manifest = build_manifest({m: parent/m for m in protocol['methods']}, maximum+additional_controls, root=ROOT)
        batch = runs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_gradient_pair')
        batch.mkdir()
        try:
            (batch/'infrastructure').mkdir()
            shutil.copytree(parent/'source', batch/'source', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(parent/'calibration', batch/'calibration')
            # Only the opt-in optimizer and experiment/checkpoint orchestration
            # change. Environment, reward, observation, planner and MPC stay frozen.
            refresh = ['navstack/independent_clip_ppo.py'] + ['scripts/'+name for name in (
                'native_road_learning.py','prepare_native_gradient_pair.py','prepare_native_recovery.py',
                'continue_native_learning.py','combine_native_endpoints.py','watch_existing_pair.py',
                'audit_native_method_endpoint.py','audit_native_policy_inference.py',
                'diagnose_native_action_learning.py','audit_native_rollout_update.py',
                'run_carla_owned_foreground.ps1','stop_owned_carla.ps1')]
            for relative in refresh:
                shutil.copy2(ROOT/relative, batch/'source'/relative)
            jobs, recovery, budgets, targets = [], {}, {}, {}
            for method in protocol['methods']:
                record = manifest['sources'][method]
                checkpoint = parent/method/record['checkpoint']
                for label, mode in (('global', GLOBAL_CLIP), ('independent', INDEPENDENT_CLIP)):
                    name = method+'_'+label
                    jobs.append({'name':name,'method':method,'gradient_clip_mode':mode})
                    destination = batch/'resume'/name
                    shutil.copytree(checkpoint, destination)
                    if {p.name:sha(p) for p in destination.iterdir() if p.is_file()} != record['checkpoint_hashes']:
                        raise ValueError('Copied branch checkpoint changed')
                    recovery[name] = {'checkpoint':'resume/'+name,'parent_batch':parent.name,
                        'parent_checkpoint':record['checkpoint'],'lineage_controls':record['lineage_controls'],
                        'consumed_controls_lower_bound':record['charged_controls_lower_bound'],
                        'retained_optimizer_steps':record['optimizer_steps_total_lower_bound'],
                        'unknown_unflushed_parent_tail':record['unknown_unflushed_ancestor_tail'],
                        'recovery_chain':record['recovery_chain']+[parent.name],
                        'model_sha256':record['checkpoint_hashes']['model.zip'],
                        'normalization_sha256':record['checkpoint_hashes']['normalization.pkl']}
                    budgets[name] = additional_controls
                    targets[name] = record['lineage_controls']+additional_controls
            now = time.time()
            child = deepcopy(protocol)
            child.pop('parent_batches', None)
            child.pop('combined_endpoint_manifest_id', None)
            child.pop('combined_endpoint_manifest_sha256', None)
            child.update(methods=[job['name'] for job in jobs], jobs=jobs,
                gradient_clipping_experiment=True, parent_batch=parent.name,
                parent_endpoint_manifest_id=manifest['manifest_id'],
                continuation_reason='Registered same-endpoint paired clipping intervention; no other learning or task change',
                parent_records=manifest['sources'], created_epoch=now, deadline_epoch=now+wall_seconds,
                method_budgets=budgets, lineage_targets=targets, recovery=recovery,
                controlled_budget_per_method=max(targets.values()), checkpoint_thresholds=[10000,additional_controls],
                source_hashes={p.relative_to(batch/'source').as_posix():sha(p)
                    for p in (batch/'source').rglob('*') if p.is_file() and p.suffix in ('.py','.ps1')},
                scope='Paired single-factor native optimization diagnostic; correlated endpoint forks, all inherited/lost costs retained; not independent seeds or final superiority evidence',
                hypothesis='Independent actor/critic gradient clipping reduces critic-dominated actor suppression and improves actual task progress/completion without worse safety; only clipping grouping changes',
                primary_development_measure='Frozen four-training-road completion and progress, with collision/out-of-bounds and intervention rates reported; no selection based solely on KL',
                disk_free_bytes=shutil.disk_usage(ROOT).free)
            child['resource_limits']['wall_seconds'] = wall_seconds
            registered_jobs(child)
            write_json(batch/'protocol.json', child)
            write_json_atomic(batch/'parent_endpoint_manifest.json', manifest)
            write_json_atomic(batch/'prepared.json', {'status':'PREPARED_UNREGISTERED',
                'epoch':time.time(),'protocol_sha256':sha(batch/'protocol.json')})
        except BaseException as exc:
            _record_abort(batch,parent.name,'gradient_pair_staging',exc)
            raise
        try:
            claims = _claim_endpoints_atomic(runs,manifest,batch)
        except BaseException as exc:
            _record_abort(batch,parent.name,'gradient_pair_registration',exc)
            raise
        write_json_atomic(batch/'registered.json', {'status':'REGISTERED_READY','epoch':time.time(),**claims})
        try:
            write_json_atomic(pointer, {'batch':str(batch)})
        except BaseException as exc:
            write_json_atomic(batch/'pointer_publication_failure.json', {
                'status':'REGISTERED_CHILD_POINTER_NOT_PUBLISHED','epoch':time.time(),
                'claims':claims,'error_type':type(exc).__name__,'error':str(exc)})
            raise
        print(batch, flush=True)
        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--additional-controls', type=int, default=20000)
    parser.add_argument('--wall-seconds', type=int, default=21600)
    args = parser.parse_args()
    prepare(args.parent,args.additional_controls,args.wall_seconds)

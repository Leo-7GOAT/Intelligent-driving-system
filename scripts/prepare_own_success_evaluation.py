"""Stage an explicitly auxiliary fitted actor for zero-learning native evaluation."""
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
from scripts.audit_native_method_endpoint import read,sha,audit
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def validate(parent,artifact):
    require_finished_parent(parent)
    metadata=verified_checkpoint_metadata(artifact)
    method=metadata['source_method'];folder=parent/method
    if metadata['status']!='complete' or metadata['source_batch']!=parent.name:
        raise ValueError('Fit does not originate from the completed parent')
    if metadata['source_model_sha256']!=sha(folder/'model.zip'):
        raise ValueError('Original trained policy changed')
    if metadata['new_physics_steps']!=0 or metadata['ppo_updates_added']!=0:
        raise ValueError('Expected a separately counted offline actor fit')
    if not metadata['critic_unchanged'] or not metadata['critic_adam_unchanged']:
        raise ValueError('Unverified critic preservation')
    if sha(artifact/'fit_source.py')!=metadata['script_sha256'] or sha(artifact/'own_success_inputs.npz')!=metadata['dataset_sha256']:
        raise ValueError('Actor-fit provenance changed')
    if sha(folder/'normalization.pkl')!=metadata['normalization_sha256']:
        raise ValueError('Observation normalization changed')
    if read(artifact/'config.json')['env']!=read(folder/'config.json')['env']:
        raise ValueError('Task semantics changed')
    if audit(parent,method)!=read(folder/'endpoint_raw_audit.json') or metadata['source_raw_audit_sha256']!=sha(folder/'endpoint_raw_audit.json'):
        raise ValueError('Parent raw audit changed')
    return method,metadata


def prepare(parent,artifact,failed_prebridge=None,train_controls=0,freeze_statistics=False,curriculum_meters=None):
    if type(train_controls) is not int or not 0<=train_controls<=8192:raise ValueError('Bounded native training budget required')
    if freeze_statistics and not train_controls:raise ValueError('Statistics contrast requires actual PPO continuation')
    if curriculum_meters is not None and (curriculum_meters!=3. or not train_controls):
        raise ValueError('Only a bounded three-metre training curriculum is registered')
    parent=Path(parent).resolve();artifact=Path(artifact).resolve()
    method,metadata=validate(parent,artifact)
    old=read(parent/'protocol.json');summary=read(parent/method/'summary.json')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Frozen parent source changed')
    if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:raise ValueError('Calibration changed')
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    registration='evaluation_registration.json' if not train_controls else 'ppo_continuation_registration.json'
    if freeze_statistics:registration='frozen_statistics_continuation_registration.json'
    if curriculum_meters is not None:registration='three_metre_curriculum_registration.json'
    if failed_prebridge is not None:
        failed=Path(failed_prebridge).resolve()
        if read(artifact/registration)['batch']!=failed.name or list((failed/'infrastructure').glob('bridge_*')):
            raise ValueError('Only a registered pre-bridge launch failure can be retried')
        if not read(failed/'failure.json')['error'].startswith("FileNotFoundError('CARLA client interpreter not found:"):
            raise ValueError('Not the known missing runtime environment failure')
        if read(failed/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped':
            raise ValueError('Failed pre-bridge server is not cleaned up')
        registration='evaluation_prebridge_retry_registration.json'
    with ProcessLease(artifact/'evaluation_prepare.lock'):
        if (artifact/registration).exists():raise FileExistsError('Evaluation is already registered')
        suffix='_own_success_evaluation' if not train_controls else '_own_success_ppo_continuation'
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+suffix)
        batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(parent/'calibration',batch/'calibration')
        destination=batch/'resume'/method;destination.parent.mkdir()
        shutil.copytree(artifact,destination,ignore=shutil.ignore_patterns('*.lock','evaluation_registration.json'))
        if verified_checkpoint_metadata(destination)!=metadata:raise ValueError('Staged fitted actor changed')
        overrides=['native_road_learning.py','audit_native_method_endpoint.py','watch_existing_pair.py','prepare_own_success_evaluation.py']
        for name in overrides:shutil.copy2(ROOT/'scripts'/name,batch/'source/scripts'/name)
        write_json(batch/'parent_protocol.json',old);write_json(batch/'parent_training_summary.json',summary)
        write_json(batch/'parent_endpoint_raw_audit.json',read(parent/method/'endpoint_raw_audit.json'))
        protocol=deepcopy(old);now=time.time()
        for key in ('jobs','actor_head_experiment','gradient_clipping_experiment','rollout_length_experiment'):protocol.pop(key,None)
        retained=old.get('recovery',{}).get(method,{}).get('retained_optimizer_steps',0)+summary['optimizer_steps']
        scope=('Auxiliary own-success actor fit, frozen development-road evaluation; no new driving training or PPO updates; not independent conditions' if not train_controls else
            'Real PPO continuation after separately counted own-success auxiliary actor fit; same four development roads, not independent replication')
        protocol.update(scope=scope,
            evaluation_only=not bool(train_controls),methods=[method],controlled_budget_per_method=train_controls,method_budgets={method:train_controls},checkpoint_thresholds=[],
            created_epoch=now,deadline_epoch=now+900,parent_batch=parent.name,
            auxiliary_actor_fit={'artifact':artifact.name,'metadata_sha256':sha(artifact/'metadata.json'),
                'actor_optimizer_steps':metadata['actor_optimizer_steps'],'own_success_samples':metadata['samples'],
                'condition_counts':metadata['condition_counts'],'retained_ppo_optimizer_steps':retained},
            recovery={method:{'checkpoint':'resume/'+method,'parent_batch':parent.name,'parent_checkpoint':artifact.name,
                'lineage_controls':summary['lineage_control_steps'],'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound'],
                'retained_optimizer_steps':retained+metadata['actor_optimizer_steps']}},
            harness_overrides=['scripts/'+name for name in overrides],resource_limits={**old['resource_limits'],'wall_seconds':900})
        if freeze_statistics:
            protocol.update(freeze_observation_statistics=True,
                hypothesis='Observation normalization drift during continued PPO destroys the successful fitted actor; freeze existing statistics while retaining identical PPO updates, task distribution and nominal4096control budget',
                comparison='Same saved fitted actor, critic, Adam and RNG as the completed moving-statistics continuation; only VecNormalize.training changes to False')
        if curriculum_meters is not None:
            from scripts.prepare_native_flat_distribution_pair import short_route
            protocol.update(routes=[short_route(route,curriculum_meters) for route in old['routes']],
                scope='Three-metre development curriculum after own-success actor fit; changed task difficulty, not six-metre competence or formal performance',
                curriculum={'prefix_meters':curriculum_meters,'original_routes':old['routes']},
                hypothesis='The persistently failed road has no successful own experience; shorter goals may produce initial own stopping successes before returning to the original goals',
                comparison='Exploratory curriculum, not a matched architecture or normalization claim; all four roads retained, original six-metre failures remain')
        protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        if failed_prebridge is not None:
            protocol['prior_prebridge_failure']={'batch':failed.name,'failure_sha256':sha(failed/'failure.json'),'applied_controls':0}
            write_json(batch/'protocol.json',protocol)
            write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        write_json(artifact/registration,{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json')})
        write_json(ROOT/'runs/own_success_evaluation.json',{'batch':str(batch)})
    print(batch,flush=True);return batch


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);parser.add_argument('--artifact',required=True)
    parser.add_argument('--failed-prebridge')
    parser.add_argument('--train-controls',type=int,default=0)
    parser.add_argument('--freeze-statistics',action='store_true')
    parser.add_argument('--curriculum-meters',type=float)
    args=parser.parse_args();prepare(args.parent,args.artifact,args.failed_prebridge,args.train_controls,args.freeze_statistics,args.curriculum_meters)

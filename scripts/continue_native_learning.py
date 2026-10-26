"""Extend completed native policies without discarding measured interaction costs."""
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
from stable_baselines3 import PPO
from navstack.independent_clip_ppo import load_ppo
import numpy as np
import pickle
import torch
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok

ORCHESTRATION_FILES=('continue_native_learning.py','watch_existing_pair.py',
                     'start_carla.ps1','run_carla_owned_foreground.ps1','stop_owned_carla.ps1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_atomic(path, value):
    """Publish a small JSON record without exposing a partial document."""
    path=Path(path)
    temporary=path.with_name(f'.{path.name}.tmp-{os.getpid()}-{time.time_ns()}')
    write_json(temporary,value)
    temporary.replace(path)


def _qualified_name(value):
    cls=value if isinstance(value,type) else type(value)
    return f'{cls.__module__}.{cls.__qualname__}'


def _array_signature(value):
    array=np.asarray(value)
    contiguous=np.ascontiguousarray(array)
    return {'shape':list(array.shape),'dtype':str(array.dtype),
            'sha256':hashlib.sha256(contiguous.tobytes()).hexdigest()}


def _space_signature(space):
    """Describe policy I/O semantics without serializing mutable RNG state."""
    result={'class':_qualified_name(space)}
    if hasattr(space,'shape'):
        result['shape']=None if space.shape is None else list(space.shape)
    if hasattr(space,'dtype'):
        result['dtype']=str(space.dtype)
    if hasattr(space,'low') and hasattr(space,'high'):
        result['low']=_array_signature(space.low)
        result['high']=_array_signature(space.high)
    for name in ('n','start'):
        if hasattr(space,name):
            value=np.asarray(getattr(space,name))
            result[name]=int(value) if value.ndim==0 else _array_signature(value)
    if hasattr(space,'nvec'):
        result['nvec']=_array_signature(space.nvec)
    if hasattr(space,'spaces'):
        children=space.spaces
        result['spaces']=({str(k):_space_signature(v) for k,v in children.items()}
                          if isinstance(children,dict) else [_space_signature(v) for v in children])
    return result


def _semantic_value(value):
    """Turn saved constructor options into deterministic JSON-like values."""
    if value is None or isinstance(value,(str,bool,int,float)):
        return value
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return _array_signature(value)
    if isinstance(value,torch.dtype):return str(value)
    if isinstance(value,type):return _qualified_name(value)
    if isinstance(value,dict):
        return {str(k):_semantic_value(value[k]) for k in sorted(value,key=str)}
    if isinstance(value,(list,tuple)):
        return [_semantic_value(v) for v in value]
    if callable(value):return {'callable_class':_qualified_name(value)}
    return {'class':_qualified_name(value),'repr':repr(value)}


def _schedule_signature(schedule):
    if schedule is None:return None
    values=[]
    for progress in (1.0,0.5,0.0):
        values.append(float(schedule(progress)))
    return values


def _ppo_semantics(model):
    policy=model.policy
    policy_options={name:_semantic_value(getattr(policy,name)) for name in (
        'normalize_images','ortho_init','log_std_init','use_expln','squash_output',
        'share_features_extractor','full_std') if hasattr(policy,name)}
    return {
        'algorithm_class':_qualified_name(model),
        'policy_class':_qualified_name(policy),
        'observation_space':_space_signature(model.observation_space),
        'action_space':_space_signature(model.action_space),
        'n_steps':int(model.n_steps),'gamma':float(model.gamma),
        'gae_lambda':float(model.gae_lambda),'n_epochs':int(model.n_epochs),
        'batch_size':int(model.batch_size),
        'normalize_advantage':bool(model.normalize_advantage),
        'ent_coef':float(model.ent_coef),'vf_coef':float(model.vf_coef),
        'max_grad_norm':float(model.max_grad_norm),'use_sde':bool(model.use_sde),
        'sde_sample_freq':int(model.sde_sample_freq),
        'target_kl':None if model.target_kl is None else float(model.target_kl),
        'learning_rate':_schedule_signature(model.lr_schedule),
        'clip_range':_schedule_signature(model.clip_range),
        'clip_range_vf':_schedule_signature(model.clip_range_vf),
        'policy_kwargs':_semantic_value(model.policy_kwargs),
        'features_extractor_class':_qualified_name(policy.features_extractor),
        'mlp_extractor_class':_qualified_name(policy.mlp_extractor),
        'action_distribution_class':_qualified_name(policy.action_dist),
        'optimizer_class':_qualified_name(policy.optimizer),
        'optimizer_defaults':_semantic_value(policy.optimizer.defaults),
        'policy_options':policy_options,
    }


def _normalization_semantics(normalizer):
    return {
        'class':_qualified_name(normalizer),
        'training':bool(normalizer.training),
        'norm_obs':bool(normalizer.norm_obs),
        'norm_reward':bool(normalizer.norm_reward),
        'gamma':float(normalizer.gamma),
        'clip_obs':float(normalizer.clip_obs),
        'clip_reward':float(normalizer.clip_reward),
        'epsilon':float(normalizer.epsilon),
        'observation_space':_space_signature(normalizer.observation_space),
        'action_space':_space_signature(normalizer.action_space),
    }


def _equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b


def _rms_state(rms):
    if isinstance(rms,dict):return {key:_rms_state(value) for key,value in rms.items()}
    return {'mean':np.asarray(rms.mean),'var':np.asarray(rms.var),'count':rms.count}


def verified_endpoint(folder, target):
    """Resume the final evaluated weights, optimizer, statistics and saved RNG."""
    folder=Path(folder)
    summary=read(folder/'summary.json')
    evaluation=read(folder/'evaluation.json')
    latest=read(folder/'latest_checkpoint.json')
    checkpoint=folder/latest['folder']
    metadata=read(checkpoint/'metadata.json')
    lineage=summary['lineage_control_steps']
    if not summary['reload_exact'] or summary['stopped_reason'] is not None or not lineage<target:
        raise ValueError('Endpoint is not a complete extendable training result')
    if metadata['physics_steps']!=summary['physics_steps'] or metadata['lineage_control_steps']!=lineage:
        raise ValueError('Checkpoint is not the final training endpoint')
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
        if sha(folder/name)!=evaluation[key] or sha(checkpoint/name)!=metadata[key]:
            raise ValueError('Endpoint or checkpoint hash mismatch')
    if sha(checkpoint/'resume_state.pkl')!=metadata['resume_state_sha256']:
        raise ValueError('Checkpoint RNG hash mismatch')
    if read(checkpoint/'config.json')!=read(folder/'config.json'):
        raise ValueError('Checkpoint configuration mismatch')
    # ZIP timestamps may differ; compare the complete tensors and optimizer state.
    final=load_ppo(folder/'model.zip',device='cpu')
    saved=load_ppo(checkpoint/'model.zip',device='cpu')
    if not _equal(final.policy.state_dict(),saved.policy.state_dict()) or not _equal(final.policy.optimizer.state_dict(),saved.policy.optimizer.state_dict()):
        raise ValueError('Checkpoint weights or optimizer differ from evaluated endpoint')
    if final.num_timesteps!=saved.num_timesteps or final._n_updates!=saved._n_updates:
        raise ValueError('Checkpoint learning counters differ')
    if _ppo_semantics(final)!=_ppo_semantics(saved):
        raise ValueError('Checkpoint PPO or policy semantics differ from evaluated endpoint')
    with (folder/'normalization.pkl').open('rb') as f:final_norm=pickle.load(f)
    with (checkpoint/'normalization.pkl').open('rb') as f:saved_norm=pickle.load(f)
    if _normalization_semantics(final_norm)!=_normalization_semantics(saved_norm):
        raise ValueError('Checkpoint normalization semantics differ from evaluated endpoint')
    if not _equal(_rms_state(final_norm.obs_rms),_rms_state(saved_norm.obs_rms)):
        raise ValueError('Checkpoint normalization differs from evaluated endpoint')
    # SB3 deliberately removes the transient discounted-return accumulator when
    # pickling VecNormalize; ret_rms is the persisted return-normalization state.
    if not _equal(_rms_state(final_norm.ret_rms),_rms_state(saved_norm.ret_rms)):
        raise ValueError('Checkpoint return-normalization state differs from evaluated endpoint')
    return checkpoint,summary,metadata


def _continuation_is_terminal(batch):
    """A pointer may be replaced only after its lifecycle guard has ended."""
    batch=Path(batch)
    receipt=batch/'guard_final.json'
    if not batch.is_dir() or not receipt.is_file():return False
    try:guard=read(receipt)
    except (OSError,ValueError,json.JSONDecodeError):return False
    return isinstance(guard,dict) and type(guard.get('exit_code')) is int


def assert_continuation_pointer_available(runs):
    pointer=Path(runs)/'native_continuation.json'
    if not pointer.exists():return
    try:value=read(pointer);batch=Path(value['batch'])
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise RuntimeError('Existing native continuation pointer is unreadable; preserve it for inspection') from exc
    if not _continuation_is_terminal(batch):
        raise RuntimeError(f'Existing native continuation is active or non-terminal: {batch}')


def assert_parent_endpoints_unclaimed(runs,parent):
    """Honor endpoint-exact claims created by multi-batch continuation."""
    ledger=Path(runs)/'native_endpoint_claims.json'
    if not ledger.exists():return
    value=read(ledger)
    if value.get('schema_version')!=1 or not isinstance(value.get('claims'),dict):
        raise RuntimeError('Endpoint claim ledger is unreadable; preserve it for inspection')
    parent=Path(parent).resolve()
    for claim in value['claims'].values():
        location=claim.get('source_batch')
        if not isinstance(location,dict):
            raise RuntimeError('Endpoint claim ledger contains an invalid source location')
        if set(location)=={'root_relative'}:
            source=(ROOT/location['root_relative']).resolve()
        elif set(location)=={'absolute'}:
            source=Path(location['absolute']).resolve()
        else:
            raise RuntimeError('Endpoint claim ledger contains an invalid source location')
        if source==parent:
            raise FileExistsError(f'Parent endpoint already claimed by combined continuation: {claim.get("method")}')


def _allocate_batch(runs):
    stem=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_continuation'
    for index in range(1000):
        batch=Path(runs)/(stem if index==0 else f'{stem}_{index:03d}')
        try:batch.mkdir()
        except FileExistsError:continue
        return batch
    raise RuntimeError('Unable to allocate a unique continuation batch directory')


def _record_abort(batch,parent,stage,exc):
    write_json_atomic(Path(batch)/'aborted.json',{
        'status':'ABORTED_UNREGISTERED_DO_NOT_RUN','epoch':time.time(),
        'parent_batch':Path(parent).name,'child_batch':Path(batch).name,
        'stage':stage,'error_type':type(exc).__name__,'error':str(exc)})


def register_prepared_child(parent,batch,target):
    """Atomically claim a parent, but only for a fully staged child."""
    parent,batch=Path(parent),Path(batch)
    prepared=batch/'prepared.json';protocol=batch/'protocol.json'
    if not prepared.is_file() or not protocol.is_file():
        raise ValueError('Child is not completely prepared')
    receipt=read(prepared)
    if receipt.get('status')!='PREPARED_UNREGISTERED' or receipt.get('protocol_sha256')!=sha(protocol):
        raise ValueError('Child preparation receipt does not match protocol')
    registration={'child_batch':batch.name,'target':target,'epoch':time.time(),
                  'prepared_sha256':sha(prepared),
                  'note':'Run only this registered child; do not replay the parent'}
    with ProcessLease(parent/'guard.lock'):
        with (parent/'continuation_registration.json').open('x',encoding='utf-8') as stream:
            json.dump(registration,stream,indent=2)
            stream.flush();os.fsync(stream.fileno())
    return registration


def prepare(parent,target=100000,wall_seconds=7200):
    parent=Path(parent).resolve()
    if not 20000<target<=1000000 or not 900<=wall_seconds<=21600:
        raise ValueError('Native continuation budget outside supported bounds')
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    runs=ROOT/'runs'
    # Serializing pointer inspection through publication prevents two preparers
    # from both observing an empty slot and silently hiding one another.
    with ProcessLease(runs/'native_continuation.prepare.lock'):
        assert_continuation_pointer_available(runs)
        old=read(parent/'protocol.json');completed=read(parent/'summary.json')
        guard=read(parent/'guard_final.json');audit=read(parent/'analysis/verified_summary.json')
        if completed['status']!='complete' or guard['exit_code']!=0 or guard['reason'] is not None or (parent/'failure.json').exists():
            raise ValueError('Parent is not normally completed')
        if (parent/'continuation_registration.json').exists():
            raise FileExistsError('Parent already has a registered continuation')
        assert_parent_endpoints_unclaimed(runs,parent)
        if old.get('cleanup_owned_carla') and read(parent/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped':
            raise ValueError('Parent owned server shutdown is not verified')
        if [m['method'] for m in audit['methods']]!=old['methods']:
            raise ValueError('Parent audit does not cover every method')
        for relative,digest in old['source_hashes'].items():
            if sha(parent/'source'/relative)!=digest:raise ValueError('Parent frozen source changed')
        if sha(parent/'calibration/calibration.json')!=old['calibration_sha256']:
            raise ValueError('Parent calibration changed')
        torch.set_num_threads(1)
        endpoints={m:verified_endpoint(parent/m,target) for m in old['methods']}
        for row in audit['methods']:
            if row['control_frames_verified']!=endpoints[row['method']][1]['physics_steps'] or row['evaluation'] is None:
                raise ValueError('Parent raw audit counters differ from endpoint')
        batch=_allocate_batch(runs)
        try:
            (batch/'infrastructure').mkdir()
            shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(parent/'calibration',batch/'calibration')
            # Learning semantics remain the parent's frozen versions. Only the
            # continuation preparer and process lifecycle helpers are refreshed.
            for name in ORCHESTRATION_FILES:
                shutil.copy2(ROOT/'scripts'/name,batch/'source/scripts'/name)
            shutil.copy2(ROOT/'navstack/independent_clip_ppo.py',batch/'source/navstack/independent_clip_ppo.py')
            hashes=dict(old['source_hashes'])
            hashes['navstack/independent_clip_ppo.py']=sha(batch/'source/navstack/independent_clip_ppo.py')
            for name in ORCHESTRATION_FILES:
                relative='scripts/'+name
                hashes[relative]=sha(batch/'source'/relative)
            recovery={};budgets={};parent_records={}
            for method,(checkpoint,summary,metadata) in endpoints.items():
                destination=batch/'resume'/method;shutil.copytree(checkpoint,destination)
                for file in checkpoint.iterdir():
                    if file.is_file() and sha(file)!=sha(destination/file.name):raise ValueError('Copied checkpoint differs')
                prior=old.get('recovery',{}).get(method,{})
                optimizer_steps=prior.get('retained_optimizer_steps',0)+summary['optimizer_steps']
                recovery[method]={'checkpoint':'resume/'+method,'parent_batch':parent.name,'parent_checkpoint':checkpoint.name,
                    'lineage_controls':summary['lineage_control_steps'],
                    'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound'],
                    'retained_optimizer_steps':optimizer_steps,'model_sha256':metadata['model_sha256'],
                    'normalization_sha256':metadata['normalization_sha256'],
                    'unknown_unflushed_parent_tail':prior.get('unknown_unflushed_parent_tail',False)}
                budgets[method]=target-summary['lineage_control_steps']
                parent_records[method]={'summary':summary,'evaluation':read(parent/method/'evaluation.json'),
                    'checkpoint_hashes':{f.name:sha(f) for f in checkpoint.iterdir() if f.is_file()}}
            now=time.time()
            old.update(created_epoch=now,deadline_epoch=now+wall_seconds,source_hashes=hashes,
                controlled_budget_per_method=target,method_budgets=budgets,recovery=recovery,
                parent_batch=parent.name,parent_records=parent_records,
                parent_raw_audit_sha256=sha(parent/'analysis/verified_summary.json'),
                scope='Completed-policy native continuation; same seed and learning state, new measured episodes, cumulative controls and inherited costs; not independent replication',
                continuation_reason=f'Continue every registered parent method to cumulative {target} controls with unchanged learning settings; this is correlated continuation, not an independent seed',
                disk_free_bytes=shutil.disk_usage(ROOT).free)
            old['resource_limits']['wall_seconds']=wall_seconds
            write_json(batch/'protocol.json',old)
            write_json_atomic(batch/'prepared.json',{
                'status':'PREPARED_UNREGISTERED','epoch':time.time(),
                'parent_batch':parent.name,'child_batch':batch.name,
                'protocol_sha256':sha(batch/'protocol.json'),
                'source_files':len(hashes),'methods':old['methods']})
        except BaseException as exc:
            _record_abort(batch,parent,'staging',exc)
            raise
        try:
            register_prepared_child(parent,batch,target)
        except BaseException as exc:
            _record_abort(batch,parent,'parent_registration',exc)
            raise
        try:
            write_json_atomic(runs/'native_continuation.json',{'batch':str(batch)})
        except BaseException as exc:
            write_json_atomic(batch/'pointer_publication_failure.json',{
                'status':'REGISTERED_CHILD_POINTER_NOT_PUBLISHED','epoch':time.time(),
                'error_type':type(exc).__name__,'error':str(exc)})
            raise
        write_json_atomic(batch/'registered.json',{
            'status':'REGISTERED_READY','epoch':time.time(),
            'parent_registration_sha256':sha(parent/'continuation_registration.json'),
            'pointer_sha256':sha(runs/'native_continuation.json')})
        print(batch,flush=True)
        return batch


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True)
    p.add_argument('--target',type=int,default=100000);p.add_argument('--wall-seconds',type=int,default=7200)
    a=p.parse_args();prepare(a.parent,a.target,a.wall_seconds)

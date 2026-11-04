"""Replay every retained collection control and relabel with a current-state oracle.

The original observations, returns, split and executed actions stay unchanged.
Only supervised targets change; new teacher labels never execute in this run.
All replay interactions and teacher queries are additional charged costs.
"""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json,jsonable
from navstack.research import atomic_json,resources_ok


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(parent):
    parent=Path(parent).resolve()
    pointer=ROOT/'runs/current_state_relabel.json'
    if pointer.exists():raise RuntimeError('Existing relabel run must be inspected, not repeated')
    p=read(parent/'protocol.json');assert (parent/'completed.json').exists()
    competence=read(parent/'current_state_teacher_audit/results.json')
    assert len(competence['episodes'])==4 and all(r['metrics']['success'] for r in competence['episodes'])
    ok,reason=resources_ok(ROOT)
    if not ok or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_current_state_relabel')
    batch.mkdir();(batch/'input').mkdir()
    shutil.copy2(parent/'parent_data/demonstrations.npz',batch/'input/original.npz')
    shutil.copy2(parent/'parent_data/protocol.json',batch/'input/original_protocol.json')
    shutil.copy2(parent/'parent_data/completed.json',batch/'input/original_completed.json')
    for index in (1,2):
        shutil.copy2(parent/f'round_{index:02d}/recovery_examples.npz',batch/f'input/round{index}.npz')
        shutil.copy2(parent/f'round_{index:02d}/collection.json',batch/f'input/round{index}_collection.json')
    hashes={}
    for folder in ('navstack','scripts','tests'):
        for path in (ROOT/folder).rglob('*.py'):
            if '__pycache__' in path.parts:continue
            relative=path.relative_to(ROOT);target=batch/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target);hashes[relative.as_posix()]=sha(path)
    traces={f'round_{i:02d}/collection_{j:03d}.jsonl.gz':sha(parent/f'round_{i:02d}/collection_{j:03d}.jsonl.gz') for i in (1,2) for j in range(32)}
    now=time.time()
    protocol={'scope':__doc__,'created_epoch':now,'deadline_epoch':now+7200,'parent_directory':str(parent),
        'parent_batch':parent.name,'source_hashes':hashes,'input_hashes':{x.name:sha(x) for x in (batch/'input').iterdir()},
        'original_trace_hashes':traces,'env':p['env'],'collection_cases':p['collection_cases'],
        'groups':['original','round1','round2'],'expected_controls':10817+21721,
        'hypothesis':'Remove unavailable held-reference/controller history from action labels while preserving every actually visited state',
        'fixed':'Original observations, returns, case IDs, train/holdout masks, mixtures and old checkpoints are immutable; new labels are a separate dataset.',
        'teacher_competence_audit':{'source_sha256':sha(parent/'current_state_teacher_audit/results.json'),'controls':competence['actual_teacher_controls'],'cases':4},
        'resource_limits':{'wall_seconds':7200,'cpu_workers':1,'disk_min_GiB':12,'ram_min_GiB':4,'gpu_stop_temperature_C':85,'automatic_retries':0},
        'disk_free_bytes':shutil.disk_usage(ROOT).free,'worker_scopes':[]}
    write_json(batch/'protocol.json',protocol);write_json(pointer,{'batch':str(batch)})
    print(batch,flush=True)


def run(batch):
    import numpy as np
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.imitation import RecoveryTeacher
    from navstack.process_lease import ProcessLease
    batch=Path(batch).resolve();p=read(batch/'protocol.json');parent=Path(p['parent_directory'])
    for name,digest in p['source_hashes'].items():assert sha(ROOT/name)==digest
    for name,digest in p['input_hashes'].items():assert sha(batch/'input'/name)==digest
    with ProcessLease(batch/'worker.lock'):
        if (batch/'started.json').exists():raise RuntimeError('No automatic repeat; retain partial case labels and replay traces')
        write_json(batch/'started.json',{'pid':os.getpid(),'epoch':time.time()})
        rows=[];files={};total=0;last_heartbeat=0.
        try:
            for group in p['groups']:
                folder=batch/group;folder.mkdir()
                original=np.load(batch/'input'/(group+'.npz'))
                updated={key:original[key].copy() for key in original.files}
                cases=read(batch/'input/original_protocol.json')['cases'] if group=='original' else p['collection_cases']
                collection=read(batch/'input/original_completed.json') if group=='original' else read(batch/'input'/(group+'_collection.json'))
                for case_id,case in enumerate(cases):
                    selected=np.flatnonzero(original['case_id']==case_id)
                    cfg=EnvConfig.from_dict(p['env']);cfg.scenario=case['scenario'];cfg.split='train'
                    env=NavigationEnv(cfg);obs,_=env.reset(seed=case['seed'])
                    if group=='original':
                        executed=original['actions'][selected]
                    else:
                        relative=f"round_{int(group[-1]):02d}/collection_{case_id:03d}.jsonl.gz"
                        assert sha(parent/relative)==p['original_trace_hashes'][relative]
                        with gzip.open(parent/relative,'rt',encoding='utf-8') as source:
                            executed=np.asarray([json.loads(line)['executed_action'] for line in source],dtype=np.float32)
                    assert len(executed)==len(selected)
                    query_seconds=0.;maximum=0.;deltas=[];labels=[];query_failures=0
                    try:
                        with gzip.open(folder/f'case_{case_id:03d}.jsonl.gz','wt',encoding='utf-8') as stream:
                            for control,index in enumerate(selected):
                                if time.time()-last_heartbeat>5:
                                    ok,reason=resources_ok(batch)
                                    if not ok or shutil.disk_usage(batch).free<12*1024**3 or time.time()>p['deadline_epoch']-60:
                                        raise RuntimeError('Relabel guard: '+str(reason))
                                    last_heartbeat=time.time();stream.flush()
                                    atomic_json(batch/'training_heartbeat.json',{'pid':os.getpid(),'epoch':last_heartbeat,
                                        'group':group,'case_id':case_id,'control':control,'actual_replay_controls':total})
                                difference=float(np.max(np.abs(obs-original['observations'][index])))
                                maximum=max(maximum,difference)
                                assert difference<1e-5, 'Physically replayed observation differs from original'
                                label,query=RecoveryTeacher(env).query()
                                updated['actions'][index]=label;labels.append(label.copy())
                                deltas.append(label-original['actions'][index]);query_seconds+=query['query_seconds']
                                query_failures+=query['teacher_controller_status']=='failed'
                                obs,_,_,_,info=env.step(executed[control]);total+=1
                                stream.write(json.dumps({'control':control,'data_index':int(index),'old_label':original['actions'][index],
                                    'new_label':label,'executed_action':executed[control],'query':query,
                                    'observation_max_difference':difference,'info':info},default=jsonable)+'\n')
                        assert env.done
                        expected=collection['episodes'][case_id]['direct_replay' if group=='original' else 'mixture_metrics']
                        for key in ('success','termination_reason','physics_steps','actions'):
                            assert info['episode_metrics'][key]==expected[key]
                        np.save(folder/f'case_{case_id:03d}_labels.npy',np.asarray(labels,dtype=np.float32))
                        row={'group':group,'case_id':case_id,**case,'controls':len(selected),'teacher_queries':len(selected),
                            'query_seconds':query_seconds,'teacher_control_failures':int(query_failures),
                            'max_observation_difference':maximum,'actual_replay_metrics':info['episode_metrics'],
                            'mean_absolute_label_difference':np.abs(deltas).mean(0),'max_absolute_label_difference':np.abs(deltas).max(0)}
                        rows.append(row);write_json(batch/'progress.json',rows)
                        print(json.dumps({k:row[k] for k in ('group','case_id','scenario','controls','max_observation_difference','mean_absolute_label_difference')},default=jsonable),flush=True)
                    finally:env.close()
                for key in original.files:
                    if key!='actions':np.testing.assert_array_equal(original[key],updated[key])
                output=batch/(group+'.npz');np.savez_compressed(output,**updated)
                files[group]={'file':output.name,'sha256':sha(output),'original_sha256':p['input_hashes'][group+'.npz'],
                    'controls':len(updated['actions']),'all_non_action_arrays_identical':True}
                write_json(batch/'completed_groups.json',files)
            assert total==p['expected_controls']
            write_json(batch/'completed.json',{'scope':p['scope'],'files':files,'episodes':rows,
                'actual_replay_controls':total,'teacher_queries':total,'ppo_updates':0,'supervised_optimizer_steps':0,
                'no_new_teacher_action_executed':True})
        except BaseException as exc:
            write_json(batch/'failure.json',{'epoch':time.time(),'error':repr(exc),'actual_replay_controls':total,'completed_groups':files});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent');p.add_argument('--batch');a=p.parse_args()
    prepare(a.parent) if a.parent else run(a.batch)

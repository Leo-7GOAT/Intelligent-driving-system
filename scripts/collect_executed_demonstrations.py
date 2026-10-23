"""Collect a declared training-only B3/direct replay dataset, including failures."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import EnvConfig
from navstack.env import NavigationEnv,rule_action
from navstack.experiments import write_json,jsonable
from navstack.imitation import replay_demonstration
from navstack.research import resources_ok
from navstack.process_lease import ProcessLease
from scripts.initialize_candidate_development import reference_protocol


def prepare():
    if (ROOT/'runs/demonstration_diagnostic.json').exists():raise RuntimeError('Inspect existing dataset; no duplicate collection')
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    reference=reference_protocol();cfg=EnvConfig.from_dict(reference['jobs'][0]['env'])
    cfg.method='B3';cfg.candidate_observation=True;cfg.reward.progress_weight=.25
    cases=[{'scenario':scenario,'seed':seed,'data_split':'train' if seed<=8106 else 'holdout'}
        for seed in range(8101,8109) for scenario in ('empty','static','narrow','dynamic')]
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_executed_demonstrations')
    batch.mkdir();hashes={}
    for folder in ('navstack','scripts'):
        for path in (ROOT/folder).rglob('*.py'):
            if '__pycache__' in path.parts:continue
            relative=path.relative_to(ROOT);target=batch/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target);hashes[relative.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    now=time.time()
    protocol={'scope':'Explicit additional-supervision development diagnostic; not PPO training, independent confirmation or hierarchy superiority',
        'created_epoch':now,'deadline_epoch':now+1200,'source_hashes':hashes,'env':cfg.to_dict(),'cases':cases,
        'development_cases':reference['development_cases'],'model_seed':11,
        'hypothesis':'Determine whether direct-control networks can represent productive obstacle/parking actions before attributing failures solely to on-policy exploration',
        'fairness':'Both high-level chosen-action observations and direct per-control observations are recorded. Any formal hierarchy comparison must share equivalent demonstration access and charge teacher/replay/supervised/on-policy costs separately.',
        'frequency':'Teacher decisions0.5s/MPC0.05s; direct replay and intended direct learner hold1 at0.05s. This differs from the previous hold10 study and is not a single-factor supervision contrast with it.',
        'resource_limits':{'cpu_workers':1,'wall_seconds':1200,'disk_min_GiB':12,'ram_min_GiB':4,'automatic_retries':0},
        'disk_free_bytes':shutil.disk_usage(ROOT).free}
    write_json(batch/'protocol.json',protocol);write_json(ROOT/'runs/demonstration_diagnostic.json',{'batch':str(batch)})
    print(batch,flush=True)


def run(batch):
    batch=Path(batch).resolve();protocol=json.loads((batch/'protocol.json').read_text(encoding='utf-8'))
    for relative,digest in protocol['source_hashes'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    with ProcessLease(batch/'worker.lock'):
        if (batch/'started.json').exists():raise RuntimeError('No automatic repeated collection')
        write_json(batch/'started.json',{'epoch':time.time()})
        flat_obs=[];flat_actions=[];returns=[];flat_cases=[];macro_obs=[];macro_actions=[];macro_cases=[];rows=[]
        try:
            for case_id,case in enumerate(protocol['cases']):
                healthy,reason=resources_ok(batch)
                if not healthy or time.time()>protocol['deadline_epoch'] or shutil.disk_usage(batch).free<12*1024**3:
                    raise RuntimeError('Collection resource/deadline guard: '+str(reason))
                cfg=EnvConfig.from_dict(protocol['env']);cfg.scenario=case['scenario'];cfg.split='train'
                env=NavigationEnv(cfg);obs,_=env.reset(seed=case['seed']);ticks=[]
                try:
                    with gzip.open(batch/f'teacher_{case_id:03d}.jsonl.gz','wt',encoding='utf-8') as stream:
                        while not env.done:
                            if time.time()>protocol['deadline_epoch']:raise RuntimeError('Collection deadline')
                            action=rule_action(env);macro_obs.append(obs.copy());macro_actions.append(action);macro_cases.append(case_id)
                            obs,_,_,_,info=env.step(action);ticks.extend(info['trace'])
                            stream.write(json.dumps({'action':action,'info':info},default=jsonable)+'\n')
                    teacher_metrics=info['episode_metrics']
                finally:env.close()
                direct=replay_demonstration(cfg,case['seed'],ticks)
                with gzip.open(batch/f'direct_replay_{case_id:03d}.jsonl.gz','wt',encoding='utf-8') as stream:
                    for tick in direct['trace']:stream.write(json.dumps(tick,default=jsonable)+'\n')
                flat_obs.extend(direct['observations']);flat_actions.extend(direct['actions']);returns.extend(direct['returns'])
                flat_cases.extend([case_id]*len(direct['trace']))
                row={'case_id':case_id,**case,'teacher':teacher_metrics,'direct_replay':direct['metrics'],
                    'max_state_difference':direct['max_state_difference'],'initial':direct['initial']}
                rows.append(row);write_json(batch/'progress.json',rows)
                print(json.dumps({'case_id':case_id,'scenario':case['scenario'],'teacher_success':teacher_metrics['success'],
                    'direct_success':direct['metrics']['success'],'controls':len(ticks)}),flush=True)
            data=batch/'demonstrations.npz'
            np.savez_compressed(data,observations=np.array(flat_obs,dtype=np.float32),actions=np.array(flat_actions,dtype=np.float32),
                returns=np.array(returns,dtype=np.float32),case_id=np.array(flat_cases,dtype=np.int32),
                macro_observations=np.array(macro_obs,dtype=np.float32),macro_actions=np.array(macro_actions,dtype=np.int64),macro_case_id=np.array(macro_cases,dtype=np.int32))
            write_json(batch/'completed.json',{'scope':protocol['scope'],'episodes':rows,'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
                'teacher_controls':sum(r['teacher']['physics_steps'] for r in rows),'direct_replay_controls':len(flat_obs),
                'teacher_decisions':len(macro_obs),'discarded_failure_episodes':0,'finished_epoch':time.time()})
        except BaseException as exc:
            write_json(batch/'failure.json',{'error':repr(exc),'completed_cases':len(rows),'epoch':time.time()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--batch')
    a=p.parse_args();prepare() if a.prepare else run(a.batch)

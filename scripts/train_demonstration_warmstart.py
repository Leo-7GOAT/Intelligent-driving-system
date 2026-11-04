"""Explicit B1 imitation initialization followed by real, separately counted PPO."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from navstack.research import resources_ok


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def prepare(dataset):
    from navstack.config import EnvConfig
    dataset=Path(dataset).resolve();data_protocol=read(dataset/'protocol.json');complete=read(dataset/'completed.json')
    if (ROOT/'runs/demonstration_warmstart.json').exists():raise RuntimeError('Inspect existing experiment, no duplicate training')
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    assert hashlib.sha256((dataset/'demonstrations.npz').read_bytes()).hexdigest()==complete['data_sha256']
    cfg=EnvConfig.from_dict(data_protocol['env']);cfg.method='B1';cfg.flat_hold=1;cfg.scenario='mixed'
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_demonstration_warmstart')
    batch.mkdir();shutil.copytree(dataset/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('navstack/imitation.py','scripts/train_demonstration_warmstart.py'):
        shutil.copy2(ROOT/name,batch/'source'/name)
    (batch/'data').mkdir()
    for name in ('demonstrations.npz','protocol.json','completed.json'):shutil.copy2(dataset/name,batch/'data'/name)
    cases=list(data_protocol['development_cases'])
    cases += [{'scenario':'empty','seed':seed,'label':f'retention_{distance}m','options':{'diagnostic_route_length':float(distance)}}
        for distance in (6,12,18,23) for seed in (7301,7302)]
    now=time.time();protocol={'scope':'One-seed explicit demonstration-assisted B1 development; BC-only and subsequent PPO are separate, not a pure-RL hierarchy comparison',
        'created_epoch':now,'deadline_epoch':now+3600,'env':cfg.to_dict(),'seed':11,'cases':cases,'worker_scopes':['B1'],
        'supervised_epochs':200,'milestones':[0,25000,100000],'dataset_batch':dataset.name,'data_sha256':complete['data_sha256'],
        'teacher_controls':complete['teacher_controls'],'direct_replay_controls':complete['direct_replay_controls'],
        'supervision_scope':'Fit 24 declared training episodes; 8 heldout demonstrations only report loss. No validation/test actions, labels or outcomes used for fitting.',
        'ppo_options':{'target_kl':.03,'use_sde':True,'sde_sample_freq':1,'log_std_init':-2,'entropy':.001,'learning_rate':.0001},
        'fairness':data_protocol['fairness'],'frequency':data_protocol['frequency'],
        'selection':'Fixed 200 supervised epochs and every PPO milestone; no best development checkpoint replaces a later failure',
        'source_hashes':{p.relative_to(batch/'source').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (batch/'source').rglob('*.py')},
        'resource_limits':{'wall_seconds':3600,'cpu_workers':1,'disk_min_GiB':12,'ram_min_GiB':4,'automatic_retries':0},'disk_free_bytes':shutil.disk_usage(ROOT).free}
    write_json(batch/'protocol.json',protocol);write_json(ROOT/'runs/demonstration_warmstart.json',{'batch':str(batch)})
    print(batch,flush=True)


def run(batch):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize,DummyVecEnv
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.experiments import actor_vector
    from navstack.imitation import fit_demonstrations
    from navstack.process_lease import ProcessLease
    from navstack.research import cumulative_train,atomic_json
    from scripts.hierarchy_development import evaluate
    batch=Path(batch).resolve();protocol=read(batch/'protocol.json');torch.set_num_threads(1)
    for relative,digest in protocol['source_hashes'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((batch/'data/demonstrations.npz').read_bytes()).hexdigest()==protocol['data_sha256']
    with ProcessLease(batch/'worker.lock'):
        if (batch/'started.json').exists():raise RuntimeError('No automatic repeated supervised/PPO experiment')
        write_json(batch/'started.json',{'epoch':time.time()});results=[]
        folder=batch/'B1';cfg=EnvConfig.from_dict(protocol['env'])
        try:
            initial=cumulative_train(folder,cfg,protocol['seed'],0,protocol['deadline_epoch']-120,ppo_options=protocol['ppo_options'])
            checkpoint=Path(initial['checkpoint'])
            raw=DummyVecEnv([lambda:NavigationEnv(cfg)]);norm=VecNormalize.load(checkpoint/'normalization.pkl',raw)
            model=PPO.load(checkpoint/'model.zip',env=norm,device='cpu');before=actor_vector(model).clone()
            data=np.load(batch/'data/demonstrations.npz');cases=read(batch/'data/protocol.json')['cases']
            mask=np.array([cases[int(i)]['data_split']=='train' for i in data['case_id']])
            def record(row):
                healthy,reason=resources_ok(batch)
                if not healthy or time.time()>protocol['deadline_epoch']-120:raise RuntimeError('Supervision guard: '+str(reason))
                atomic_json(batch/'training_heartbeat.json',{'pid':os.getpid(),'epoch':time.time(),**row})
                print(json.dumps(row),flush=True)
            history=fit_demonstrations(model,norm,data['observations'],data['actions'],data['returns'],mask,
                epochs=protocol['supervised_epochs'],seed=protocol['seed'],callback=record)
            assert not model.policy.optimizer.state and model._n_updates==0
            supervised=folder/'supervised_initial';supervised.mkdir()
            model.save(supervised/'model.zip');norm.save(supervised/'normalization.pkl')
            shutil.copy2(checkpoint/'config.json',supervised/'config.json')
            with (checkpoint/'resume.pkl').open('rb') as stream:snapshot=pickle.load(stream)
            snapshot['initial_actor']=actor_vector(model).clone()
            with (supervised/'resume.pkl').open('wb') as stream:pickle.dump(snapshot,stream)
            initial.update(checkpoint=str(supervised),reason='supervised_initialization_before_PPO',actor_l2_change=0.)
            write_json(supervised/'training_summary.json',initial);atomic_json(folder/'latest.json',initial)
            write_json(batch/'supervision.json',{'history':history,'training_examples':int(mask.sum()),'holdout_examples':int((~mask).sum()),
                'normalization_count':float(norm.obs_rms.count),'actor_l2_change_supervised':float(torch.linalg.vector_norm(actor_vector(model)-before)),
                'ppo_updates':0,'teacher_controls':protocol['teacher_controls'],'direct_replay_controls':protocol['direct_replay_controls']})
            norm.close()
            for target in protocol['milestones']:
                saved=cumulative_train(folder,cfg,protocol['seed'],target,protocol['deadline_epoch']-120,ppo_options=protocol['ppo_options'])
                if saved['reason']!='milestone_reached':raise RuntimeError('PPO milestone incomplete: '+saved['reason'])
                evaluate(folder,saved['checkpoint'],EnvConfig.from_dict(cfg.to_dict()),protocol['cases'],target)
                outcome=read(folder/f'validation_{target}.json')
                row={'target_ppo_controls':target,'actual_ppo_controls':saved['physics_steps'],'ppo_optimizer_steps':saved['optimizer_steps'],
                    'actor_l2_change_after_supervision':saved['actor_l2_change'],
                    'full_task_success':sum(e['success'] for e in outcome['episodes'] if 'label' not in e),
                    'retention':{str(d):sum(e['success'] for e in outcome['episodes'] if e.get('label')==f'retention_{d}m') for d in (6,12,18,23)}}
                results.append(row);write_json(batch/'progress.json',results);print(json.dumps(row),flush=True)
            write_json(batch/'completed.json',{'scope':protocol['scope'],'results':results})
        except BaseException as exc:
            write_json(batch/'failure.json',{'error':repr(exc),'epoch':time.time(),'results':results});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--batch');a=p.parse_args()
    prepare(a.dataset) if a.dataset else run(a.batch)

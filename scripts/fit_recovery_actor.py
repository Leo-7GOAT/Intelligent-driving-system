"""Single-factor actor-only supervised fit against completed aggregation round 2."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.experiments import write_json
from navstack.research import atomic_json,resources_ok


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(parent, relabel=None, control=None):
    parent=Path(parent).resolve()
    if bool(relabel)!=bool(control):raise ValueError('Relabeled comparison requires its explicit actor-only control')
    study='current_state_actor_fit' if relabel else 'recovery_actor_only'
    pointer=ROOT/'runs'/(study+'.json')
    if pointer.exists():raise RuntimeError('Inspect existing registered contrast; no duplicate')
    p=read(parent/'protocol.json');original=read(parent/'round_02/supervision.json')
    assert (parent/'round_02/validation_0.json').exists()
    ok,reason=resources_ok(ROOT)
    if not ok or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    relabeled=None
    if relabel:
        import ast
        import numpy as np
        from scripts.verify_relabel_teacher import verify
        relabel,control=Path(relabel).resolve(),Path(control).resolve()
        verify(relabel)
        relabeled=read(relabel/'completed.json');control_protocol=read(control/'protocol.json')
        assert (control/'completed.json').exists() and control_protocol['value_weight']==0.
        assert control_protocol['parent_batch']==parent.name
        assert control_protocol['epochs']==200 and control_protocol['seed']==13
        assert control_protocol['env']==p['env'] and control_protocol['cases']==p['cases']
        def fitting_definition(path):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='fit_demonstrations'))
        assert fitting_definition(ROOT/'navstack/imitation.py')==fitting_definition(control/'source/navstack/imitation.py'), 'Fitting implementation differs from control'
        assert read(relabel/'protocol.json')['parent_batch']==parent.name
        for name,digest in control_protocol['initial_hashes'].items():
            assert sha(parent/'round_01/supervised_checkpoint'/name)==digest
        for group,old_path in [('original',parent/'parent_data/demonstrations.npz'),
                              ('round1',parent/'round_01/recovery_examples.npz'),('round2',parent/'round_02/recovery_examples.npz')]:
            assert sha(old_path)==control_protocol['data_hashes'][group+'.npz']==sha(control/'data'/(group+'.npz'))
            assert sha(relabel/(group+'.npz'))==relabeled['files'][group]['sha256']
            old,new=np.load(old_path),np.load(relabel/(group+'.npz'))
            assert set(old.files)==set(new.files)
            for key in old.files:
                if key!='actions':np.testing.assert_array_equal(old[key],new[key])
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+study)
    batch.mkdir();shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('navstack/imitation.py','scripts/fit_recovery_actor.py','scripts/verify_relabel_teacher.py'):
        shutil.copy2(ROOT/name,batch/'source'/name)
    shutil.copytree(parent/'round_01/supervised_checkpoint',batch/'initial')
    (batch/'data').mkdir()
    for relative,name in [('parent_data/demonstrations.npz','original.npz'),('parent_data/protocol.json','original_protocol.json'),
        ('round_01/recovery_examples.npz','round1.npz'),('round_02/recovery_examples.npz','round2.npz'),
        ('round_02/supervision.json','control_supervision.json'),('round_02/validation_0.json','control_validation.json')]:
        source=parent/relative
        if relabel and name in ('original.npz','round1.npz','round2.npz'):source=relabel/name
        if relabel and name=='control_supervision.json':source=control/'supervision.json'
        if relabel and name=='control_validation.json':source=control/'evaluation/validation_0.json'
        shutil.copy2(source,batch/'data'/name)
    now=time.time()
    protocol={'scope':'One-seed offline gradient-coupling contrast; extra supervision, no new PPO and no architecture advantage claim',
        'created_epoch':now,'deadline_epoch':now+1800,'parent_batch':parent.name,'env':p['env'],'cases':p['cases'],
        'epochs':200,'seed':13,'value_weight':0.,'control_value_weight':.5,
        'hypothesis':'The global supervised gradient clip couples high-variance mixture-return regression to actor fitting; omit only value gradients',
        'fixed':'Same round-one model and normalization, identical combined train/holdout data, permutation seed13, 200epochs, Adam1e-3 and global clip. Norm frozen. All18cases; no best checkpoint selection.',
        'control_supervised_optimizer_steps':original['history'][-1]['supervised_optimizer_steps'],
        'resource_limits':{'wall_seconds':1800,'cpu_workers':1,'disk_min_GiB':12,'ram_min_GiB':4,'automatic_retries':0},
        'worker_scopes':['evaluation'],'disk_free_bytes':shutil.disk_usage(ROOT).free,
        'data_hashes':{x.name:sha(x) for x in (batch/'data').iterdir()},
        'initial_hashes':{x.name:sha(x) for x in (batch/'initial').iterdir()},
        'source_hashes':{x.relative_to(batch/'source').as_posix():sha(x) for x in (batch/'source').rglob('*.py')}}
    if relabel:
        protocol.update(scope='One-seed current-state action-label contrast against actor-only fitting; not pure RL or final confirmation',
            control_value_weight=0.,control_batch=control.name,relabel_batch=relabel.name,
            hypothesis='Remove held-reference teacher history from supervised targets while fixing every visited observation and optimizer condition',
            fixed='Same round-one initialization, all non-action arrays exactly unchanged, same train/holdout split, fixed stats, seed13, 200epochs and actor-only Adam; only direct action labels differ.',
            additional_label_costs={'actual_replay_controls':relabeled['actual_replay_controls'],'teacher_queries':relabeled['teacher_queries']},
            inherited_macro_labels='Historical unchanged macro arrays are not new current-state supervision for hierarchical policies')
    write_json(batch/'protocol.json',protocol);write_json(pointer,{'batch':str(batch)})
    print(batch,flush=True)


def run(batch):
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.experiments import actor_vector,load_run
    from navstack.imitation import fit_demonstrations
    from navstack.process_lease import ProcessLease
    from scripts.hierarchy_development import evaluate
    batch=Path(batch).resolve();p=read(batch/'protocol.json');torch.set_num_threads(1)
    for name,digest in p['source_hashes'].items():assert sha(ROOT/name)==digest
    for folder,key in [('data','data_hashes'),('initial','initial_hashes')]:
        for name,digest in p[key].items():assert sha(batch/folder/name)==digest
    with ProcessLease(batch/'worker.lock'):
        if (batch/'started.json').exists():raise RuntimeError('No automatic repeated fitting')
        write_json(batch/'started.json',{'epoch':time.time(),'pid':os.getpid()})
        model,norm=load_run(batch/'initial')
        before=actor_vector(model).clone()
        critic={name:t.detach().clone() for name,t in model.policy.named_parameters() if 'value_net' in name}
        mean,var,count=norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count
        data=[np.load(batch/'data'/name) for name in ('original.npz','round1.npz','round2.npz')]
        cases=read(batch/'data/original_protocol.json')['cases']
        masks=[np.array([cases[int(i)]['data_split']=='train' for i in data[0]['case_id']])]+[d['train_mask'] for d in data[1:]]
        mask=np.concatenate(masks)
        arrays={k:np.concatenate([d[k] for d in data]) for k in ('observations','actions','returns')}
        try:
            def record(row):
                ok,reason=resources_ok(batch)
                if not ok or time.time()>p['deadline_epoch']-120:raise RuntimeError(str(reason))
                atomic_json(batch/'training_heartbeat.json',{'epoch':time.time(),'pid':os.getpid(),**row})
                print(json.dumps(row),flush=True)
            history=fit_demonstrations(model,norm,arrays['observations'],arrays['actions'],arrays['returns'],mask,
                epochs=p['epochs'],seed=p['seed'],callback=record,update_normalization=False,value_weight=p['value_weight'])
            assert history[-1]['supervised_optimizer_steps']==p['control_supervised_optimizer_steps']
            assert all(torch.equal(critic[name],t) for name,t in model.policy.named_parameters() if name in critic)
            assert model._n_updates==0 and not model.policy.optimizer.state
            np.testing.assert_array_equal(mean,norm.obs_rms.mean);np.testing.assert_array_equal(var,norm.obs_rms.var)
            assert count==norm.obs_rms.count
            checkpoint=batch/'supervised_checkpoint';checkpoint.mkdir()
            model.save(checkpoint/'model.zip');norm.save(checkpoint/'normalization.pkl')
            shutil.copy2(batch/'initial/config.json',checkpoint/'config.json')
            write_json(batch/'supervision.json',{'history':history,'actor_l2_change':float(torch.linalg.vector_norm(actor_vector(model)-before)),
                'critic_unchanged':True,'normalization_frozen':True,'ppo_updates':0,'training_examples':int(mask.sum()),
                'holdout_examples':int((~mask).sum()),'model_sha256':sha(checkpoint/'model.zip')})
            folder=batch/'evaluation';folder.mkdir()
            evaluate(folder,checkpoint,EnvConfig.from_dict(p['env']),p['cases'],0)
            value=read(folder/'validation_0.json')
            result={'scope':p['scope'],'full_task_success':sum(e['success'] for e in value['episodes'] if 'label' not in e),
                'retention':{str(d):sum(e['success'] for e in value['episodes'] if e.get('label')==f'retention_{d}m') for d in (6,12,18,23)},
                'supervised_optimizer_steps':history[-1]['supervised_optimizer_steps'],'additional_collection_controls':0,
                'evaluation_controls':sum(e['physics_steps'] for e in value['episodes']),'ppo_updates':0}
            write_json(batch/'completed.json',result);print(json.dumps(result),flush=True)
        except BaseException as exc:
            write_json(batch/'failure.json',{'epoch':time.time(),'error':repr(exc)});raise
        finally:norm.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent');p.add_argument('--batch')
    p.add_argument('--relabel');p.add_argument('--control');a=p.parse_args()
    prepare(a.parent,a.relabel,a.control) if a.parent else run(a.batch)

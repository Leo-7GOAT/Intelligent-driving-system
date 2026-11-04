"""Bounded, explicitly supervised recovery-state aggregation, then actual PPO.

Mixture success is never reported as learned-policy success. Evaluation uses
only model.predict and the physical direct-control environment, no teacher.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.experiments import write_json, jsonable
from navstack.research import atomic_json, resources_ok


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(parent):
    from navstack.config import EnvConfig
    parent = Path(parent).resolve()
    pointer = ROOT/'runs/recovery_demonstrations.json'
    if pointer.exists():
        raise RuntimeError('Inspect existing registered experiment; no duplicate preparation')
    previous = read(parent/'protocol.json')
    assert (parent/'completed.json').exists()
    checkpoint = parent/'B1/supervised_initial'
    cfg = EnvConfig.from_dict(previous['env'])
    assert cfg.method == 'B1' and cfg.flat_hold == 1 and cfg.planner.max_seconds is None
    assert not cfg.route_stages and not cfg.curriculum and not cfg.empty_pretraining_steps
    ok, reason = resources_ok(ROOT)
    if not ok or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError(str(reason))
    batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_recovery_demonstrations')
    batch.mkdir()
    shutil.copytree(parent/'data', batch/'parent_data')
    (batch/'initial').mkdir()
    for name in ('model.zip','normalization.pkl','config.json'):
        shutil.copy2(checkpoint/name, batch/'initial'/name)
    shutil.copy2(parent/'supervision.json', batch/'parent_supervision.json')
    hashes = {}
    for folder in ('navstack','scripts','tests'):
        for path in (ROOT/folder).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT)
            target = batch/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[relative.as_posix()] = sha(path)
    cases = [{'scenario':scenario, 'seed':seed, 'data_split':'train' if seed <= 8206 else 'holdout'}
             for seed in range(8201,8209) for scenario in ('empty','static','narrow','dynamic')]
    now = time.time()
    protocol = {'scope':'One-seed recovery-state supervised aggregation plus separately counted PPO; not pure RL or final confirmation',
        'created_epoch':now, 'deadline_epoch':now+7200, 'source_hashes':hashes,
        'initial_hashes':{p.name:sha(p) for p in (batch/'initial').iterdir()},
        'parent_batch':parent.name, 'data_sha256':sha(batch/'parent_data/demonstrations.npz'),
        'env':cfg.to_dict(), 'seed':11, 'collection_cases':cases, 'cases':previous['cases'],
        'rounds':[{'index':1,'teacher_probability':.5},{'index':2,'teacher_probability':.25}],
        'supervised_epochs_per_round':200,
        'normalization':'Frozen original training-demonstration statistics during both aggregation fits; heldout cases never update stats or gradients. PPO later resumes ordinary on-policy stats.',
        'value_targets':'Discounted actual mixture rewards for recovery examples, original teacher returns for original examples; extra supervision, not on-policy PPO targets',
        'selection':'Use both fixed 200-epoch round endpoints, then final round for PPO; never select best development result',
        'ppo_milestones':[25000,100000], 'ppo_options':previous['ppo_options'],
        'teacher_scope':'Read actual learner states through private planner/MPC workspace. Execute teacher or deterministic learner by fixed per-10-control Bernoulli mixture, record source every control. No teacher in evaluation or PPO.',
        'fairness':previous['fairness'], 'frequency':previous['frequency'],
        'inherited_costs':{'original_teacher_controls':previous['teacher_controls'],
            'original_direct_replay_controls':previous['direct_replay_controls'],
            'original_supervised_optimizer_steps':read(parent/'supervision.json')['history'][-1]['supervised_optimizer_steps'],
            'discarded_sibling_ppo_controls':read(parent/'completed.json')['results'][-1]['actual_ppo_controls']},
        'worker_scopes':['round_01','round_02','B1'],
        'resource_limits':{'wall_seconds':7200,'cpu_workers':1,'disk_min_GiB':12,'ram_min_GiB':4,'gpu_stop_temperature_C':85,'automatic_retries':0},
        'disk_free_bytes':shutil.disk_usage(ROOT).free}
    write_json(batch/'protocol.json', protocol)
    write_json(pointer, {'batch':str(batch)})
    print(batch, flush=True)


def run(batch):
    import copy
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.experiments import load_run, actor_vector
    from navstack.imitation import collect_recovery_episode, fit_demonstrations
    from navstack.process_lease import ProcessLease
    from navstack.research import cumulative_train
    from scripts.hierarchy_development import evaluate
    batch = Path(batch).resolve()
    protocol = read(batch/'protocol.json')
    for relative, digest in protocol['source_hashes'].items():
        assert sha(ROOT/relative) == digest
    for name, digest in protocol['initial_hashes'].items():
        assert sha(batch/'initial'/name) == digest
    assert sha(batch/'parent_data/demonstrations.npz') == protocol['data_sha256']
    torch.set_num_threads(1)
    with ProcessLease(batch/'worker.lock'):
        if (batch/'started.json').exists():
            raise RuntimeError('No automatic repetition; preserve completed rounds and partial evidence')
        write_json(batch/'started.json', {'epoch':time.time(),'pid':os.getpid()})
        results, collections, supervision = [], [], []
        model, norm = load_run(batch/'initial')
        stats = copy.deepcopy(norm.obs_rms)
        cfg = EnvConfig.from_dict(protocol['env'])
        data = np.load(batch/'parent_data/demonstrations.npz')
        original_cases = read(batch/'parent_data/protocol.json')['cases']
        arrays = {k:[data[k]] for k in ('observations','actions','returns')}
        initial_mask = np.asarray([original_cases[int(i)]['data_split']=='train' for i in data['case_id']])
        if 'train_mask' in data:
            np.testing.assert_array_equal(initial_mask,data['train_mask'])
        masks = [initial_mask]
        last_heartbeat = 0.

        def heartbeat(stage, **extra):
            nonlocal last_heartbeat
            healthy, reason = resources_ok(batch)
            if not healthy or shutil.disk_usage(batch).free < 12*1024**3 or time.time() > protocol['deadline_epoch']-120:
                raise RuntimeError('Recovery study resource/deadline guard: '+str(reason))
            last_heartbeat = time.time()
            atomic_json(batch/'training_heartbeat.json', {'pid':os.getpid(),'epoch':last_heartbeat,'stage':stage,**extra})

        def evaluate_endpoint(folder, checkpoint, target, kind):
            evaluate(folder, checkpoint, EnvConfig.from_dict(cfg.to_dict()), protocol['cases'], target)
            evaluated = read(folder/f'validation_{target}.json')
            row = {'kind':kind,'target':target,'checkpoint':str(checkpoint),
                'full_task_success':sum(e['success'] for e in evaluated['episodes'] if 'label' not in e),
                'retention':{str(d):sum(e['success'] for e in evaluated['episodes'] if e.get('label')==f'retention_{d}m') for d in (6,12,18,23)},
                'evaluation_controls':sum(e['physics_steps'] for e in evaluated['episodes'])}
            if kind == 'PPO' and target == 0 and 'zero_update_reference' in protocol:
                assert all(row[key] == value for key,value in protocol['zero_update_reference'].items()), 'Zero-update endpoint changed from audited supervised parent'
            results.append(row)
            write_json(batch/'progress.json', results)
            print(json.dumps(row), flush=True)

        try:
            for setting in protocol['rounds']:
                index = setting['index']
                folder = batch/f'round_{index:02d}'
                folder.mkdir()
                rows = []
                fresh = {k:[] for k in ('observations','actions','returns','train_mask','case_id','teacher_executed')}
                for case_id, case in enumerate(protocol['collection_cases']):
                    heartbeat('collection', round=index, case_id=case_id)
                    config = EnvConfig.from_dict(cfg.to_dict())
                    config.scenario, config.split = case['scenario'], 'train'
                    with gzip.open(folder/f'collection_{case_id:03d}.jsonl.gz','wt',encoding='utf-8') as stream:
                        def record(row):
                            stream.write(json.dumps(row,default=jsonable)+'\n')
                            if time.time()-last_heartbeat > 5:
                                stream.flush()
                                heartbeat('collection', round=index, case_id=case_id, control=row['teacher_query']['control'])
                        sample = collect_recovery_episode(config, model, norm, case['seed'], setting['teacher_probability'],
                            mixture_seed=100000*index+case['seed'], callback=record,
                            teacher_mode=protocol.get('teacher_mode','held_reference'))
                    for key in ('observations','actions','returns','teacher_executed'):
                        fresh[key].extend(sample[key])
                    count = sample['teacher_queries']
                    fresh['case_id'].extend([case_id]*count)
                    fresh['train_mask'].extend([case['data_split']=='train']*count)
                    rows.append({'case_id':case_id,**case,'initial':sample['initial'],'mixture_metrics':sample['metrics'],
                        'teacher_queries':count,'teacher_executed_controls':int(sample['teacher_executed'].sum()),
                        'learner_executed_controls':int((~sample['teacher_executed']).sum()),
                        'teacher_decisions':sample['teacher_decisions'],'teacher_query_seconds':sample['teacher_query_seconds'],
                        'teacher_control_failures':sample['teacher_control_failures']})
                    write_json(folder/'collection_progress.json', rows)
                    print(json.dumps({'round':index,'case_id':case_id,'mixture_success':sample['metrics']['success'],
                        'reason':sample['metrics']['termination_reason'],'controls':count}),flush=True)
                fresh = {k:np.asarray(v,dtype=bool if k in ('train_mask','teacher_executed') else np.int32 if k=='case_id' else np.float32) for k,v in fresh.items()}
                np.savez_compressed(folder/'recovery_examples.npz', **fresh)
                collected = {'round':index,'teacher_probability':setting['teacher_probability'],
                    'teacher_mode':protocol.get('teacher_mode','held_reference'),'episodes':rows,
                    'data_sha256':sha(folder/'recovery_examples.npz'),'source_model_round':index-1,
                    'controls':len(fresh['case_id']),'teacher_queries':len(fresh['case_id']),
                    'teacher_executed_controls':int(fresh['teacher_executed'].sum()),
                    'learner_executed_controls':int((~fresh['teacher_executed']).sum()),
                    'failed_episodes_retained':sum(not r['mixture_metrics']['success'] for r in rows)}
                write_json(folder/'collection.json', collected)
                collections.append(collected)
                for key in arrays:
                    arrays[key].append(fresh[key])
                masks.append(fresh['train_mask'])
                combined = {k:np.concatenate(v) for k,v in arrays.items()}
                train_mask = np.concatenate(masks)
                before = actor_vector(model).clone()
                critic_before = {name:t.detach().clone() for name,t in model.policy.named_parameters()
                                 if 'value_net' in name or name=='log_std'} if protocol.get('value_weight',.5)==0 else None
                def fit_record(row):
                    heartbeat('supervision', round=index, **row)
                    print(json.dumps({'round':index,**row}),flush=True)
                history = fit_demonstrations(model,norm,combined['observations'],combined['actions'],combined['returns'],train_mask,
                    epochs=protocol['supervised_epochs_per_round'],seed=protocol.get('supervised_seed_base',11)+index,
                    callback=fit_record,update_normalization=False,value_weight=protocol.get('value_weight',.5))
                np.testing.assert_array_equal(stats.mean,norm.obs_rms.mean)
                np.testing.assert_array_equal(stats.var,norm.obs_rms.var)
                assert stats.count == norm.obs_rms.count
                assert model._n_updates == 0 and not model.policy.optimizer.state
                if critic_before is not None:
                    assert all(torch.equal(critic_before[name],t) for name,t in model.policy.named_parameters() if name in critic_before)
                checkpoint = folder/'supervised_checkpoint'
                checkpoint.mkdir()
                model.save(checkpoint/'model.zip'); norm.save(checkpoint/'normalization.pkl')
                shutil.copy2(batch/'initial/config.json',checkpoint/'config.json')
                trained = {'round':index,'history':history,'training_examples':int(train_mask.sum()),
                    'holdout_examples':int((~train_mask).sum()),'normalization_frozen':True,
                    'actor_l2_change_supervised':float(torch.linalg.vector_norm(actor_vector(model)-before)),
                    'ppo_updates':0,'model_sha256':sha(checkpoint/'model.zip'),'normalization_sha256':sha(checkpoint/'normalization.pkl')}
                write_json(folder/'supervision.json', trained); supervision.append(trained)
                evaluate_endpoint(folder,checkpoint,0,f'supervised_round_{index}')
            if not protocol['ppo_milestones']:
                write_json(batch/'completed.json', {'scope':protocol['scope'],'results':results,
                    'new_collection_controls':sum(c['controls'] for c in collections),
                    'teacher_queries':sum(c['teacher_queries'] for c in collections),
                    'new_supervised_optimizer_steps':sum(s['history'][-1]['supervised_optimizer_steps'] for s in supervision),
                    'final_ppo_controls':0,'inherited_costs':protocol['inherited_costs']})
                return
            # A new real PPO counter starts only after all declared supervised
            # aggregation. Ancestral controls and supervisor updates stay charged.
            folder = batch/'B1'
            initial = cumulative_train(folder,cfg,protocol['seed'],0,protocol['deadline_epoch']-120,ppo_options=protocol['ppo_options'])
            random_checkpoint = Path(initial['checkpoint'])
            checkpoint = folder/'supervised_initial'
            checkpoint.mkdir()
            model.save(checkpoint/'model.zip'); norm.save(checkpoint/'normalization.pkl')
            shutil.copy2(batch/'initial/config.json',checkpoint/'config.json')
            with (random_checkpoint/'resume.pkl').open('rb') as stream:
                snapshot = pickle.load(stream)
            snapshot['initial_actor'] = actor_vector(model).clone()
            with (checkpoint/'resume.pkl').open('wb') as stream:
                pickle.dump(snapshot,stream)
            initial.update(checkpoint=str(checkpoint),reason='supervised_recovery_initialization',actor_l2_change=0.)
            write_json(checkpoint/'training_summary.json',initial); atomic_json(folder/'latest.json',initial)
            for target in protocol['ppo_milestones']:
                heartbeat('PPO', target=target)
                saved = cumulative_train(folder,cfg,protocol['seed'],target,protocol['deadline_epoch']-120,ppo_options=protocol['ppo_options'])
                if saved['reason'] != 'milestone_reached':
                    raise RuntimeError('Incomplete PPO endpoint: '+saved['reason'])
                evaluate_endpoint(folder,Path(saved['checkpoint']),target,'PPO')
                results[-1].update(actual_ppo_controls=saved['physics_steps'],ppo_optimizer_steps=saved['optimizer_steps'],
                                   actor_l2_change_after_supervision=saved['actor_l2_change'])
                write_json(batch/'progress.json',results)
            write_json(batch/'completed.json', {'scope':protocol['scope'],'results':results,
                'new_collection_controls':sum(c['controls'] for c in collections),
                'teacher_queries':sum(c['teacher_queries'] for c in collections),
                'new_supervised_optimizer_steps':sum(s['history'][-1]['supervised_optimizer_steps'] for s in supervision),
                'final_ppo_controls':results[-1]['actual_ppo_controls'],'inherited_costs':protocol['inherited_costs']})
        except BaseException as exc:
            write_json(batch/'failure.json', {'epoch':time.time(),'error':repr(exc),'results':results})
            raise
        finally:
            norm.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent')
    parser.add_argument('--batch')
    args = parser.parse_args()
    prepare(args.parent) if args.parent else run(args.batch)

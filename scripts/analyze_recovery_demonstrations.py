"""Verify aggregation labels, executed mixture, heldout isolation and endpoints."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from navstack.experiments import load_run, actor_vector


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(batch, partial=False, output=None):
    torch.set_num_threads(1)
    batch = Path(batch)
    protocol = read(batch/'protocol.json')
    for name,digest in protocol['source_hashes'].items():
        assert sha(batch/'source'/name) == digest
    for name,digest in protocol['initial_hashes'].items():
        assert sha(batch/'initial'/name) == digest
    assert sha(batch/'parent_data/demonstrations.npz') == protocol['data_sha256']
    complete = (batch/'completed.json').exists()
    if not complete and not partial:
        raise ValueError('Incomplete; --partial explicitly permits provisional analysis')
    if complete:
        guard = read(batch/'guard_final.json')
        assert guard['exit_code'] == 0 and guard['reason'] is None
        assert not (batch/'failure.json').exists()
    original_data = np.load(batch/'parent_data/demonstrations.npz')
    original_cases = read(batch/'parent_data/protocol.json')['cases']
    training = sum(original_cases[int(i)]['data_split']=='train' for i in original_data['case_id'])
    heldout = len(original_data['case_id'])-training
    source_model, initial_norm = load_run(batch/'initial')
    initial_stats = initial_norm.obs_rms
    rounds, hashes = [], {}
    try:
        for setting in protocol['rounds']:
            index = setting['index']
            folder = batch/f'round_{index:02d}'
            if not (folder/'supervision.json').exists():
                assert not complete
                break
            collection, fit = read(folder/'collection.json'), read(folder/'supervision.json')
            assert collection['teacher_probability'] == setting['teacher_probability']
            data_path = folder/'recovery_examples.npz'
            assert sha(data_path) == collection['data_sha256']
            data = np.load(data_path)
            assert len(collection['episodes']) == len(protocol['collection_cases'])
            assert data['observations'].shape == (collection['controls'],605)
            assert data['actions'].shape == (collection['controls'],2)
            # Independently reload the preceding checkpoint and reconstruct all
            # learner predictions. A trace's "learner" label alone is not proof.
            predicted = np.concatenate([
                source_model.predict(initial_norm.normalize_obs(data['observations'][start:start+512]),
                                     deterministic=True)[0]
                for start in range(0, len(data['observations']), 512)
            ])
            prediction_max_error = 0.
            single_row_checks = 0
            teacher_count, total, queries, query_seconds = 0, 0, 0, 0.
            for case_id,(episode,case) in enumerate(zip(collection['episodes'],protocol['collection_cases'])):
                assert episode['case_id'] == case_id and all(episode[k]==v for k,v in case.items())
                selected = np.flatnonzero(data['case_id']==case_id)
                np.testing.assert_array_equal(data['train_mask'][selected],case['data_split']=='train')
                trace = folder/f'collection_{case_id:03d}.jsonl.gz'
                hashes[trace.relative_to(batch).as_posix()] = sha(trace)
                previous = episode['initial']['initial']
                terminal = None
                own_teacher, own_queries, own_seconds = 0, 0, 0.
                with gzip.open(trace,'rt',encoding='utf-8') as stream:
                    count = 0
                    for control,line in enumerate(stream):
                        row = json.loads(line)
                        assert terminal is None
                        assert row['state_before'] == previous
                        assert row['teacher_query']['control'] == control
                        assert row['teacher_query']['teacher_decision'] == (protocol.get('teacher_mode')=='current_state' or control%10==0)
                        teacher = row['executed_by']=='teacher'
                        if setting['teacher_probability'] == 0:
                            assert not teacher, 'Teacher executed in pure-learner collection'
                        if setting['teacher_probability'] == 1:
                            assert teacher, 'Learner executed in all-teacher collection'
                        assert row['executed_by'] in ('teacher','learner')
                        error = float(np.max(np.abs(predicted[selected[control]]-row['learner_action'])))
                        prediction_max_error = max(prediction_max_error, error)
                        # Batched CPU GEMM can differ from the collector's one-row
                        # inference. Resolve larger differences with that exact
                        # computation, rather than widening the error tolerance.
                        if error > 2e-6:
                            single = source_model.predict(initial_norm.normalize_obs(
                                data['observations'][selected[control]:selected[control]+1]),deterministic=True)[0][0]
                            np.testing.assert_array_equal(single,row['learner_action'])
                            single_row_checks += 1
                        assert bool(data['teacher_executed'][selected[control]]) == teacher
                        np.testing.assert_array_equal(data['actions'][selected[control]],row['teacher_action'])
                        expected = row['teacher_action'] if teacher else row['learner_action']
                        np.testing.assert_array_equal(row['executed_action'],expected)
                        command = row['teacher_query']['requested_command']
                        vehicle = protocol['env']['vehicle']
                        label = np.array([command['acceleration']/(vehicle['max_accel'] if command['acceleration']>=0 else vehicle['max_decel']),
                                          command['steering']/vehicle['max_steer']],dtype=np.float32).clip(-1,1)
                        np.testing.assert_array_equal(data['actions'][selected[control]],label)
                        info = row['info']
                        assert len(info['trace']) == info['executed_substeps'] == 1
                        actual = info['trace'][0]['requested_command']
                        expected = np.clip(expected,-1,1)
                        assert abs(actual['acceleration']-float(expected[0])*(vehicle['max_accel'] if expected[0]>=0 else vehicle['max_decel'])) < 1e-6
                        assert abs(actual['steering']-float(expected[1])*vehicle['max_steer']) < 1e-6
                        previous = info['trace'][0]['state']
                        assert abs(previous['time']-row['state_before']['time']-.05) < 1e-8
                        own_teacher += int(teacher)
                        own_queries += 1
                        own_seconds += row['teacher_query']['query_seconds']
                        count += 1
                        if info['termination_reason']:
                            terminal = info['episode_metrics']
                assert count == len(selected) == episode['mixture_metrics']['physics_steps']
                assert terminal == episode['mixture_metrics']
                assert own_teacher == episode['teacher_executed_controls']
                assert count-own_teacher == episode['learner_executed_controls']
                assert own_queries == episode['teacher_queries']
                assert abs(own_seconds-episode['teacher_query_seconds']) < 1e-6
                teacher_count += own_teacher
                total += count
                queries += own_queries
                query_seconds += own_seconds
            assert total == collection['controls'] == collection['teacher_queries']
            assert teacher_count == collection['teacher_executed_controls']
            assert total-teacher_count == collection['learner_executed_controls']
            training += int(data['train_mask'].sum())
            heldout += int((~data['train_mask']).sum())
            assert (training,heldout) == (fit['training_examples'],fit['holdout_examples'])
            assert fit['history'][-1]['supervised_epoch'] == protocol['supervised_epochs_per_round']
            checkpoint = folder/'supervised_checkpoint'
            model,norm = load_run(checkpoint)
            assert sha(checkpoint/'model.zip') == fit['model_sha256']
            assert sha(checkpoint/'normalization.pkl') == fit['normalization_sha256']
            np.testing.assert_array_equal(norm.obs_rms.mean,initial_stats.mean)
            np.testing.assert_array_equal(norm.obs_rms.var,initial_stats.var)
            assert norm.obs_rms.count == initial_stats.count
            assert model._n_updates == 0 and not model.policy.optimizer.state
            change = float(torch.linalg.vector_norm(actor_vector(model)-actor_vector(source_model)))
            assert change > 0 and abs(change-fit['actor_l2_change_supervised']) < 1e-6
            if protocol.get('value_weight',.5) == 0:
                parameters = dict(model.policy.named_parameters())
                for name,value in source_model.policy.named_parameters():
                    if 'value_net' in name or name == 'log_std':
                        assert torch.equal(value,parameters[name]), 'Actor-only fit changed critic or exploration scale'
            source_model = model
            norm.close()
            assert fit['actor_l2_change_supervised'] > 0
            rounds.append({'round':index,'controls':total,'teacher_queries':queries,'teacher_executed_controls':teacher_count,
                'learner_executed_controls':total-teacher_count,'teacher_query_seconds':query_seconds,
                'mixture_successes':sum(e['mixture_metrics']['success'] for e in collection['episodes']),
                'mixture_termination_reasons':dict(Counter(e['mixture_metrics']['termination_reason'] for e in collection['episodes'])),
                'training_examples':training,'heldout_examples':heldout,'supervised_optimizer_steps':fit['history'][-1]['supervised_optimizer_steps'],
                'learner_prediction_max_error':prediction_max_error,
                'exact_single_row_prediction_checks':single_row_checks,
                'actor_l2_change_verified':change,
                'critic_and_log_std_unchanged':True if protocol.get('value_weight',.5)==0 else None,
                'final_train_action_mse':fit['history'][-1]['train_action_mse'],'final_holdout_action_mse':fit['history'][-1]['holdout_action_mse']})
        results = read(batch/'progress.json') if (batch/'progress.json').exists() else []
        expected = [(f"supervised_round_{s['index']}",0) for s in protocol['rounds']]+[('PPO',t) for t in protocol['ppo_milestones']]
        assert [(r['kind'],r['target']) for r in results] == expected[:len(results)]
        if complete:
            assert len(results)==len(expected) and len(rounds)==len(protocol['rounds'])
        endpoints = []
        for result in results:
            folder = batch/('B1' if result['kind']=='PPO' else 'round_'+result['kind'].rsplit('_',1)[1].zfill(2))
            checkpoint = folder/Path(result['checkpoint']).name
            evaluation_path = folder/f"validation_{result['target']}.json"
            evaluation = read(evaluation_path)
            assert sha(checkpoint/'model.zip') == evaluation['model_sha256']
            assert len(evaluation['episodes']) == len(protocol['cases'])
            counts, terminal = Counter(), {}
            trace = folder/f"validation_{result['target']}_trace.jsonl.gz"
            hashes[trace.relative_to(batch).as_posix()] = sha(trace)
            with gzip.open(trace,'rt',encoding='utf-8') as stream:
                for line in stream:
                    item = json.loads(line)
                    case_id,info = item['case_id'],item['info']
                    assert case_id not in terminal
                    assert len(info['trace']) == info['executed_substeps'] == 1
                    assert all(t['controller_status']=='direct' for t in info['trace']), 'Teacher at evaluation'
                    counts[case_id] += len(info['trace'])
                    tick = info['trace'][0]
                    action = np.asarray(item['action']).clip(-1,1)
                    vehicle = protocol['env']['vehicle']
                    assert abs(tick['requested_command']['acceleration']-float(action[0])*(vehicle['max_accel'] if action[0]>=0 else vehicle['max_decel'])) < 1e-6
                    assert abs(tick['requested_command']['steering']-float(action[1])*vehicle['max_steer']) < 1e-6
                    assert abs(tick['state']['time']-counts[case_id]*protocol['env']['control_dt']) < 1e-8
                    if info['termination_reason']:
                        terminal[case_id] = info['episode_metrics']
            for case_id,(episode,case) in enumerate(zip(evaluation['episodes'],protocol['cases'])):
                assert all(episode.get(k)==case.get(k) for k in ('scenario','seed','label','options'))
                assert all(episode[k]==v for k,v in terminal[case_id].items())
                assert counts[case_id] == episode['physics_steps']
            full = [e for e in evaluation['episodes'] if 'label' not in e]
            assert sum(e['success'] for e in full) == result['full_task_success']
            assert sum(counts.values()) == result['evaluation_controls']
            assert {str(d):sum(e['success'] for e in evaluation['episodes'] if e.get('label')==f'retention_{d}m') for d in (6,12,18,23)} == result['retention']
            if result['kind']=='PPO':
                saved = read(checkpoint/'training_summary.json')
                assert saved['physics_steps'] == result['actual_ppo_controls'] >= result['target']
                assert saved['optimizer_steps'] == result['ppo_optimizer_steps']
                assert saved['actor_l2_change'] == result['actor_l2_change_after_supervision']
                if result['target'] == 0:
                    assert saved['physics_steps'] == saved['optimizer_steps'] == saved['actor_l2_change'] == 0
                else:
                    assert saved['optimizer_steps'] > 0 and saved['actor_l2_change'] > 0
            endpoints.append({**result,'reasons':dict(Counter(e['termination_reason'] for e in full))})
        if complete:
            completed = read(batch/'completed.json')
            assert completed['new_collection_controls'] == sum(r['controls'] for r in rounds)
            assert completed['teacher_queries'] == sum(r['teacher_queries'] for r in rounds)
            assert completed['new_supervised_optimizer_steps'] == sum(r['supervised_optimizer_steps'] for r in rounds)
        if complete and protocol['ppo_milestones']:
            train_controls, previous_time, terminals = 0, {}, {}
            for trace in sorted((batch/'B1').glob('transitions*.jsonl.gz')):
                hashes[trace.relative_to(batch).as_posix()] = sha(trace)
                with gzip.open(trace,'rt',encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        assert row['mode']=='flat' and row['plan'] is None
                        assert len(row['trace'])==1 and row['trace'][0]['controller_status']=='direct'
                        train_controls += 1
                        assert row['physics_steps']==train_controls
                        episode = row['episode_id']
                        assert episode not in terminals, 'Training control after termination'
                        clock = row['trace'][0]['state']['time']
                        assert abs(clock-previous_time.get(episode,0.)-.05)<1e-8
                        previous_time[episode] = clock
                        if row['termination_reason']:
                            terminals[episode]=row['termination_reason']
            assert train_controls==completed['final_ppo_controls']==endpoints[-1]['actual_ppo_controls']
            updates=[json.loads(line) for line in (batch/'B1/optimization.jsonl').read_text(encoding='utf-8').splitlines()]
            assert updates[-1]['optimizer_steps']==endpoints[-1]['ppo_optimizer_steps']
            training_audit={'actual_direct_controls':train_controls,'optimization_phases':len(updates),
                'optimizer_steps':updates[-1]['optimizer_steps'],'termination_reasons':dict(Counter(terminals.values()))}
        else:
            training_audit=None
            if complete:
                assert completed['final_ppo_controls']==0
        output = Path(output) if output else batch/'analysis'
        output.mkdir(exist_ok=True)
        fig,axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
        axes[0].bar([str(r['round']) for r in rounds],[r['teacher_executed_controls'] for r in rounds],label='Teacher executes')
        axes[0].bar([str(r['round']) for r in rounds],[r['learner_executed_controls'] for r in rounds],bottom=[r['teacher_executed_controls'] for r in rounds],label='Learner executes')
        axes[0].set(xlabel='Aggregation round',ylabel='Actual collection controls',title='Data collection (not development evaluation)')
        axes[0].legend()
        axes[1].bar(['Supervised\nround '+r['kind'].rsplit('_',1)[1] if r['kind']!='PPO'
                     else 'PPO\n'+format(r['actual_ppo_controls'],',') for r in endpoints],
                    [r['full_task_success']/10*100 for r in endpoints])
        axes[1].set(ylabel='Completion (%)',ylim=(0,105),title='Teacher-free full development task')
        axes[1].tick_params(axis='x',labelsize=8)
        fig.suptitle('One supervised development seed; '+('complete' if complete else 'PROVISIONAL'))
        fig.savefig(output/'recovery_learning.png',dpi=160);plt.close(fig)
        summary = {'scope':protocol['scope'],'complete':complete,'rounds':rounds,'endpoints':endpoints,
            'inherited_costs':protocol['inherited_costs'],'input_hashes':hashes,'training_audit':training_audit,
            'limitations':'Mixture controls and oracle queries are additional supervision; not pure PPO, independent seeds, or final confirmation.'}
        (output/'verified_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in summary.items() if k!='input_hashes'}))
    finally:
        initial_norm.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--partial',action='store_true');p.add_argument('--output')
    args=p.parse_args();analyze(args.batch,args.partial,args.output)

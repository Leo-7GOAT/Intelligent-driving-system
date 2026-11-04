"""Raw-control audit for cumulative paired normalization retention training."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from navstack.experiments import actor_vector,load_run
from scripts.train_recovery_demonstrations import read,sha


def verify_command(row, vehicle):
    assert row['mode']=='flat' and row['plan'] is None and len(row['trace'])==1
    tick=row['trace'][0]
    assert tick['controller_status']=='direct'
    action=np.asarray(row['policy_action']).clip(-1,1)
    command=tick['requested_command']
    assert abs(command['acceleration']-float(action[0])*(vehicle['max_accel'] if action[0]>=0 else vehicle['max_decel']))<1e-6
    assert abs(command['steering']-float(action[1])*vehicle['max_steer'])<1e-6
    return tick


def analyze(batch, partial=False, output=None):
    batch=Path(batch).resolve();p=read(batch/'protocol.json')
    complete=(batch/'completed.json').exists()
    if not complete and not partial:
        raise ValueError('Incomplete continuation; explicitly request --partial')
    if complete:
        guard=read(batch/'guard_final.json')
        assert guard['exit_code']==0 and guard['reason'] is None
    for name,digest in p['source_hashes'].items():assert sha(batch/'source'/name)==digest
    progress=read(batch/'progress.json') if (batch/'progress.json').exists() else []
    assert [(r['target'],r['branch']) for r in progress]==[
        (target,b['name']) for target in p['milestones'] for b in p['branches']][:len(progress)]
    if complete:assert len(progress)==len(p['milestones'])*len(p['branches'])
    results=[];torch.set_num_threads(1)
    for branch in p['branches']:
        name=branch['name'];folder=batch/name;initial_cp=folder/'initial_checkpoint'
        for filename,digest in p['initial_checkpoint_hashes'][name].items():
            assert sha(initial_cp/filename)==digest
        assert read(folder/'ppo_options.json')==branch['ppo_options']
        with (initial_cp/'resume.pkl').open('rb') as stream:snapshot=pickle.load(stream)
        with (initial_cp/'normalization.pkl').open('rb') as stream:initial_norm=pickle.load(stream)
        initial_summary=read(initial_cp/'training_summary.json')
        assert initial_summary['physics_steps']==p['inherited_controls']
        initial_model,normalizer=load_run(initial_cp)
        initial_actor=actor_vector(initial_model).clone();normalizer.close()
        selected=[row for row in progress if row['branch']==name];endpoints=[]
        for row in selected:
            target=row['target'];evaluation=read(folder/f'validation_{target}.json')
            cp=folder/Path(evaluation['checkpoint']).name;saved=read(cp/'training_summary.json')
            assert sha(cp/'model.zip')==evaluation['model_sha256']
            assert saved['reload_exact'] and saved['physics_steps']==row['actual_physics_steps']
            assert row['additional_physics_steps']==saved['physics_steps']-p['inherited_controls']
            model,norm=load_run(cp)
            learned=float(torch.linalg.vector_norm(actor_vector(model)-initial_actor))
            assert learned>0 and model._n_updates>initial_model._n_updates
            assert abs(float(torch.linalg.vector_norm(actor_vector(model)-snapshot['initial_actor']))-saved['actor_l2_change'])<1e-5
            norm.close()
            terminals={};counts=Counter()
            with gzip.open(folder/f'validation_{target}_trace.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:
                    raw=json.loads(line);info=raw['info'];case=raw['case_id']
                    assert case not in terminals
                    tick=verify_command({'policy_action':raw['action'],**info},p['env']['vehicle'])
                    counts[case]+=1
                    assert abs(tick['state']['time']-counts[case]*p['env']['control_dt'])<1e-8
                    if info['termination_reason']:terminals[case]=info['episode_metrics']
            assert len(terminals)==len(evaluation['episodes'])==len(p['cases'])
            for i,(episode,case) in enumerate(zip(evaluation['episodes'],p['cases'],strict=True)):
                assert all(episode.get(k)==case.get(k) for k in ('scenario','seed','label','options'))
                assert all(episode[k]==v for k,v in terminals[i].items())
                assert counts[i]==episode['physics_steps']
            full=[e for e in evaluation['episodes'] if 'label' not in e]
            assert sum(e['success'] for e in full)==row['full_task_success']
            assert {str(d):sum(e['success'] for e in evaluation['episodes'] if e.get('label')==f'retention_{d}m') for d in (6,12,18,23)}==row['retention']
            endpoints.append({**row,'model_sha256':evaluation['model_sha256'],
                'new_actor_l2':learned,'optimizer_steps':saved['optimizer_steps'],
                'evaluation_controls':sum(counts.values()),'reasons':dict(Counter(e['termination_reason'] for e in full))})
        cutoff=endpoints[-1]['actual_physics_steps'] if endpoints else p['inherited_controls']
        # Only fully closed transition chunks before the last evaluated endpoint
        # are read. A later live milestone is not mistaken for audited progress.
        total=p['inherited_controls'];terminals={}
        clocks={snapshot['counts']['episodes']:snapshot['env']['plant'].state.time}
        for path in sorted(folder.glob('transitions_*.jsonl.gz')):
            if int(path.name.split('_')[1])>=cutoff:continue
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);episode=row['episode_id'];assert episode not in terminals
                    tick=verify_command(row,p['env']['vehicle']);total+=1
                    assert row['physics_steps']==total<=cutoff
                    clock=tick['state']['time']
                    assert abs(clock-clocks.get(episode,0)-p['env']['control_dt'])<1e-8
                    clocks[episode]=clock
                    if row['termination_reason']:terminals[episode]=row['termination_reason']
        assert total==cutoff
        updates=[]
        if (folder/'optimization.jsonl').exists():
            updates=[json.loads(line) for line in (folder/'optimization.jsonl').read_text(encoding='utf-8').splitlines()]
            updates=[r for r in updates if r['physics_steps']<=cutoff]
        if endpoints:
            assert updates[-1]['optimizer_steps']==endpoints[-1]['optimizer_steps']
        frozen=branch['ppo_options'].get('freeze_observation_stats',False);checked=[]
        for marker in sorted(folder.glob('checkpoint_*/training_summary.json')):
            saved=read(marker)
            if saved['physics_steps']>cutoff:continue
            with (marker.parent/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
            assert norm.training== (not frozen) and not norm.norm_reward
            if frozen:
                for key in ('mean','var','count'):
                    assert np.array_equal(getattr(norm.obs_rms,key),getattr(initial_norm.obs_rms,key))
            else:
                assert abs(norm.obs_rms.count-initial_norm.obs_rms.count-(saved['physics_steps']-p['inherited_controls']))<1e-6
            checked.append(marker.parent.name)
        results.append({'branch':name,'endpoints':endpoints,'new_training_controls':total-p['inherited_controls'],
            'new_optimizer_steps':updates[-1]['optimizer_steps']-initial_summary['optimizer_steps'] if updates else 0,
            'optimization_phases_added':len(updates),'termination_reasons':dict(Counter(terminals.values())),
            'frozen_observation_stats':frozen,'normalization_checkpoints_checked':checked})
    result={'scope':p['scope'],'complete':complete,'inherited_controls_per_branch':p['inherited_controls'],'branches':results}
    destination=Path(output) if output else batch/'continuation_verification.json'
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True)
    parser.add_argument('--partial',action='store_true');parser.add_argument('--output')
    args=parser.parse_args();analyze(args.batch,args.partial,args.output)

"""Audit exploration-only branching, separating intervention from PPO drift."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from navstack.experiments import load_run,actor_vector


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(batch, partial=False):
    batch=Path(batch).resolve();p=read(batch/'protocol.json')
    complete=(batch/'completed.json').exists()
    if not complete and not partial: raise ValueError('Incomplete; explicitly request --partial')
    if complete:
        guard=read(batch/'guard_final.json');assert guard['exit_code']==0 and guard['reason'] is None
    for name,digest in p['source_hashes'].items(): assert sha(batch/'source'/name)==digest
    branch=p['branches'][0]['name'];folder=batch/branch
    for name,digest in p['initial_checkpoint_hashes'][branch].items():
        assert sha(folder/'initial_checkpoint'/name)==digest
    parent=batch.parent/p['parent_batch'];old=read(parent/'protocol.json')
    assert old['env']==p['env'] and old['seed']==p['seed'] and old['cases']==p['cases']
    assert sha(parent/'analysis/verified_summary.json')==p['parent_audit_sha256']
    zero=read(parent/'B1/validation_0.json');baseline_cp=parent/'B1'/Path(zero['checkpoint']).name
    torch.set_num_threads(1)
    baseline,bn=load_run(baseline_cp);initial,normalizer=load_run(folder/'initial_checkpoint')
    try:
        for name,value in baseline.policy.state_dict().items():
            if name!='log_std': assert torch.equal(value,initial.policy.state_dict()[name])
        assert torch.all(baseline.policy.log_std==-2) and torch.all(initial.policy.log_std==-4)
        baseline_vector=actor_vector(baseline);initial_vector=actor_vector(initial)
        intervention=float(torch.linalg.vector_norm(initial_vector-baseline_vector))
    finally:
        bn.close();normalizer.close()
    progress=read(batch/'progress.json')
    assert [r['target'] for r in progress]==p['milestones'][:len(progress)]
    if complete: assert len(progress)==len(p['milestones'])
    endpoints=[];vehicle=p['env']['vehicle']
    for r in progress:
        target=r['target'];evaluation=read(folder/f'validation_{target}.json')
        cp=folder/Path(evaluation['checkpoint']).name;saved=read(cp/'training_summary.json')
        assert sha(cp/'model.zip')==evaluation['model_sha256']
        model,norm=load_run(cp)
        try:
            vector=actor_vector(model)
            learned=float(torch.linalg.vector_norm(vector-initial_vector))
            inherited=float(torch.linalg.vector_norm(vector-baseline_vector))
            assert abs(inherited-saved['actor_l2_change'])<1e-5
            assert learned==0 if target==0 else learned>0
            assert saved['reload_exact'] and saved['physics_steps']==r['actual_physics_steps']
        finally: norm.close()
        counts,terminals=Counter(),{}
        with gzip.open(folder/f'validation_{target}_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);case_id=row['case_id'];info=row['info']
                assert case_id not in terminals
                assert len(info['trace'])==info['executed_substeps']==1
                tick=info['trace'][0];assert tick['controller_status']=='direct'
                action=np.asarray(row['action']).clip(-1,1);command=tick['requested_command']
                assert abs(command['acceleration']-float(action[0])*(vehicle['max_accel'] if action[0]>=0 else vehicle['max_decel']))<1e-6
                assert abs(command['steering']-float(action[1])*vehicle['max_steer'])<1e-6
                counts[case_id]+=1
                assert abs(tick['state']['time']-counts[case_id]*p['env']['control_dt'])<1e-8
                if info['termination_reason']: terminals[case_id]=info['episode_metrics']
        assert len(terminals)==len(evaluation['episodes'])==len(p['cases'])
        for i,(episode,case) in enumerate(zip(evaluation['episodes'],p['cases'],strict=True)):
            assert all(episode.get(k)==case.get(k) for k in ('scenario','seed','label','options'))
            assert all(episode[k]==v for k,v in terminals[i].items())
            assert counts[i]==episode['physics_steps']
        full=[e for e in evaluation['episodes'] if 'label' not in e]
        assert sum(e['success'] for e in full)==r['full_task_success']
        assert {str(d):sum(e['success'] for e in evaluation['episodes'] if e.get('label')==f'retention_{d}m') for d in (6,12,18,23)}==r['retention']
        endpoints.append({**r,'optimizer_steps':saved['optimizer_steps'],'evaluation_controls':sum(counts.values()),
            'ppo_actor_l2_from_changed_initial':learned,'saved_actor_l2_including_intervention':inherited,
            'reasons':dict(Counter(e['termination_reason'] for e in full)),'model_sha256':evaluation['model_sha256']})
    training_audit=None
    if complete:
        total=0;clocks={};terminals={}
        for path in sorted(folder.glob('transitions*.jsonl.gz')):
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);episode=row['episode_id']
                    assert episode not in terminals and row['mode']=='flat' and row['plan'] is None
                    assert len(row['trace'])==1 and row['trace'][0]['controller_status']=='direct'
                    tick=row['trace'][0];action=np.asarray(row['policy_action']).clip(-1,1);command=tick['requested_command']
                    assert abs(command['acceleration']-float(action[0])*(vehicle['max_accel'] if action[0]>=0 else vehicle['max_decel']))<1e-6
                    assert abs(command['steering']-float(action[1])*vehicle['max_steer'])<1e-6
                    total+=1;assert row['physics_steps']==total
                    clock=tick['state']['time'];assert abs(clock-clocks.get(episode,0.)-p['env']['control_dt'])<1e-8
                    clocks[episode]=clock
                    if row['termination_reason']: terminals[episode]=row['termination_reason']
        updates=[json.loads(line) for line in (folder/'optimization.jsonl').read_text(encoding='utf-8').splitlines()]
        assert total==endpoints[-1]['actual_physics_steps']
        assert updates[-1]['optimizer_steps']==endpoints[-1]['optimizer_steps']
        training_audit={'controls':total,'optimizer_steps':updates[-1]['optimizer_steps'],
                        'phases':len(updates),'termination_reasons':dict(Counter(terminals.values()))}
    result={'scope':'Exploration branch diagnostic; explicit parameter intervention separated from subsequent PPO changes',
        'complete':complete,'initial_log_std_intervention_l2':intervention,'endpoints':endpoints,
        'inherited_costs':p['inherited_costs'],'training_audit':training_audit,
        'note':'The unchanged resume ledger references the original pre-intervention actor. Its actor_l2_change includes the manual log_std shift and is not a PPO-only parameter change.'}
    (batch/'exploration_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();analyze(args.batch,args.partial)

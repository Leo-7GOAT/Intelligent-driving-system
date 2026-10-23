"""Rebuild all predeclared paired stability endpoints; no best-point selection."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def analyze(batch):
    batch=Path(batch);protocol=read(batch/'protocol.json')
    complete=read(batch/'completed.json')
    for relative,digest in protocol['source_hashes'].items():
        assert hashlib.sha256((batch/'source'/relative).read_bytes()).hexdigest()==digest
    fig,axes=plt.subplots(1,4,figsize=(18,4.6),constrained_layout=True)
    rows,hashes=[],{}
    for branch in protocol['branches']:
        name=branch['name'];folder=batch/name
        initial_hashes=protocol.get('initial_checkpoint_hashes',{}).get(name,protocol['checkpoint_hashes'])
        for filename,digest in initial_hashes.items():
            assert hashlib.sha256((folder/'initial_checkpoint'/filename).read_bytes()).hexdigest()==digest
        selected=[r for r in complete['results'] if r['branch']==name]
        assert [r['target'] for r in selected]==protocol['milestones']
        reasons={}
        for result in selected:
            path=folder/f"validation_{result['target']}.json";value=read(path)
            hashes[path.relative_to(batch).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            assert len(value['episodes'])==len(protocol['cases'])
            for episode,case in zip(value['episodes'],protocol['cases']):
                assert all(episode.get(key)==case.get(key) for key in ('scenario','seed','label','options'))
            checkpoint=folder/Path(value['checkpoint']).name
            assert hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest()==value['model_sha256']
            for filename in ('model.zip','normalization.pkl','config.json','training_summary.json'):
                item=checkpoint/filename
                hashes[item.relative_to(batch).as_posix()]=hashlib.sha256(item.read_bytes()).hexdigest()
            assert sum(e['success'] for e in value['episodes'] if 'label' not in e)==result['full_task_success']
            assert {str(d):sum(e['success'] for e in value['episodes'] if e.get('label')==f'retention_{d}m') for d in (6,12,18,23)}==result['retention']
            reasons[str(result['target'])]=[{'scenario':e['scenario'],'seed':e['seed'],'label':e.get('label'),
                'success':e['success'],'reason':e['termination_reason']} for e in value['episodes']]
        updates=[json.loads(line) for line in (folder/'optimization.jsonl').read_text(encoding='utf-8').splitlines()]
        assert updates and all(r['train/learning_rate']==branch['learning_rate'] for r in updates)
        x=[r['additional_physics_steps']/1000 for r in selected]
        axes[0].plot(x,[r['full_task_success']/10*100 for r in selected],'-o',label=name)
        axes[1].plot(x,[sum(e['success'] for e in reasons[str(r['target'])] if e['scenario']=='static')/2*100 for r in selected],'-o',label=name)
        axes[2].plot(x,[r['retention']['23']/2*100 for r in selected],'-o',label=name)
        kl=[r['train/approx_kl'] for r in updates]
        axes[3].plot([(r['physics_steps']-protocol['inherited_controls'])/1000 for r in updates],kl,'.-',alpha=.7,label=name)
        rows.append({'branch':name,'learning_rate':branch['learning_rate'],'milestones':selected,
            'observed_kl':{'median':float(np.median(kl)),'p95':float(np.quantile(kl,.95)),'max':max(kl)},
            'all_endpoint_outcomes':reasons,'optimization_phases_added':len(updates)})
    for axis in axes:
        axis.set_xlabel('Additional physical controls (thousands)');axis.grid(alpha=.2);axis.legend()
    axes[0].set(title='Complete development list (10 cases)',ylabel='Completion (%)',ylim=(-5,105))
    axes[1].set(title='Static obstacle task (2 cases)',ylabel='Completion (%)',ylim=(-5,105))
    axes[2].set(title='23 m parking retention (2 probes)',ylabel='Completion (%)',ylim=(-5,105))
    axes[3].set(title='Observed PPO update KL',ylabel='Approximate KL')
    fig.suptitle('One seed, two correlated branches; all predeclared endpoints, not final confirmation')
    output=batch/'analysis';output.mkdir(exist_ok=True)
    fig.savefig(output/'update_stability.png',dpi=150);plt.close(fig)
    result={'scope':protocol['scope'],'inherited_controls_per_branch':protocol['inherited_controls'],'input_hashes':hashes,'branches':rows}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('branch','learning_rate','milestones','observed_kl')} for r in rows]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);analyze(p.parse_args().batch)

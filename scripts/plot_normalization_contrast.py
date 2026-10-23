"""Plot every registered endpoint of three correlated development branches."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def plot(baseline, noise, frozen, output):
    paths = [Path(baseline),Path(noise),Path(frozen)]
    records = [read(path) for path in paths]
    assert all(record['complete'] for record in records), 'No partial curve as final evidence'
    assert records[2]['normalization_audit']['mean_variance_count_exact_at_every_endpoint']
    labels = ['Original noise, updating statistics',
              'Reduced noise, updating statistics',
              'Reduced noise, frozen statistics']
    rows = []
    for label, record in zip(labels,records,strict=True):
        assert [e['target'] for e in record['endpoints']] == [0,25000,100000]
        for endpoint in record['endpoints']:
            reasons = endpoint['reasons']
            assert sum(reasons.values()) == 10
            assert set(endpoint['retention']) == {'6','12','18','23'}
            assert all(0 <= n <= 2 for n in endpoint['retention'].values())
            rows.append({'condition':label, 'target':endpoint['target'],
                'actual_controls':endpoint.get('actual_physics_steps',endpoint.get('actual_ppo_controls')),
                'optimizer_steps':endpoint.get('optimizer_steps',endpoint.get('ppo_optimizer_steps')),
                'full_successes':endpoint['full_task_success'], 'full_cases':10,
                'parking_successes':sum(endpoint['retention'].values()), 'parking_cases':8,
                'reasons':reasons})
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    summary = {'scope':'One policy lineage with inherited supervision; correlated sequential diagnostics, no independent-seed uncertainty or hierarchy claim',
        'input_hashes':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'rows':rows}
    (output/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,axes = plt.subplots(1,2,figsize=(10,5.3))
    for index,label in enumerate(labels):
        data=[row for row in rows if row['condition']==label]
        for axis,key,denom in [(axes[0],'full_successes',10),(axes[1],'parking_successes',8)]:
            axis.plot([row['actual_controls']/1000 for row in data],
                      [100*row[key]/denom for row in data],
                      marker=['o','s','^'][index],linestyle=['-','--',':'][index],
                      linewidth=1.7,markersize=7,markerfacecolor='none',label=label)
    for axis,title in zip(axes,['Full-task development (10 cases)','Parking probes (8 cases)'],strict=True):
        axis.set(title=title,xlabel='Additional PPO controls (thousands)',ylabel='Completion (%)',ylim=(-5,105))
        axis.set_xticks([0,25.6,102.4])
        axis.grid(alpha=.25)
    fig.suptitle('PPO retention: exploration and training normalization',fontsize=14,y=.97)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',bbox_to_anchor=(.5,.05),ncol=1,frameon=False)
    fig.text(.5,.015,'One inherited policy; all declared endpoints retained. No independent-seed confidence intervals.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.29,top=.86,wspace=.3)
    fig.savefig(output/'normalization_contrast.png',dpi=160,metadata={'Software':'navstack diagnostic plot'})
    plt.close(fig)
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('baseline','noise','frozen','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    plot(args.baseline,args.noise,args.frozen,args.output)

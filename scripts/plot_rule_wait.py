"""Plot an audited selected-state diagnosis; alternate paths are never execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from navstack.config import EnvConfig
from navstack.world import body_corners
from scripts.audit_native_method_endpoint import read,sha


def plot(batch,diagnostic,output):
    batch,diagnostic,output=Path(batch),Path(diagnostic),Path(output)
    receipt=output.with_suffix('.json')
    if output.exists() or receipt.exists():raise FileExistsError('Preserve figure evidence')
    report=read(diagnostic);protocol=read(batch/'protocol.json')
    scene_path=batch/f"scene_{report['case_id']:04d}.json"
    if sha(scene_path)!=report['scene_sha256'] or sha(batch/'protocol.json')!=report['protocol_sha256']:
        raise ValueError('Scene/protocol differs from the verified selected-state diagnosis')
    scene=read(scene_path)['scene'];config=EnvConfig.from_dict(protocol['config'])
    state=report['selection']['state'];candidates=report['candidate_references']['actions']
    fig,(ax,bars)=plt.subplots(1,2,figsize=(11,5),layout='constrained',gridspec_kw={'width_ratios':[1.7,1]})
    ax.imshow(scene['road_grid'].get('collision_drivable',scene['road_grid']['drivable']),origin='lower',extent=(0,scene['width'],0,scene['height']),
        cmap='Greys_r',vmin=0,vmax=1,alpha=.45,interpolation='nearest')
    route=np.asarray(scene['native_layout']['path']);ax.plot(route[:,0],route[:,1],'--',color='#7a8793',label='Known map route')
    trace=report['selection']['measured_prefix_path']
    ax.plot([s['x'] for s in trace],[s['y'] for s in trace],color='#c76528',lw=2,label='Measured CARLA prefix')
    targets={}
    for row in candidates[1:]:targets.setdefault(tuple(row['target']),[]).append(row)
    labels=set()
    for target,rows in targets.items():
        available=any(r['mode']=='reference' for r in rows)
        label='Reference available' if available else 'No reference in original budget'
        ax.scatter(*target[:2],s=30,marker='o' if available else 'x',color='#32865b' if available else '#a25058',
            label=label if label not in labels else None,zorder=3);labels.add(label)
        if available:ax.annotate('/'.join(str(r['action']) for r in rows),target[:2],xytext=(4,4),textcoords='offset points',fontsize=8)
    for query in report['direct_goal_queries']:
        if query['status']=='success':
            path=np.asarray(query['path']);ax.plot(path[:,0],path[:,1],color='#355c9a',lw=1.7,label='Offline goal path (7200 cap)')
    body=body_corners((state['x'],state['y'],state['yaw']),config.vehicle)
    ax.fill(body[:,0],body[:,1],facecolor='#c76528',edgecolor='#9c3c0b',alpha=.3,label='Body before first wait')
    ax.scatter(*scene['goal'][:2],marker='*',s=100,color='black',label='Parking goal',zorder=4)
    ax.set(xlabel='Local x (m)',ylabel='Local y (m)',title='Case3: available alternatives after selected searches fail')
    ax.set_aspect('equal');ax.legend(fontsize=7,loc='lower left')
    queries=report['direct_goal_queries'];x=np.arange(len(queries))
    bars.bar(x,[q['expanded_nodes'] for q in queries],color=['#a25058','#355c9a'])
    bars.set_xticks(x,[f"cap {q['node_budget']}\n{q['status']}" for q in queries])
    bars.set(ylabel='Expanded nodes',title='Same measured state and goal')
    for i,q in enumerate(queries):bars.text(i,q['expanded_nodes']+35,str(q['expanded_nodes']),ha='center')
    bars.set_ylim(0,max(q['expanded_nodes'] for q in queries)*1.2);bars.spines[['top','right']].set_visible(False)
    fig.suptitle('Interrupted B3 development run: selected prefix verified; blue alternative was not executed',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);plt.close(fig)
    result={'scope':report['scope'],'diagnostic_sha256':sha(diagnostic),'figure_sha256':sha(output),
        'plot_script_sha256':sha(__file__),'new_controls':0,'optimizer_updates':0}
    with receipt.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--diagnostic',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();plot(a.batch,a.diagnostic,a.output)

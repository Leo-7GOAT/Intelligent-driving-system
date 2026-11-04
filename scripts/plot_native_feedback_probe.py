"""Rebuild all four measured trajectories of the rejected feedback-fit probe."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha


def plot(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError('Preserve prior figure')
    names=['20261101T084713744547Z_native_sync_delivery_transfer','20261101T094244452999Z_native_feedback_contrast_probe']
    datasets=[];proof=[];scenes={}
    for name in names:
        folder=root/'runs'/name/'B1';audit=read(folder/'endpoint_raw_audit.json')
        path=folder/'evaluation_trace.jsonl.gz'
        if sha(path)!=audit['evaluation_trace_sha256']:raise ValueError('Raw-bound trajectory differs')
        with gzip.open(path,'rt',encoding='utf-8') as stream:rows=[json.loads(s) for s in stream]
        if {r['case_id'] for r in rows}!=set(range(4)):raise ValueError('All4cases required')
        datasets.append((rows,read(folder/'evaluation.json')))
        proof.append({'batch':name,'trace_sha256':sha(path),'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json')})
        if not scenes:
            for path in folder.glob('scene_*.json'):
                scene=read(path)['scene'];scenes[scene['condition']]=scene
    fig,axes=plt.subplots(2,2,figsize=(11,8.5),constrained_layout=True)
    colors=['#6b7280','#7c3aed'];labels=['Frozen sync control','Pose-response fit']
    for case,ax in enumerate(axes.flat):
        scene=scenes[case];road=np.asarray(scene['road_grid']['drivable'])
        ax.imshow(road,origin='lower',extent=[0,scene['width'],0,scene['height']],cmap='Greys_r',vmin=0,vmax=1,alpha=.16,interpolation='nearest')
        outcomes=[]
        for index,(rows,evaluation) in enumerate(datasets):
            selected=[r for r in rows if r['case_id']==case];initial=selected[0]['info']['initial_scenario']
            ticks=[t for r in selected for t in r['info']['trace']]
            xy=np.asarray([[initial['initial']['x'],initial['initial']['y']]]+[[t['state']['x'],t['state']['y']] for t in ticks])
            ax.plot(xy[:,0],xy[:,1],color=colors[index],linestyle='--' if index==0 else '-',lw=1.7,label=labels[index])
            ax.scatter(*xy[-1],marker='x',s=48,color=colors[index],zorder=5)
            outcome=next(e for e in evaluation['episodes'] if e['case_id']==case)
            outcomes.append(('Control' if index==0 else 'Fit')+': '+outcome['termination_reason'].replace('out_of_bounds','bounds')+' ('+str(len(ticks))+')')
        goal=initial['goal'];ax.add_patch(plt.Circle(goal[:2],.8,fill=False,color='#15803d',lw=1.5))
        ax.scatter(*goal[:2],marker='+',color='#15803d',s=45,zorder=6)
        ax.scatter(initial['initial']['x'],initial['initial']['y'],s=22,color='#111827',zorder=6)
        ax.set(title='TRAIN road '+str(case)+'\n'+'; '.join(outcomes),xlabel='x (m)',ylabel='y (m)',aspect='equal')
        ax.grid(alpha=.2);ax.set_xlim(0,scene['width']);ax.set_ylim(0,max(scene['height'],float(xy[:,1].max())+1))
    axes.flat[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Measured CARLA trajectories: local response fit is NOT qualified\nSame 4 known TRAIN roads and synchronous low level; no new PPO in this comparison',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160,metadata={'Software':'Matplotlib'});plt.close(fig)
    metadata={'scope':__doc__,'sources':proof,'script_sha256':sha(Path(__file__)),'figure_sha256':sha(output),
        'successes':[3,2],'bounds':[1,2],'limits':'Known TRAINconditions,single seed,extra counterfactual supervision. Neither arm passes the zero-unsafe basic gate. Native measurements,not simulated rollout predictions.'}
    with output.with_suffix('.json').open('x',encoding='utf-8') as stream:json.dump(metadata,stream,indent=2)
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(ROOT));p.add_argument('--output',required=True);a=p.parse_args();plot(a.root,a.output)

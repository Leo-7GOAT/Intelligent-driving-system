"""Freeze and plot an append-only native training log prefix, not evaluation.

Replay consumes saved snapshot files only. No CARLA or learned model is loaded.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def capture(batch,method,snapshot):
    batch,snapshot=Path(batch),Path(snapshot)
    protocol=read(batch/'protocol.json')
    if method not in protocol['methods']:raise ValueError('Unregistered training method')
    if snapshot.exists():raise FileExistsError('Preserve training snapshot')
    payloads={'protocol.json':(batch/'protocol.json').read_bytes(),
              'config.json':(batch/method/'config.json').read_bytes(),
              'plot_source.py':Path(__file__).read_bytes()}
    for name in ('episodes.jsonl','optimization.jsonl'):
        data=(batch/method/name).read_bytes()
        # An append in progress may expose an incomplete last line. Retain only
        # complete records; the still-running writer and original stay untouched.
        prefix=data[:data.rfind(b'\n')+1]
        if not prefix:raise ValueError('No complete training records: '+name)
        for line in prefix.splitlines():json.loads(line)
        payloads[name]=prefix
    snapshot.mkdir(parents=True)
    for name,data in payloads.items():(snapshot/name).write_bytes(data)
    metadata=dict(scope='Immutable completed-line training-log snapshot; stochastic training episodes, no evaluation score or completed raw-frame audit',
        batch=batch.name,method=method,seed=protocol['seed'],
        files={name:dict(sha256=digest(data),bytes=len(data)) for name,data in payloads.items()})
    (snapshot/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return snapshot


def render(snapshot,output):
    snapshot,output=Path(snapshot),Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError('Preserve training-prefix figure')
    metadata=read(snapshot/'manifest.json')
    for name,row in metadata['files'].items():
        data=(snapshot/name).read_bytes()
        if len(data)!=row['bytes'] or digest(data)!=row['sha256']:raise ValueError('Training snapshot hash mismatch: '+name)
    episodes=[json.loads(line) for line in (snapshot/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    updates=[json.loads(line) for line in (snapshot/'optimization.jsonl').read_text(encoding='utf-8').splitlines()]
    ids=[r['episode_id'] for r in episodes]
    steps=[r['physics_steps'] for r in updates]
    if len(ids)!=len(set(ids)) or any(b<=a for a,b in zip(steps,steps[1:])):
        raise ValueError('Duplicate episodes or nonmonotonic optimization records')
    config=read(snapshot/'config.json')['env']
    flat=config['method']=='B1'
    figure,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    colors=['#17826d' if r['success'] else '#ba4638' for r in episodes]
    x=[r['at_physics_steps'] for r in episodes]
    for ax,key,title in ((axes[0,0],'reward','Logged episode reward'),(axes[0,1],'progress','Episode task progress (m)')):
        ax.scatter(x,[r[key] for r in episodes],c=colors,s=33,alpha=.9)
        ax.set_title(title);ax.axhline(0,color='.65',lw=.7)
        ax.set_xlabel('Training physical controls at episode end');ax.grid(alpha=.2)
    axes[0,0].scatter([],[],c='#17826d',label='Training success')
    axes[0,0].scatter([],[],c='#ba4638',label='Other training termination')
    axes[0,0].legend(loc='best',fontsize=8)
    axes[1,0].plot(steps,[r['actor_l2_change'] for r in updates],color='#355e9a')
    axes[1,0].set_title('Actor L2 distance from this run\'s initial weights')
    entropy=[-r['metrics']['train/entropy_loss'] for r in updates]
    values=entropy if flat else [math.log(31)-value for value in entropy]
    axes[1,1].plot(steps,values,color='#8150a2')
    axes[1,1].set_title('Gaussian differential entropy' if flat else 'Categorical entropy deficit: ln(31) minus H')
    if not flat:
        axes[1,1].axhline(0,color='.65',lw=.7)
        axes[1,1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for ax in axes[1]:
        ax.set_xlabel('Training physical controls at PPO optimization');ax.grid(alpha=.2)
    figure.suptitle(metadata['method']+' seed '+str(metadata['seed'])+' — STOCHASTIC TRAINING LOG PREFIX\n'
        +'Four known development roads; raw-frame endpoint audit pending',fontsize=13)
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=160,metadata={'Software':'navstack training-prefix diagnostic'})
    plt.close(figure)
    result=dict(scope=metadata['scope'],batch=metadata['batch'],method=metadata['method'],seed=metadata['seed'],
        snapshot_manifest_sha256=digest((snapshot/'manifest.json').read_bytes()),
        episodes=len(episodes),training_terminations=dict(Counter(r['termination_reason'] for r in episodes)),
        optimization_phases=len(updates),last_optimization_controls=steps[-1],
        last_logged_actor_change=updates[-1]['actor_l2_change'],
        last_logged_policy_entropy=entropy[-1],maximum_categorical_entropy=None if flat else math.log(31),
        figure_sha256=digest(output.read_bytes()),new_controls=0,optimizer_updates=0)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--capture-batch');choice.add_argument('--snapshot');parser.add_argument('--method')
    parser.add_argument('--output',required=True);args=parser.parse_args();out=Path(args.output)
    snap=Path(args.snapshot) if args.snapshot else capture(args.capture_batch,args.method,out.with_name(out.stem+'_inputs'))
    render(snap,out)

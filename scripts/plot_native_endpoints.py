"""Plot every registered frozen native-road probe from raw-audited CARLA states."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
from navstack.road_grid import RoadGrid
from navstack.native_cases import registered_native_cases
from navstack.world import body_corners
from scripts.audit_native_policy_inference import recorded_scenes


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def render(batch,output,methods=None):
    batch,output=Path(batch),Path(output)
    manifest_path=output.with_suffix('.json')
    if output.exists() or manifest_path.exists():
        raise FileExistsError('Preserve previous endpoint figure')
    protocol=read(batch/'protocol.json')
    case_inventory=None
    if protocol.get('native_case_manifest') is not None:
        registered_native_cases(protocol)
        case_inventory=(protocol['native_validation']['manifest'] if protocol.get('native_validation')
                        else protocol['native_case_manifest'])['cases']
    registered=protocol['methods']
    methods=registered if methods is None else methods
    if not methods or len(set(methods))!=len(methods) or any(m not in registered for m in methods):
        raise ValueError('Requested methods must be unique registered arms')
    partial=set(methods)!=set(registered)
    records=[]
    for method in methods:
        folder=batch/method
        audit=read(folder/'endpoint_raw_audit.json')
        for name,key in [('evaluation.json','evaluation_sha256'),('evaluation_trace.jsonl.gz','evaluation_trace_sha256'),
                         ('config.json','config_sha256')]:
            if sha(folder/name)!=audit[key]: raise ValueError('Raw-audited figure input changed')
        config=read(folder/'config.json')['env']
        evaluation=read(folder/'evaluation.json')
        scenes={digest:RoadGrid(scene['road_grid'])
                for digest,scene in recorded_scenes(batch,folder).items()}
        cases={}
        with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);info=row['info']
                case=cases.setdefault(row['case_id'],{'initial':info['initial_scenario'],'states':[]})
                case['states'].extend(tick['state'] for tick in info['trace'])
        if sorted(cases)!=sorted(e['case_id'] for e in evaluation['episodes']):
            raise ValueError('Figure must contain every evaluated case')
        for episode in evaluation['episodes']:
            case=cases[episode['case_id']]
            if len(case['states'])!=episode['physics_steps']:
                raise ValueError('Figure control count differs from audited endpoint')
            case['episode']=episode
        records.append((method,config,scenes,cases,audit))
    columns=max(len(row[3]) for row in records)
    if any(len(row[3])!=columns for row in records):
        raise ValueError('Registered methods must share the complete probe set')
    fig,axes=plt.subplots(len(records),columns,figsize=(4*columns,3.5*len(records)),squeeze=False)
    evidence={}
    for r,(method,config,scenes,cases,audit) in enumerate(records):
        evidence[method]={'raw_audit_sha256':sha(batch/method/'endpoint_raw_audit.json'),
            'evaluation_trace_sha256':audit['evaluation_trace_sha256'],'cases':[]}
        for c,(case_id,case) in enumerate(sorted(cases.items())):
            ax=axes[r,c];initial=case['initial'];episode=case['episode']
            road=scenes[initial['road_sha256']]
            states=case['states'];goal=initial['goal'];last=states[-1]
            ax.imshow(~getattr(road,'collision_drivable',road.drivable),origin='lower',extent=(0,road.width,0,road.height),cmap='Greys',vmin=0,vmax=1,alpha=.20)
            ax.plot([initial['initial']['x']]+[s['x'] for s in states],
                    [initial['initial']['y']]+[s['y'] for s in states],color='#2266aa',lw=1.5,label='Measured trajectory')
            ax.scatter(initial['initial']['x'],initial['initial']['y'],marker='o',s=18,color='#2266aa',label='Start')
            ax.add_patch(Circle(goal[:2],config['goal_tolerance'],fill=False,color='#16743a',lw=1.2,label='Goal position tolerance'))
            ax.arrow(goal[0],goal[1],2*math.cos(goal[2]),2*math.sin(goal[2]),width=.09,color='#16743a',length_includes_head=True)
            ax.scatter(last['x'],last['y'],marker='x',s=35,color='#b3261e',label='Terminal state')
            footprint=body_corners((last['x'],last['y'],last['yaw']),SimpleNamespace(**config['vehicle']))
            ax.add_patch(Polygon(footprint,facecolor='#b3261e',edgecolor='#b3261e',alpha=.20,lw=1))
            ax.arrow(last['x'],last['y'],2*math.cos(last['yaw']),2*math.sin(last['yaw']),width=.07,color='#b3261e',length_includes_head=True)
            ax.set(xlim=(0,road.width),ylim=(0,road.height),aspect='equal',xlabel='Local x (m)',ylabel='Local y (m)')
            case_label=(f"case {case_id} / road {case_inventory[case_id]['route_index']}" if case_inventory is not None else f"road {case_id}")
            ax.set_title(f"{method} | {case_label} | {episode['termination_reason']}\n"
                         f"progress {episode['progress']:.2f} m | intervention {100*episode['intervention_rate']:.1f}%",fontsize=9)
            evidence[method]['cases'].append({'case_id':case_id,'controls':len(states),
                'road_sha256':road.sha256,'termination_reason':episode['termination_reason'],
                'success':episode['success'],'final_state':last,'goal':goal})
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9,frameon=False)
    if case_inventory is not None:
        roads=len({case['route_index'] for case in case_inventory})
        arm_scope=f"PARTIAL: {len(methods)}/{len(registered)} arms" if partial else 'all outcomes retained'
        road_scope='geometry-separated validation roads' if protocol.get('native_validation') else 'known roads'
        fig.suptitle(f"Seed {protocol['seed']} | frozen probes on {columns} DEVELOPMENT conditions / {roads} {road_scope} | {arm_scope}",fontsize=13)
    elif partial:
        fig.suptitle(f"Seed {protocol['seed']} | all {columns} TRAINING roads | PARTIAL: {len(methods)}/{len(registered)} registered arms\nAll outcomes retained within each displayed arm",fontsize=13)
    else:
        # Preserve the published complete-inventory figure's exact rendering.
        fig.suptitle(f"Seed {protocol['seed']} | frozen probes on all {columns} TRAINING roads | all outcomes retained",fontsize=13)
    fig.tight_layout(rect=(0,.06,1,.95))
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160,metadata={'Software':'navstack audited endpoint plot'})
    plt.close(fig)
    manifest={'batch':batch.name,'scope':'Audited measured trajectories on training roads, not held-out research confirmation. Arrows show goal/terminal yaw; gray cells are sampled non-driving road cells.',
        'protocol_sha256':sha(batch/'protocol.json'),'script_sha256':sha(Path(__file__)),
        'figure_sha256':sha(output),'methods':evidence,'new_controls':0,
        'complete_registered_arm_inventory':not partial,
        'omitted_registered_arms':[m for m in registered if m not in methods]}
    if protocol.get('native_validation'):
        manifest['scope']='Audited measured trajectories on geometry-separated DEVELOPMENT validation roads,not final research confirmation;all outcomes retained.'
    with manifest_path.open('x',encoding='utf-8') as stream: json.dump(manifest,stream,indent=2)
    print(json.dumps({'figure':str(output),'methods':methods,'cases_per_method':columns,'new_controls':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--method',action='append',help='Explicit partial arm selection; every case in each arm is retained')
    args=parser.parse_args()
    render(args.batch,args.output,args.method)

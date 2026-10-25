"""Descriptive four-road engineering plot; no independent-training inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source,output):
    source,output=Path(source),Path(output)
    receipt=output.with_suffix('.json')
    if output.exists() or receipt.exists():raise FileExistsError('Preserve figure evidence')
    report=json.loads(source.read_text(encoding='utf-8'))
    if len(report['cases'])!=4 or not report['matched_measured_starts'] or not report['matched_rule_planner_trajectory_mpc']:
        raise ValueError('Requires the completed four-road matched comparison')
    colors={'legacy':'#9b5656','feedforward':'#347786'}
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    x=np.arange(4)
    for ax,key,label in zip(axes,('progress','speed_errors_rms','jerk_rms'),
        ('Progress (m)','Speed error RMS (m/s)','Jerk RMS (m/s³)')):
        for offset,(mode,color) in zip((-.19,.19),colors.items()):
            values=[r[mode][key] for r in report['cases']]
            bars=ax.bar(x+offset,values,.36,label=mode,color=color)
            if key=='progress':
                for bar,row in zip(bars,report['cases']):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.3,
                        'S' if row[mode]['success'] else 'T' if row[mode]['termination_reason']=='deadline' else 'F',
                        ha='center',fontsize=9)
        ax.set_xticks(x,[str(r['case_id']) for r in report['cases']]);ax.set_xlabel('Known development road')
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylim(0,max(r[m]['progress'] for r in report['cases'] for m in colors)*1.15)
    axes[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('B3 actuator comparison: four paired training roads, one execution per adapter\nS = success; T = deadline; descriptive control evidence, no PPO claim',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160);plt.close(fig)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result={'scope':report['scope'],'comparison_sha256':digest(source),'figure_sha256':digest(output),
        'plot_script_sha256':digest(__file__),'new_controls':0,'optimizer_updates':0}
    with receipt.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--comparison',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plot(a.comparison,a.output)

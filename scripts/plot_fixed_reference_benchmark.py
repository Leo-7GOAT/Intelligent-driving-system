"""Scientific control-only figure from all paired lightweight episodes."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot(batch,output):
    rows=json.loads((Path(batch)/'summary.json').read_text(encoding='utf-8'))['episodes']
    if len(rows)!=80:raise ValueError('Keep all80controller episodes')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    groups=[('nominal','mpc20'),('nominal','pursuit_pi'),('lag_delay_scale','mpc20'),('lag_delay_scale','pursuit_pi')]
    colors=['#2563eb','#d97706']*2
    values=[lambda r:r['tracking_rms']['lateral'],lambda r:r['jerk_rms'],lambda r:1000*r['control_timing']['p95']]
    for ax,measure,title in zip(axes,values,['Lateral tracking RMSE (m)','Acceleration jerk RMS (m/s³)','Control computation P95 (ms)']):
        data=[[measure(r) for r in rows if (r['plant_variant'],r['controller'])==group] for group in groups]
        for x,(samples,color) in enumerate(zip(data,colors)):
            ax.bar(x,np.mean(samples),color=color,alpha=.75,width=.7)
            ax.scatter(x+np.linspace(-.13,.13,len(samples)),samples,color=color,edgecolors='white',linewidth=.3,s=15,zorder=3)
        ax.set(title=title,xticks=range(4),xticklabels=['MPC\nnominal','Pursuit/PI\nnominal','MPC\nmismatch','Pursuit/PI\nmismatch'])
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[2].set_yscale('log')
    fig.suptitle('Fixed-reference controller comparison — lightweight physical integrator',fontsize=13)
    fig.supxlabel('20 episodes per group; both controllers complete 40/40. Points are route × initial-error conditions.\nComputation excludes CARLA/RPC and planning; this is not PPO or CARLA performance.',fontsize=9)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(output)+'.png',dpi=180);fig.savefig(str(output)+'.svg');plt.close(fig)
    svg=Path(str(output)+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    print(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args();plot(args.batch,args.output_prefix)

"""Plot all four RPC-audited native controller pairs, including braking spikes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(baseline, candidate, output):
    results = [json.loads((Path(p)/'analysis/verified_summary.json').read_text(encoding='utf-8'))
               for p in (baseline, candidate)]
    if any(len(r['episodes']) != 4 or r['bridge_frame_binding']['actual_control_frames_bound'] != r['total_control_frames'] for r in results):
        raise ValueError('Require all four RPC-audited cases in each arm')
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout='constrained')
    measures = [lambda r: r['tracking_rms']['lateral'], lambda r: r['tracking_rms']['speed'], lambda r: r['jerk_rms_m_s3']]
    titles = ['Lateral RMSE (m)', 'Speed RMSE (m/s)', 'Measured acceleration jerk RMS (m/s³)']
    for ax, measure, title in zip(axes, measures, titles):
        for case in range(4):
            ax.plot([0, 1], [measure(r['episodes'][case]) for r in results], '-o', alpha=.7, label=f'Road{case}')
        ax.set(title=title, xticks=[0, 1], xticklabels=['MPC20', 'Pursuit/PI'])
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Actual CARLA fixed-reference controller comparison: four known roads')
    fig.supxlabel('Both4/4;every frame RPC-bound. One execution per controller/road;not independent PPO seeds.\nAll braking spikes retained;CPU-integrator smoothness advantage does not transfer.', fontsize=9)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous figure')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'candidate', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    plot(args.baseline, args.candidate, args.output)

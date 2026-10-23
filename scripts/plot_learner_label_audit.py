"""Plot recorded learner paths and queried controls on two fixed training layouts."""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from navstack.config import EnvConfig
from navstack.env import NavigationEnv


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', required=True)
    parser.add_argument('--relabel', required=True)
    args = parser.parse_args()
    audit, relabel = Path(args.audit), Path(args.relabel)
    assert (audit/'verification.json').exists()
    protocol = json.loads((audit/'protocol.json').read_text(encoding='utf-8'))
    cases = json.loads((relabel/'input/original_protocol.json').read_text(encoding='utf-8'))['cases']
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), constrained_layout=True)
    for index in (0, 1):
        case = protocol['cases'][index]
        assert case['split'] == 'train'
        source_id = next(i for i, item in enumerate(cases) if (item['seed'], item['scenario']) == (case['seed'], case['scenario']))
        learner = rows(audit/f'case_{index:03d}.jsonl.gz')
        original = rows(relabel/'original'/f'case_{source_id:03d}.jsonl.gz')
        cfg = EnvConfig.from_dict(protocol['env'])
        cfg.scenario, cfg.split = case['scenario'], case['split']
        env = NavigationEnv(cfg)
        try:
            env.reset(seed=case['seed'])  # Geometry only; no simulated controls in this plotting script.
            ax = axes[index, 0]
            for x0, y0, x1, y1 in env.world.static:
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, color='gray', alpha=.5))
            for data, name, color in [(original, 'Recorded teacher execution', 'tab:blue'),
                                      (learner, 'Pure learner execution', 'tab:red')]:
                states = [row['info']['trace'][0]['state'] for row in data]
                ax.plot([s['x'] for s in states], [s['y'] for s in states], label=name, color=color)
                ax.scatter(states[-1]['x'], states[-1]['y'], color=color, marker='x')
            ax.set(xlabel='x (m)', ylabel='y (m)', xlim=(0, cfg.width), ylim=(4, 20),
                   title=f"Static seed {case['seed']}: learner collision")
            ax.set_aspect('equal')
            if index == 0:
                ax.legend(fontsize=7)
        finally:
            env.close()
        times = np.arange(len(learner))*cfg.control_dt
        for component, label in [(0, 'Longitudinal'), (1, 'Steering')]:
            ax = axes[index, component+1]
            ax.plot(times, [row['learner_action'][component] for row in learner], label='Executed learner request', color='tab:red')
            ax.plot(times, [row['teacher_action'][component] for row in learner], label='Query at same actual state', color='tab:green', linestyle='--')
            ax.set(xlabel='Simulation time (s)', ylabel='Normalized requested action', title=label)
            ax.grid(alpha=.2)
            if index == 0:
                ax.legend(fontsize=7)
    fig.suptitle('One fixed actor-only model: teacher queries never execute; no policy learning in this audit')
    fig.savefig(audit/'learner_distribution_shift.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

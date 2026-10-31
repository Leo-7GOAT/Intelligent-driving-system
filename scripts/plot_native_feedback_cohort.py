"""Rebuild the all-five-seed DEVELOPMENT feedback comparison figure."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.analyze_native_feedback_cohort import paired_seed_statistics
from scripts.audit_native_method_endpoint import read, sha


def plot(report, output):
    report, output = Path(report), Path(output)
    metadata = output.with_suffix('.json')
    if output.exists() or metadata.exists():
        raise FileExistsError('Use fresh figure outputs')
    cohort = read(report)
    rows = cohort['results']
    seeds = [r['seed'] for r in rows]
    if len(seeds) < 5 or seeds != cohort['seeds'] or len(set(seeds)) != len(seeds):
        raise ValueError('Require every independent training seed')
    effects = [r['paired_difference_percentage_points']/100 for r in rows]
    np.testing.assert_allclose(effects,
        [(r['H_successes']-r['H0_successes'])/r['cases'] for r in rows], rtol=0, atol=1e-14)
    statistics = paired_seed_statistics(effects)
    if statistics != cohort['statistics']:
        raise ValueError('Saved statistics do not reconstruct from all seed outcomes')
    x = np.arange(len(rows))
    fig, (scores, differences) = plt.subplots(1, 2, figsize=(10.5, 4.3), layout='constrained')
    for offset, method, color in ((-.21, 'H', '#176b9c'), (.21, 'H0', '#93989e')):
        values = [100*r[method+'_successes']/r['cases'] for r in rows]
        bars = scores.bar(x+offset, values, .36, label=method, color=color)
        scores.bar_label(bars, labels=[f"{r[method+'_successes']}/{r['cases']}" for r in rows], fontsize=7, padding=3)
    scores.set(xticks=x, xticklabels=seeds, xlabel='Independent training seed',
               ylabel='Safe task completion (%)', ylim=(0, 112), title='All canonical seed pairs')
    scores.legend(frameon=False, loc='lower right')
    differences.axvline(0, color='#737373', linewidth=1)
    differences.scatter(np.array(effects)*100, x, color='#176b9c', s=45)
    mean = statistics['paired_mean_difference']*100
    lower, upper = np.array(statistics['paired_95pct_t_interval'])*100
    differences.errorbar(mean, len(rows)+.35, xerr=[[mean-lower], [upper-mean]],
                        fmt='D', color='#b14a36', capsize=5, label='Mean and 95% t interval')
    differences.set(yticks=[*x, len(rows)+.35], yticklabels=[*[str(s) for s in seeds], 'Mean'],
                    xlabel='H minus H0 (percentage points)', title='Paired differences across seeds', xlim=(-25, 25))
    differences.invert_yaxis()
    differences.set_title(f'Paired differences across seeds\nMean {mean:+.2f} pp; 95% interval [{lower:.2f}, {upper:.2f}] pp', fontsize=10)
    fig.suptitle('Feedback ablation: 12 reused development conditions, five trained seed pairs\nExploratory result; stable benefit is not established', fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, metadata={'Software': 'navstack all-seed development figure'})
    plt.close(fig)
    metadata.write_text(json.dumps({'scope': __doc__, 'source_report_sha256': sha(report),
        'script_sha256': sha(Path(__file__)), 'figure_sha256': sha(output),
        'seeds': seeds, 'statistics': statistics, 'new_controls': 0,
        'optimizer_updates': 0, 'research_acceptance': 'UNMET'}, indent=2), encoding='utf-8')
    print(json.dumps({'figure_sha256': sha(output), 'seeds': seeds}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plot(args.report, args.output)

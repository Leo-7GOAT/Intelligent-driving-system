"""Rebuild the five-seed DEVELOPMENT planner-ablation figure."""
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
    if output.exists() or metadata.exists(): raise FileExistsError('Fresh figure outputs required')
    cohort = read(report); rows = cohort['results']; seeds = [r['seed'] for r in rows]
    if seeds != [11, 22, 33, 44, 55]: raise ValueError('All five trained seeds required')
    effects = [(r['successes']['H'] - r['successes']['B2']) / r['cases'] for r in rows]
    statistics = paired_seed_statistics(effects)
    if statistics != cohort['statistics']: raise ValueError('Saved statistics differ from outcomes')
    x = np.arange(5)
    fig, (scores, differences) = plt.subplots(1, 2, figsize=(10.5, 4.6), layout='constrained')
    for offset, method, color in ((-.21, 'H', '#176b9c'), (.21, 'B2', '#93989e')):
        bars = scores.bar(x+offset, [100*r['successes'][method]/r['cases'] for r in rows], .36, label=method, color=color)
        scores.bar_label(bars, labels=[f"{r['successes'][method]}/{r['cases']}" for r in rows], fontsize=8, padding=3)
    scores.set(xticks=x, xticklabels=seeds, xlabel='Independent training seed',
               ylabel='Safe task completion (%)', ylim=(0, 112), title='Hybrid A* (H) vs body-spline Grid A* (B2)')
    scores.legend(frameon=False, loc='lower right')
    differences.axvline(0, color='#737373', linewidth=1)
    differences.scatter(np.array(effects)*100, x, color='#176b9c', s=45)
    mean = statistics['paired_mean_difference']*100
    lower, upper = np.array(statistics['paired_95pct_t_interval'])*100
    differences.errorbar(mean, 5.35, xerr=[[mean-lower], [upper-mean]], fmt='D', color='#b14a36', capsize=5)
    differences.set(yticks=[*x, 5.35], yticklabels=[*[str(s) for s in seeds], 'Mean'],
                    xlabel='H minus B2 (percentage points)', xlim=(-5, 50))
    differences.invert_yaxis()
    p_value = statistics['exploratory_exact_sign_flip_p_two_sided']
    differences.set_title(f'Mean {mean:+.2f} pp; 95% t interval [{lower:.2f}, {upper:.2f}]\nExact sign-flip p = {p_value:g}; retrospective analysis', fontsize=10)
    fig.suptitle('Planner-stack comparison: 12 reused DEVELOPMENT conditions, five trained seed pairs\nWalker spawn initialization differs; no isolated causal or final-confirmation claim', fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, metadata={'Software': 'navstack development planner figure'}); plt.close(fig)
    metadata.write_text(json.dumps({'scope': __doc__, 'source_report_sha256': sha(report),
        'script_sha256': sha(Path(__file__)), 'figure_sha256': sha(output), 'statistics': statistics,
        'new_controls': 0, 'optimizer_updates': 0, 'research_acceptance': 'UNMET'}, indent=2), encoding='utf-8')
    print(json.dumps({'figure_sha256': sha(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); plot(args.report, args.output)

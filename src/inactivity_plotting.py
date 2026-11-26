"""Auditable static Task 6 figures from the verified aggregate report."""

from contextlib import nullcontext
import os
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_artifacts import digest, file_hash, read_json, write_json
from src.experiments import Limits, experiment_lock, _remove_stage
from src.inactivity import ARMS
from src.inactivity_reporting import validate_report, guard


def render_study(root, *, limits=None, locked=False):
    from src.inactivity_study import analyze_study
    root, limits = Path(root).resolve(), limits or Limits()
    with nullcontext() if locked else experiment_lock(root):
        analysis = Path(analyze_study(root, limits=limits, locked=True)['directory'])
        summary = validate_report(analysis)
        identity = {'analysis_sha256': file_hash(analysis / 'complete.json'), 'plot_source': file_hash(__file__)}
        destination = root / 'figures' / digest(identity)
        if destination.exists():
            record = read_json(destination / 'complete.json')
            if (record['identity'] != identity or
                    any(file_hash(destination/k) != v for k, v in record['files'].items())):
                raise ValueError('Corrupt rendered figures')
            return {'directory': str(destination), 'reused': True}
        guard(root, limits)
        destination.parent.mkdir(exist_ok=True)
        stage = destination.parent / ('.staging-' + uuid4().hex)
        stage.mkdir()
        with np.load(analysis / 'tables.npz', allow_pickle=False) as z:
            tables = {k: z[k] for k in z.files}
        cells, fields = summary['configurations'], summary['series']
        colors = plt.get_cmap('tab10').colors
        def panels(title):
            fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
            fig.suptitle(title, fontsize=15)
            for cell, ax in zip(cells, axes.flat):
                ax.set_title(cell, fontsize=10)
                ax.tick_params(labelsize=8)
                ax.grid(alpha=.2)
            return fig, axes.flat
        def save(fig, name):
            guard(root, limits)
            fig.savefig(stage / f'{name}.png', dpi=140)
            fig.savefig(stage / f'{name}.pdf')
            plt.close(fig)
        try:
            fig, axes = panels('Time to first confirmed stall — Kaplan–Meier, pointwise 95% intervals')
            for cell, ax in zip(cells, axes):
                curve = summary['references'][cell]['thresholds'][str(summary['threshold'])]['time_to_confirmation']
                x, y = [0]+curve['time'], [1]+curve['survival']
                ax.step(x, y, where='post')
                low = np.array([1]+[np.nan if v is None else v for v in curve['low']])
                high = np.array([1]+[np.nan if v is None else v for v in curve['high']])
                ax.fill_between(x, low, high, step='post', alpha=.2)
                ax.set(ylim=(-.03, 1.03), xlabel='Trading day', ylabel='No confirmed stall yet')
                ax.text(.03, .07, f"{curve['events']}/{curve['n']} confirmed", transform=ax.transAxes, fontsize=8)
            save(fig, 'stall_onset')

            fig, axes = panels('Trading activity — means in consecutive 100-day windows; all reference seeds')
            for i, ax in enumerate(axes):
                a = tables[f'cell{i}_daily_mean']
                for field, label in (('execution_fraction', 'Any execution'), ('distinct_exchange_fraction', 'Distinct traders')):
                    xs, ys = [], []
                    for start in range(0, len(a), 100):
                        end = min(len(a), start+100)
                        xs.append((start+1+end)/2)
                        ys.append(np.mean(a[start:end, fields.index(field)]))
                    ax.plot(xs, ys, label=label)
                ax.set(xlabel='Trading day', ylabel='Fraction of active days', ylim=(-.03, 1.03))
            fig.legend(*fig.axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
            save(fig, 'activity')

            for name, title, names in (
                ('resource_paths', 'Resources around first stall onset — solid cash, dashed asset shares',
                 [('cash_share_'+g, g+' cash', colors[k], '-') for k, g in enumerate(('noise', 'chartist', 'fundamentalist'))]+
                 [('asset_share_'+g, g+' shares', colors[k], '--') for k, g in enumerate(('noise', 'chartist', 'fundamentalist'))]),
                ('decision_paths', 'Actual decision reasons around first stall onset — fractions of each strategy',
                 [('chartist_buy_order_fraction', 'Chartist buy order', colors[0], '-'),
                  ('fundamentalist_below_threshold_fraction', 'Fundamentalist below threshold', colors[1], '-'),
                  ('noise_no_holdings_fraction', 'Noise sell blocked by holdings', colors[2], '-')]),
                ('book_paths', 'Outstanding book after matching, before expiry — event-aligned means',
                 [('bid_count', 'Bids', colors[0], '-'), ('ask_count', 'Asks', colors[1], '-')])):
                fig, axes = panels(title)
                for i, ax in enumerate(axes):
                    counts = tables[f'cell{i}_event_count']
                    a = tables[f'cell{i}_event_mean']
                    for field, label, color, style in names:
                        ax.plot(tables['event_day'], np.where(counts > 0, a[:, fields.index(field)], np.nan),
                                label=label, color=color, linestyle=style)
                    ax.axvline(0, color='black', alpha=.4, linewidth=.8)
                    ax.set(xlabel='Days from onset', ylabel='Orders' if name == 'book_paths' else 'Fraction')
                    if name != 'book_paths':
                        ax.set_ylim(-.03, 1.03)
                    if not np.any(counts):
                        ax.text(.25, .5, 'No confirmed stalls', transform=ax.transAxes)
                    count_axis = ax.twinx()
                    count_axis.plot(tables['event_day'], counts, color='grey', alpha=.3, linewidth=.8)
                    count_axis.set_ylim(0, max(1, summary['references'][cells[i]]['n_references']))
                    count_axis.set_ylabel('Contributing runs', fontsize=7, color='grey')
                    count_axis.tick_params(labelsize=7, colors='grey')
                fig.legend(*fig.axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=min(6, len(names)))
                save(fig, name)

            fig, axes = panels('Zero-volume spell duration — descriptive survival; repeated spells are dependent')
            for cell, ax in zip(cells, axes):
                curve = summary['references'][cell]['spell_duration_descriptive']
                if curve['n']:
                    ax.step([0]+curve['time'], [1]+curve['survival'], where='post')
                else:
                    ax.text(.3, .5, 'No zero-volume spells', transform=ax.transAxes)
                ax.set(xscale='symlog', xlabel='Zero-volume days', ylabel='Spell survival', ylim=(-.03, 1.03))
            save(fig, 'spell_durations')

            fig, axes = panels('First distinct-trader exchange after intervention — conditional recovery by arm')
            for i, ax in enumerate(axes):
                count = summary['recovery'][cells[i]]['eligible_checkpoints']
                if count:
                    for k, arm in enumerate(ARMS):
                        ax.plot(np.arange(1, summary['continuation_days']+1), tables[f'cell{i}_distinct_cdf'][k],
                                label=arm, color=colors[k], linewidth=1)
                else:
                    ax.text(.3, .5, 'Not applicable', transform=ax.transAxes)
                ax.set(xlabel='Continuation day', ylabel='Recovered fraction', ylim=(-.03, 1.03))
                ax.text(.03, .07, f'{count} checkpoints', transform=ax.transAxes, fontsize=8)
            # Some first panels may have no eligible checkpoints.
            handles = [plt.Line2D([], [], color=colors[k], label=arm) for k, arm in enumerate(ARMS)]
            fig.legend(handles=handles, loc='outside lower center', ncol=8,
                       title='Bits: clear queues / reset shared memory / redistribute portfolios')
            save(fig, 'recovery')

            fig, axes = panels('Primary recovery factorial effects — checkpoint-paired pointwise 95% intervals')
            labels = ['Queues', 'Memory', 'Portfolios', 'Queues × memory', 'Queues × portfolios', 'Memory × portfolios', 'Three-way']
            for cell, ax in zip(cells, axes):
                data = list(summary['recovery'][cell]['primary']['factorial'].values())
                if data[0]['estimate'] is not None:
                    for j, row in enumerate(data):
                        ax.plot([row['low'], row['high']], [j, j], color=colors[j])
                        ax.plot(row['estimate'], j, 'o' if not row['flags'] else 'x', color=colors[j])
                else:
                    ax.text(.3, .5, 'Not applicable', transform=ax.transAxes)
                ax.axvline(0, color='black', alpha=.3)
                ax.set(yticks=range(7), yticklabels=labels, xlabel='Difference in tail activity', xlim=(-1.05, 1.05))
                ax.invert_yaxis()
                ax.tick_params(axis='y', labelsize=7)
            fig.supxlabel('× marks fewer than 10 checkpoints or a degenerate empirical bootstrap distribution.', fontsize=10)
            save(fig, 'factorial_effects')
            write_json(stage / 'complete.json', {'identity': identity,
                       'files': {p.name: file_hash(p) for p in sorted(stage.iterdir())}})
            os.rename(stage, destination)
            return {'directory': str(destination), 'reused': False}
        finally:
            plt.close('all')
            if stage.exists():
                _remove_stage(stage)

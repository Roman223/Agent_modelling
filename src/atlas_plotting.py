"""Static, exportable parameter maps with explicit seed coverage."""

import gzip
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter
import numpy as np

from src.atlas import METRIC_LABELS, axes, variant_name, variants
from src.atlas_runner import atomic_json, collect
from src.experiment_artifacts import ROOT, file_hash
from src.experiments import GIB, MIB, directory_bytes, experiment_lock


def plot_metric(summary, metric):
    """Return two figures. Heatmaps mask incomplete cells; curves mark partial data."""
    sweep, grid, gamma = axes()
    key = 'frequency' if metric == 'terminal_stall' else 'median'
    lookup = {(c['variant'], c['alpha'], c['gamma']): c for c in summary['cells']}
    panels = [variant_name(v) for v in variants()]
    estimates = [c['metrics'][metric][key] for c in summary['cells']
                 if c['metrics'][metric][key] is not None]
    lo, hi = (min(estimates), max(estimates)) if estimates else (0, 1)
    if lo == hi:
        lo, hi = lo - max(0.01, abs(lo) * 0.05), hi + max(0.01, abs(hi) * 0.05)
    norm = Normalize(lo, hi)
    cmap = plt.colormaps['viridis'].with_extremes(bad='#e5e7eb')
    heat, heat_axes = plt.subplots(2, 4, figsize=(18, 9), layout='constrained', sharex=True, sharey=True)
    curve, curve_axes = plt.subplots(2, 4, figsize=(18, 8), layout='constrained', sharex=True, sharey=True)
    def log_edges(values):
        middle = np.sqrt(np.array(values[:-1]) * values[1:])
        return np.r_[values[0] ** 2 / middle[0], middle, values[-1] ** 2 / middle[-1]]
    for name, ax, line in zip(panels, heat_axes.flat, curve_axes.flat):
        values = np.full((len(gamma), len(grid)), np.nan)
        for i, a in enumerate(grid):
            for j, g in enumerate(gamma):
                cell = lookup.get((name, a, g))
                stat = cell['metrics'][metric] if cell else None
                valid, expected = (stat['valid'], stat['expected']) if stat else (0, 0)
                if stat and valid == expected and valid:
                    values[j, i] = stat[key]
                ax.text(a, g, f'{valid}/{expected}' if expected else '—', fontsize=6,
                        ha='center', va='center', color='black',
                        bbox={'facecolor': 'white', 'alpha': 0.65, 'edgecolor': 'none', 'pad': 0.3})
        mesh = ax.pcolormesh(log_edges(grid), log_edges(gamma), np.ma.masked_invalid(values), shading='flat', cmap=cmap, norm=norm)
        ax.set(xscale='log', yscale='log', title=name.replace('_', '\n'), xlabel='Chartist alpha', ylabel='Fundamentalist gamma')
        ax.set_xticks([0.025, 0.1, 0.4], ['.025', '.1', '.4'])
        ax.set_yticks([0.5, 2, 8], ['.5', '2', '8'])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        y, q25, q75, partial, counts = [], [], [], [], []
        for a in sweep:
            cell = lookup.get((name, a, 2.0))
            stat = cell['metrics'][metric] if cell else None
            y.append(stat[key] if stat and stat[key] is not None else np.nan)
            q25.append(stat['q25'] if stat and stat['q25'] is not None else np.nan)
            q75.append(stat['q75'] if stat and stat['q75'] is not None else np.nan)
            partial.append(bool(stat and stat['valid'] < stat['expected']))
            counts.append(f'{stat["valid"]}/{stat["expected"]}' if stat else '—')
        line.plot(sweep, y, 'o-', markersize=3, color='#16697a')
        if metric != 'terminal_stall':
            line.fill_between(sweep, q25, q75, color='#16697a', alpha=0.2)
        for a, value, count, incomplete in zip(sweep, y, counts, partial):
            line.annotate(count, (a, 0), xycoords=('data', 'axes fraction'), rotation=90,
                          xytext=(0, 4), textcoords='offset points', ha='center', fontsize=6)
            if incomplete and np.isfinite(value):
                line.plot(a, value, 'x', color='#cc3311')
        line.set(xscale='log', xlabel='Chartist alpha (gamma = 2)', title=name.replace('_', '\n'))
        line.set_xticks([0.025, 0.1, 0.4], ['.025', '.1', '.4'])
        line.xaxis.set_minor_formatter(NullFormatter())
        line.margins(y=0.15)
        line.grid(alpha=0.2)
    heat.colorbar(mesh, ax=list(heat_axes.flat), label=METRIC_LABELS[metric], shrink=0.8)
    heat.suptitle(f'{METRIC_LABELS[metric]}\nNumbers: valid/expected seeds; grey: incomplete or undefined')
    band = 'frequency across seeds' if metric == 'terminal_stall' else 'median and interquartile band'
    curve.suptitle(f'{METRIC_LABELS[metric]} — {band}\nCounts at bottom: valid/expected seeds; red ×: partial estimate')
    return heat, curve


def render_atlas(directory, analysis_id=None):
    directory = Path(directory).resolve()
    root = directory.parent
    with experiment_lock(directory):
        summary, records = collect(directory, analysis_id)
        output = root / 'reports' / summary['analysis_id']
        output.mkdir(parents=True, exist_ok=True)
        # Reports have their own reservation within the remaining half GiB.
        def guard():
            if directory_bytes(root) + 64 * MIB > 5 * GIB or shutil.disk_usage(root).free < 10 * GIB + 64 * MIB:
                raise RuntimeError('Atlas output/free-disk limit prevents rendering')
            if directory_bytes(root / 'reports') + 64 * MIB > 256 * MIB:
                raise RuntimeError('Report output reservation exhausted')
        guard()
        atomic_json(output / 'summary.json', summary)
        np.savez_compressed(output / 'tables.npz',
                            alpha=np.array([c['alpha'] for c in summary['cells']]),
                            gamma=np.array([c['gamma'] for c in summary['cells']]),
                            variant=np.array([c['variant'] for c in summary['cells']]),
                            metric=np.array(list(METRIC_LABELS)),
                            median=np.array([[c['metrics'][m]['median'] if c['metrics'][m]['median'] is not None else np.nan
                                              for m in METRIC_LABELS] for c in summary['cells']]),
                            valid=np.array([[c['metrics'][m]['valid'] for m in METRIC_LABELS] for c in summary['cells']]),
                            expected=np.array([c['expected'] for c in summary['cells']]),
                            terminal_stall_frequency=np.array([c['metrics']['terminal_stall']['frequency']
                                                               if c['metrics']['terminal_stall']['frequency'] is not None else np.nan
                                                               for c in summary['cells']]))
        with gzip.open(output / 'per_run.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for run_id, record in records.items():
                stream.write(json.dumps({'run_id': run_id, **record}, allow_nan=False) + '\n')
        for metric in METRIC_LABELS:
            guard()
            heat, curve = plot_metric(summary, metric)
            for label, figure in (('map', heat), ('sweep', curve)):
                figure.savefig(output / f'{metric}_{label}.png', dpi=130)
                figure.savefig(output / f'{metric}_{label}.pdf')
                plt.close(figure)
        atomic_json(output / 'render.json', {
            'analysis_id': summary['analysis_id'], 'plot_source_sha256': file_hash(ROOT / 'src/atlas_plotting.py'),
            'expected_runs': summary['expected_runs'], 'analyzed_runs': summary['analyzed_runs'],
            'missing_analysis': summary['missing_analysis'], 'operational_failures': summary['operational_failures'],
            'files': {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.name != 'render.json'},
        })
    print(json.dumps({'report_directory': str(output), 'analyzed_runs': summary['analyzed_runs'],
                      'operational_failures': summary['operational_failures']}))
    return output

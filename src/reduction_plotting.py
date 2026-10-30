"""Streaming comparison summaries and publication-sized task-4 figures."""

import gzip
import json
import os
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np

from src.atlas_runner import manifest_at
from src.experiment_artifacts import digest, execution_fingerprint, file_hash, read_compact, read_json, write_json
from src.experiments import Limits, directory_bytes, disk_guard, experiment_lock
from src.reduction_study import comparison_summary, read_reduction, validate_input

METRICS = {
    'final_price': 'Final price', 'log_return_volatility': 'Daily volatility',
    'positive_volume_volatility': 'Positive-volume volatility', 'maximum_drawdown': 'Window drawdown',
    'cumulative_log_return': 'Window log return', 'mean_volume': 'Mean volume (shares/day)',
    'zero_volume_fraction': 'Exact no-trade fraction', 'subshare_volume_fraction': 'Sub-share volume fraction',
    'longest_no_trade_streak': 'Longest no-trade streak (days)',
    'terminal_no_trade_streak': 'Terminal no-trade streak (days)', 'terminal_stall': 'Terminal stall frequency',
    'max_outstanding_orders': 'Maximum outstanding parcels/orders',
    'within_strategy_volume_fraction': 'Within-strategy volume fraction',
}
PRIMARY = ('reference-v6', 'reference-v7', 'liquidity-v6', 'liquidity-v7')
LABELS = ('Reference\npersistent', 'Reference\nday', 'Liquidity\npersistent', 'Liquidity\nday')
COLORS = ('#2878A5', '#D47A26', '#45905B', '#9955A0')


def collect_study(root):
    root = Path(root)
    study, manifest = manifest_at(root), manifest_at(root / 'reduction')
    if manifest['study_sha256'] != digest(study):
        raise ValueError('Study/reduction manifest mismatch')
    records, sources = [], {}
    for entry in manifest['inputs']:
        validate_input(entry)
        meta, arrays, _ = read_compact(entry['artifact'])
        record = {'model': 'abm', 'cohort': entry['cohort'], 'cell': entry['cell'],
                  'seed': meta['run']['seed'], 'points': None, 'run_id': meta['run']['run_id'],
                  'complete_sha256': entry['complete_sha256'],
                  'summary': comparison_summary(arrays, study['burn_in'])}
        records.append(record)
        sources[record['run_id']] = entry['artifact']
    for run in manifest['runs']:
        path = root / 'reduction/runs' / run['run_id']
        meta, arrays, summary = read_reduction(path, run, manifest['execution'])
        cell = run['coordinate'] + ('-v6' if run['config']['order_lifetime'] == 'persistent' else '-v7')
        record = {'model': 'reduction', 'cohort': run['cohort'], 'cell': cell,
                  'seed': run['initial_condition_seed'], 'points': run['config']['quadrature_points'],
                  'run_id': run['run_id'], 'complete_sha256': file_hash(path / 'complete.json'), 'summary': summary}
        records.append(record)
        sources[record['run_id']] = str(path)
    return study, manifest, records, sources


def aggregate_records(records):
    groups = {}
    for row in records:
        key = (row['model'], row['cohort'], row['cell'], row['points'])
        groups.setdefault(key, []).append(row)
    output = []
    for (model, cohort, cell, points), rows in groups.items():
        metrics = {}
        for name in METRICS:
            values = [r['summary']['values'].get(name) for r in rows]
            valid = np.array([v for v in values if v is not None], dtype=float)
            metrics[name] = {'valid': len(valid), 'expected': len(rows),
                             'q25': float(np.quantile(valid, .25)) if len(valid) else None,
                             'median': float(np.median(valid)) if len(valid) else None,
                             'q75': float(np.quantile(valid, .75)) if len(valid) else None}
            if name == 'terminal_stall':
                metrics[name].update(count=int(valid.sum()), frequency=float(valid.mean()) if len(valid) else None)
        output.append({'model': model, 'cohort': cohort, 'cell': cell, 'points': points,
                       'seeds': [r['seed'] for r in rows], 'metrics': metrics,
                       'final_cash_median': np.median([r['summary']['final_cash'] for r in rows], axis=0).tolist(),
                       'final_shares_median': np.median([r['summary']['final_shares'] for r in rows], axis=0).tolist()})
    return output


def render_study(root):
    root = Path(root).resolve()
    with experiment_lock(root):
        if disk_guard(root, Limits(), reserved=64*1024**2):
            raise ValueError('Report disk/output quota reached')
        study, manifest, records, sources = collect_study(root)
        runtime = execution_fingerprint()
        identity = {'study_sha256': digest(study), 'reduction_manifest_sha256': digest(manifest),
                    'inputs': {r['run_id']: r['complete_sha256'] for r in records},
                    'source_sha256': {name: value for name, value in runtime['source_sha256'].items()
                                      if name in ('src/reduction_plotting.py', 'src/reduction_study.py', 'src/experiment_artifacts.py')},
                    'python': runtime['python'], 'packages': runtime['packages']}
        destination = root / 'reports' / digest(identity)
        if (destination / 'render.json').exists():
            old = read_json(destination / 'render.json')
            if old['identity'] == identity and all(file_hash(destination / name) == value for name, value in old['files'].items()):
                return {'directory': str(destination), 'reused': True, 'records': len(records)}
            raise ValueError('Existing report is corrupt')
        destination.parent.mkdir(exist_ok=True)
        stage = destination.parent / ('.staging-' + uuid4().hex)
        stage.mkdir()
        summary = {'schema_version': 1, 'identity': identity, 'days': study['days'], 'burn_in': study['burn_in'],
                   'counts': {model: sum(r['model'] == model for r in records) for model in ('abm', 'reduction')},
                   'groups': aggregate_records(records), 'table_columns': list(METRICS),
                   'table_run_ids': [r['run_id'] for r in records],
                   'interpretation': ['Historical and fresh cohorts are separate; five fresh seeds remain exploratory.',
                                      'Quartiles are dispersion across seeds or deterministic initial histories, not confidence intervals.',
                                      'Primary fidelity comparisons use zero closing noise and constant dividends.',
                                      'Fractional positive volume below one share is reported without rounding.',
                                      'Order parcels depend on quadrature resolution and are not counts of ABM agents.',
                                      'No attractor, stability, Lyapunov, or bifurcation classification is performed.']}
        write_json(stage / 'summary.json', summary)
        with gzip.open(stage / 'per_run.jsonl.gz', 'wt') as stream:
            for row in records:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        numeric = np.array([[r['summary']['values'].get(k) if r['summary']['values'].get(k) is not None else np.nan
                             for k in METRICS] for r in records], dtype=float)
        np.savez_compressed(stage / 'tables.npz', values=numeric)
        primary_points = study['quadrature_points'][0]

        def rows(model, cell, cohort='fresh', points=None):
            return [r for r in records if r['model'] == model and r['cell'] == cell and r['cohort'] == cohort
                    and (model == 'abm' or r['points'] == (primary_points if points is None else points))]

        def trajectories(selected, field):
            values = []
            for row in selected:
                # Plotting needs a few numeric columns, read one run at a time.
                with np.load(Path(sources[row['run_id']]) / 'daily.npz', allow_pickle=False) as archive:
                    values.append(archive[field])
            return np.array(values)

        def band(ax, series, color, label, style='-'):
            q = np.quantile(series, [.25, .5, .75], axis=0)
            x = np.arange(1, study['days'] + 1)
            ax.fill_between(x, q[0], q[2], color=color, alpha=.13, linewidth=0)
            ax.plot(x, q[1], style, color=color, label=label, linewidth=1.4)
            ax.axvline(study['burn_in'], color='.6', linewidth=.7, linestyle=':')
            ax.grid(alpha=.18)

        def save(fig, name):
            fig.savefig(stage / f'{name}.png', dpi=160)
            fig.savefig(stage / f'{name}.pdf')
            plt.close(fig)

        with plt.rc_context({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}):
            fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True, layout='constrained')
            for variant, ax in enumerate(axes.flat):
                cell = f'reference-v{variant}'
                mapped = f'reference-v{6 + variant % 2}'
                band(ax, trajectories(rows('abm', cell), 'price'), '#555555', 'Fresh ABM')
                band(ax, trajectories(rows('abm', cell, 'historical'), 'price'), '#B18A46', 'Historical ABM', ':')
                band(ax, trajectories(rows('reduction', mapped), 'price'), '#2878A5', f'Hard reduction, K={primary_points}', '--')
                ax.set_title(f"Closing noise {0.05 if variant < 4 else 0:g}; {'stochastic' if variant % 4 < 2 else 'constant'} dividends; {'persistent' if variant % 2 == 0 else 'day'}")
                ax.set_yscale('log')
                ax.set_ylabel('Price (log scale)')
            axes[0, 0].legend(fontsize=8)
            for ax in axes[-1]:
                ax.set_xlabel('Trading day')
            fig.suptitle('Reference alpha=0.1, gamma=2 | Median and IQR; cohorts kept separate')
            save(fig, 'reference_prices')

            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout='constrained')
            for col, cell in enumerate(('liquidity-v6', 'liquidity-v7')):
                for row, field in enumerate(('price', 'volume')):
                    ax = axes[row, col]
                    band(ax, trajectories(rows('abm', cell), field), '#555555', 'Fresh ABM')
                    band(ax, trajectories(rows('abm', cell, 'historical'), field), '#B18A46', 'Historical ABM', ':')
                    band(ax, trajectories(rows('reduction', cell), field), '#2878A5', f'Hard reduction, K={primary_points}', '--')
                    ax.set_ylabel('Price (log scale)' if row == 0 else 'Volume (shares/day; symlog)')
                    if row == 0:
                        ax.set_yscale('log')
                    if row == 1:
                        ax.set_yscale('symlog', linthresh=1)
                        ax.set_ylim(bottom=0)
                        ax.axhline(1, color='#C84B41', alpha=.5, linewidth=.8)
                        ax.set_xlabel('Trading day')
                axes[0, col].set_title('Persistent orders' if col == 0 else 'Day orders')
            axes[0, 0].legend(fontsize=8)
            fig.suptitle('Exploratory liquidity anchor: alpha≈0.2, gamma=0.5 | Zero noise, constant dividends')
            save(fig, 'liquidity_paths')

            for coordinate in ('reference', 'liquidity'):
                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout='constrained')
                for col, v in enumerate((6, 7)):
                    cell = f'{coordinate}-v{v}'
                    for row, field in enumerate(('total_cash', 'total_assets')):
                        ax = axes[row, col]
                        for model, style in (('abm', '-'), ('reduction', '--')):
                            data = trajectories(rows(model, cell), field)
                            fractions = data / data.sum(axis=2, keepdims=True)
                            for kind, name in enumerate(('Noise', 'Chartist', 'Fundamentalist')):
                                band(ax, fractions[:, :, kind] * 100, COLORS[kind],
                                     name + (' ABM' if model == 'abm' else ' reduction'), style)
                        ax.set_ylabel('Share of cash (%)' if row == 0 else 'Share of holdings (%)')
                        ax.set_ylim(-2, 102)
                    axes[0, col].set_title('Persistent orders' if v == 6 else 'Day orders')
                    axes[1, col].set_xlabel('Trading day')
                axes[0, 0].legend(fontsize=7, ncol=2)
                fig.suptitle(f'{coordinate.title()} resources | Fresh ABM versus deterministic initial-history ensemble')
                save(fig, f'{coordinate}_resources')

            selected_metrics = ('terminal_stall', 'zero_volume_fraction', 'subshare_volume_fraction',
                                'mean_volume', 'maximum_drawdown', 'max_outstanding_orders')
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
            for metric, ax in zip(selected_metrics, axes.flat):
                for model, offset, color in (('abm', -.18, '#777777'), ('reduction', .18, '#2878A5')):
                    samples = [[r['summary']['values'][metric] for r in rows(model, cell)] for cell in PRIMARY]
                    centers = np.array([np.mean(s) if metric == 'terminal_stall' else np.median(s) for s in samples])
                    ax.bar(np.arange(4) + offset, centers, width=.34, color=color, alpha=.85, label=model)
                    if metric != 'terminal_stall':
                        lo = centers - np.array([np.quantile(s, .25) for s in samples])
                        hi = np.array([np.quantile(s, .75) for s in samples]) - centers
                        ax.errorbar(np.arange(4) + offset, centers, yerr=[lo, hi], fmt='none', color='.2', linewidth=.8)
                    else:
                        for i, s in enumerate(samples):
                            ax.text(i + offset, centers[i] + .025, f'{int(sum(s))}/{len(s)}', ha='center', fontsize=8)
                ax.set_title(METRICS[metric])
                ax.set_xticks(range(4), LABELS, fontsize=8)
                ax.grid(axis='y', alpha=.18)
                if metric in ('terminal_stall', 'zero_volume_fraction', 'subshare_volume_fraction', 'maximum_drawdown'):
                    ax.set_ylim(0, 1.15 if metric == 'terminal_stall' else 1.05)
                if metric in ('mean_volume', 'max_outstanding_orders'):
                    ax.set_yscale('symlog', linthresh=1)
                    ax.set_ylim(bottom=0)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f'Trading continuity and closure gaps | Fresh cohort; primary K={primary_points}; final analysis window')
            save(fig, 'continuity')

            fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
            resolutions = sorted(study['quadrature_points'])
            for metric, ax in zip(selected_metrics, axes.flat):
                for cell, label, color in zip(PRIMARY, LABELS, COLORS):
                    samples = [[r['summary']['values'][metric] for r in rows('reduction', cell, points=k)] for k in resolutions]
                    center = [np.mean(s) if metric == 'terminal_stall' else np.median(s) for s in samples]
                    ax.plot(resolutions, center, 'o-', color=color, label=label.replace('\n', ' '))
                    if metric != 'terminal_stall':
                        ax.fill_between(resolutions, [np.quantile(s, .25) for s in samples],
                                        [np.quantile(s, .75) for s in samples], color=color, alpha=.12)
                ax.set_title(METRICS[metric])
                ax.set_xticks(resolutions)
                ax.set_xlabel('Quote quadrature points')
                ax.grid(alpha=.18)
                if metric in ('mean_volume', 'max_outstanding_orders'):
                    ax.set_yscale('symlog', linthresh=1)
                    ax.set_ylim(bottom=0)
                if metric in ('terminal_stall', 'zero_volume_fraction', 'subshare_volume_fraction'):
                    ax.set_ylim(0, 1.05)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle('Quadrature sensitivity | Fresh initial histories; no claim of pathwise convergence')
            save(fig, 'quadrature_sensitivity')
        files = {p.name: file_hash(p) for p in stage.iterdir()}
        write_json(stage / 'render.json', {'schema_version': 1, 'identity': identity, 'files': files,
                                          'records': len(records), 'figures': 6})
        if disk_guard(root, Limits()):
            raise ValueError('Report output limit reached')
        os.rename(stage, destination)
        return {'directory': str(destination), 'records': len(records), 'figures': 6,
                'artifact_bytes': directory_bytes(destination)}

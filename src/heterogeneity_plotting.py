"""Auditable numeric reports and exportable figures for Task 5."""

from collections import Counter
import gzip
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import matplotlib.pyplot as plt

from src.atlas_runner import manifest_at
from src.experiment_artifacts import canonical, digest, file_hash, write_json
from src.experiments import Limits, directory_bytes, disk_guard, experiment_lock, _fsync_tree, _fsync_directory
from src.heterogeneity_analysis import SETTINGS, collect_fidelity, discover_structure
from src.heterogeneity_study import reference_jobs, continuation_jobs, fingerprint, validate_artifact

COLORS = ['#155f9b', '#d78024', '#2b876b', '#a64253', '#755699', '#555555', '#8c6d31', '#4f97a3']


def _save(fig, directory, name):
    fig.savefig(directory / f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def figures(directory, study, fidelity, discovery, numeric):
    levels = sorted(k for k in study['levels'] if k != 'full') + ['full']
    cells = sorted(study['configurations'])
    horizons = [h for h in study['horizons'] if h >= 100]
    comparisons = fidelity['comparisons']
    if comparisons and horizons:
        fig, axes = plt.subplots(len(cells), len(horizons), figsize=(6*len(horizons), 3.1*len(cells)), squeeze=False)
        for i, cell in enumerate(cells):
            for j, horizon in enumerate(horizons):
                ax = axes[i, j]
                for q, day in enumerate(study['checkpoints']):
                    rows = {r['level']: r for r in comparisons if r['cell'] == cell and r['cohort'] == 'validation'
                            and r['checkpoint'] == day and r['horizon'] == horizon}
                    if not rows:
                        continue
                    x = np.arange(len(levels)) + (q - (len(study['checkpoints'])-1)/2)*.08
                    y = np.array([rows[str(k)]['score'] for k in levels])
                    low, high = np.array([rows[str(k)]['score_interval'] for k in levels]).T
                    # Sampling bounds need not bracket the plug-in estimate.
                    ax.plot(x, y, 'o-', color=COLORS[q], label=f'Checkpoint {day}', ms=3)
                    ax.vlines(x, low, high, color=COLORS[q], alpha=.65)
                ax.axhline(1, color='#af2436', ls='--', label='Tolerance boundary')
                ax.set_yscale('symlog', linthresh=1)
                ax.set_ylim(bottom=0)
                ax.set_xticks(range(len(levels)), [str(k) for k in levels])
                ax.set_title(f'{cell} | {horizon} future days')
                ax.set_xlabel('Portfolio groups per strategy')
                ax.set_ylabel('Maximum normalized distribution error')
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'Held-out ABM fidelity | {len(study["validation_seeds"])} seeds | Bootstrap intervals; stall bounds checked separately', y=1.005)
        fig.tight_layout()
        _save(fig, directory, 'fidelity_profiles')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for q, level in enumerate(levels):
        rows = [r for r in fidelity['records'] if r['cohort'] == 'validation' and r['level'] == str(level)
                and r['horizon'] == max(study['horizons'])]
        detail = [g for r in rows for g in r['detail'].values() if g['retained_dispersion'] is not None]
        if not detail:
            continue
        retention = [g['retained_dispersion'] for g in detail]
        distance = [g['joint_wasserstein'] for g in detail]
        axes[0].scatter(retention, distance, s=6, alpha=.3, color=COLORS[q], label=str(level))
        axes[1].plot(q, np.median(retention), 'o', color=COLORS[q])
        axes[1].vlines(q, *np.quantile(retention, [.25, .75]), color=COLORS[q])
    axes[0].set(xlabel='Fraction of portfolio dispersion retained', ylabel='Normalized joint portfolio distance',
                title='Checkpoint compression; undefined zero-dispersion groups omitted')
    axes[0].set_yscale('symlog', linthresh=.01)
    axes[0].legend(title='Groups', fontsize=8)
    axes[1].set(xticks=range(len(levels)), xticklabels=[str(k) for k in levels], xlabel='Groups per strategy',
                ylabel='Dispersion retained: median and IQR', title='Held-out checkpoint portfolios')
    fig.tight_layout()
    _save(fig, directory, 'retained_individuality')
    if discovery.get('status') == 'ok':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, mode in zip(axes, ('current', 'lagged')):
            if discovery.get(mode, {}).get('status') != 'ok':
                ax.text(.5, .5, discovery.get(mode, {}).get('status', 'undefined'), ha='center')
                continue
            xy, labels = numeric[f'{mode}_validation_projection'], numeric[f'{mode}_validation_clusters']
            # Fixed thinning only affects rendering; all rows remain in numeric data.
            stride = max(1, len(xy)//10000)
            ax.scatter(xy[::stride, 0], xy[::stride, 1], c=labels[::stride], cmap='tab10', s=3, alpha=.35)
            ax.set(xlabel='Principal component 1', ylabel='Principal component 2', title=f'{mode.capitalize()} features | held-out reference runs')
        fig.suptitle('Structures learned on development runs; projections are not attractor claims')
        fig.tight_layout()
        _save(fig, directory, 'discovered_structure')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, mode in zip(axes, ('current', 'lagged')):
            for q, cell in enumerate(cells):
                rows = [r for r in discovery['compressed_projections'] if r['mode'] == mode and r['cell'] == cell
                        and r['checkpoint'] == max(study['checkpoints'])]
                bylevel = {r['level']: r for r in rows}
                if bylevel:
                    ax.plot(range(len(levels)), [bylevel[str(k)]['reconstruction_mse'] for k in levels],
                            'o-', label=cell, color=COLORS[q], ms=3)
            ax.set(xticks=range(len(levels)), xticklabels=[str(k) for k in levels], xlabel='Groups per strategy',
                   ylabel='Standardized feature reconstruction MSE', title=f'{mode.capitalize()} | checkpoint {max(study["checkpoints"])}')
            ax.set_yscale('symlog', linthresh=.01)
            ax.legend(fontsize=7)
        fig.tight_layout()
        _save(fig, directory, 'frozen_representation_error')


def render_study(root, *, resamples=2000):
    if type(resamples) is not int or resamples < 100:
        raise ValueError('At least 100 bootstrap resamples are required')
    root = Path(root).resolve()
    with experiment_lock(root):
        study = manifest_at(root)
        if study['analysis_settings'] != SETTINGS:
            raise ValueError('Restore the predeclared analysis settings before evaluating held-out data')
        refs = reference_jobs(study)
        jobs = continuation_jobs(root, study, refs)
        inputs = {job['run_id']: file_hash(root / job['phase'] / 'runs' / job['run_id'] / 'complete.json') for job in refs + jobs}
        for job in jobs:
            validate_artifact(root / 'continuation/runs' / job['run_id'], job, study['execution'])
        identity = {'study_sha256': digest(study), 'inputs': inputs, 'settings': SETTINGS,
                    'bootstrap_resamples': resamples, 'renderer': fingerprint()}
        report_id = digest(identity)
        parent = root / 'reports'
        parent.mkdir(exist_ok=True)
        target = parent / report_id
        if target.exists():
            complete = json.loads((target / 'complete.json').read_text())
            if set(complete['files']) != {p.name for p in target.iterdir() if p.is_file() and p.name != 'complete.json'}:
                raise ValueError('Report contents changed')
            if any(file_hash(target / name) != value for name, value in complete['files'].items()):
                raise ValueError('Report checksum mismatch')
            return {'directory': str(target), 'reused': True}
        if disk_guard(root, Limits(), reserved=256*1024**2):
            raise ValueError('Report disk/output budget reached')
        stage = parent / f'.staging-{uuid4().hex}'
        stage.mkdir()
        fidelity = collect_fidelity(root, study, jobs, resamples=resamples)
        discovery, numeric = discover_structure(root, study, refs, jobs)
        summary = {'identity': identity, 'counts': {'reference_runs': len(refs), 'checkpoint_jobs': len(jobs)},
                   'comparisons': fidelity['comparisons'], 'thresholds': fidelity['thresholds'], 'discovery': discovery,
                   'abm_sampling_baselines': fidelity['abm_sampling_baselines'],
                   'threshold_status_counts': dict(Counter(r['status'] for r in fidelity['thresholds'] if r['cohort'] == 'validation')),
                   'limitations': ['One-off initial portfolio compression; all agents and subsequent trading remain individual.',
                                   'Profiles can split through integer share balancing; portfolio assignments and orders remain retained.',
                                   'Intervals are exploratory per comparison, not a universal jointly certified threshold.',
                                   'Equal starting RNG states do not align subsequent shocks after draw consumption diverges.',
                                   'Task 4 is a different closure, not the one-profile endpoint; its reported fidelity failures remain.']}
        write_json(stage / 'summary.json', summary)
        with gzip.open(stage / 'records.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(fidelity['records'], stream, allow_nan=False, separators=(',', ':'))
        np.savez_compressed(stage / 'discovery.npz', **numeric)
        figures(stage, study, fidelity, discovery, numeric)
        write_json(stage / 'complete.json', {'report_id': report_id,
                   'files': {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}})
        _fsync_tree(stage)
        os.rename(stage, target)
        _fsync_directory(parent)
        return {'directory': str(target), 'reused': False, 'thresholds': summary['threshold_status_counts']}

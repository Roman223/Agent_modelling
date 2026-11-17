"""Stream Task 7 control summaries and render auditable scientific figures."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import resource

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from src.atlas_runner import atomic_json
from src.experiment_artifacts import ROOT, digest, file_hash, read_json
from src.experiments import Limits, MIB, disk_guard, experiment_lock
from src.multifractal_study import (FAMILIES, analysis_directory, cached_record, jobs_for,
                                    load_study, make_job, result_path, windows_for)
from src.run_logging import json_safe

COLORS = {'shuffled': '#2878b5', 'iaaft': '#d68428', 'garch': '#5a9a63'}


def read_complete(output, manifest, job):
    record = cached_record(result_path(output, job), job, manifest['analysis_id'])
    if record is None or record['state'] != 'complete':
        raise ValueError(f'Incomplete or corrupt analysis job: {job["job_id"]}')
    return record['result']


def control_summary(records, order, fit_name, observed, expected):
    counts = Counter(record['diagnostic']['status'] for record in records)
    fitted, curves = [], []
    for record in records:
        if record['diagnostic']['status'] != 'ok' or record['analysis'] is None:
            continue
        part = record['analysis'].get('orders', {}).get(order)
        if part is None:
            continue
        fit = part[fit_name]
        fitted.append(fit)
        h = np.asarray(fit['h'], dtype=float)
        if np.isfinite(h).all():
            curves.append(h)
    result = {'expected': expected, 'generated': len(records), 'diagnostic_counts': dict(counts),
              'weak_scaling_count': sum(f['status'] == 'weak_scaling' for f in fitted),
              'statistics': {}, 'finite_h_curves': len(curves)}
    for metric in ('delta_h', 'spectrum_width'):
        values = [f[metric] for f in fitted if f[metric] is not None and np.isfinite(f[metric])]
        finite = len(values)
        quantiles = np.quantile(values, [.025, .5, .975]).tolist() if values else [None]*3
        original = observed.get(metric) if observed else None
        complete = finite == expected and len(records) == expected
        result['statistics'][metric] = {
            'observed': original, 'valid': finite, 'expected': expected, 'quantiles': quantiles,
            'observed_minus_control_median': original-quantiles[1] if original is not None and finite else None,
            'status': 'observed_undefined' if original is None else 'complete' if complete else 'incomplete_controls',
            'above_reference_interval': bool(original > quantiles[2]) if complete and original is not None else None}
    result['h_quantiles'] = np.quantile(curves, [.025, .5, .975], axis=0).tolist() if len(curves) == expected else None
    return result


def collect_study(root):
    root = Path(root).resolve()
    manifest = load_study(root)
    output = analysis_directory(root, manifest)
    windows = windows_for(root, manifest)
    inventory = read_json(output/'inventory.json')
    if (inventory['sha256'] != digest(inventory['payload'])
            or inventory['payload'] != {'analysis_id': manifest['analysis_id'], 'windows': windows}):
        raise ValueError('Analysis inventory changed')
    rows = []
    for window in windows:
        observed = read_complete(output, manifest, make_job(window, 'observed'))
        garch = read_complete(output, manifest, make_job(window, 'fit'))['fit']
        row = {'window': window, 'observed': observed, 'garch_fit': garch, 'controls': {}}
        for family in FAMILIES:
            replicas = []
            for job in jobs_for([window], 'controls', manifest['settings']):
                if job['family'] == family:
                    replicas.extend(read_complete(output, manifest, job)['replicates'])
            if [r['replicate'] for r in replicas] != list(range(manifest['settings']['replicates'])):
                raise ValueError('Missing or duplicate control replicates')
            family_summary = {}
            for order in ('1', '2'):
                part = observed['analysis'].get('orders', {}).get(order, {})
                family_summary[order] = {fit_name: control_summary(replicas, order, fit_name,
                                                                  part.get(fit_name), manifest['settings']['replicates'])
                                         for fit_name in ('fit', 'common_fit')}
            row['controls'][family] = family_summary
        rows.append(row)
    return manifest, rows


def title_for(row):
    w = row['window']
    return w['label'] + (f" · seed {w['seed']}" if w['kind'] == 'simulation' else '')


def panel_grid(count, subtitle):
    columns = min(4, count)
    rows = (count+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.3*columns, 3.0*rows),
                             layout='constrained', squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    fig.suptitle(subtitle, fontsize=15)
    return fig, list(axes.flat)[:count]


def plot_scaling(primary, settings):
    fig, axes = panel_grid(len(primary), f'Daily return fluctuations — {settings["primary_length"]:,} observations; linear detrending')
    for row, ax in zip(primary, axes):
        ax.set_title(title_for(row), fontsize=9)
        part = row['observed']['analysis'].get('orders', {}).get('1')
        if part is None:
            ax.text(.5, .5, row['observed']['analysis']['status'].replace('_', ' '),
                    transform=ax.transAxes, ha='center', va='center', color='#777777')
            ax.set(xticks=[], yticks=[])
            continue
        scales, q = np.array(part['scales']), np.array(settings['q'])
        logf = np.array(part['log_fluctuations'], dtype=float)
        for power, color in ((-3, '#2878b5'), (0, '#666666'), (3, '#c34c43')):
            j = int(np.flatnonzero(q == power)[0])
            valid = np.isfinite(logf[:, j])
            ax.plot(scales[valid], np.exp(logf[valid, j]), 'o-', ms=3, color=color, label=f'q = {power}')
        ax.set(xscale='log', yscale='log', xlabel='Scale (trading observations)', ylabel='Fq (normalized returns)')
        ticks = [int(s) for s in scales if int(s) & (int(s)-1) == 0]
        ax.set_xscale('log', base=2)
        ax.set_xticks(ticks, [str(s) for s in ticks], fontsize=8)
        ax.xaxis.set_minor_locator(NullLocator())
        if part['fit']['status'] != 'ok':
            ax.text(.98, .03, part['fit']['status'].replace('_', ' '),
                    ha='right', transform=ax.transAxes, fontsize=7, color='#8a3a28')
        ax.grid(alpha=.2)
    handles, labels = next((ax.get_legend_handles_labels() for ax in axes if ax.get_legend_handles_labels()[0]), ([], []))
    if handles:
        axes[0].legend(handles, labels, fontsize=7)
    return fig


def plot_hurst(primary, settings):
    fig, axes = panel_grid(len(primary), 'Generalized Hurst curves — observed and 95% control reference intervals')
    q = np.array(settings['q'])
    for row, ax in zip(primary, axes):
        ax.set_title(title_for(row), fontsize=9)
        part = row['observed']['analysis'].get('orders', {}).get('1')
        if part:
            ax.plot(q, np.array(part['fit']['h'], float), 'o-', color='black', ms=3, label='Observed')
            if part['fit']['status'] != 'ok':
                ax.text(.03, .03, part['fit']['status'].replace('_', ' '),
                        transform=ax.transAxes, fontsize=7, color='#8a3a28')
            for family in FAMILIES:
                comparison = row['controls'][family]['1']['fit']
                bands = comparison['h_quantiles']
                if bands is not None:
                    lo, median, hi = np.array(bands)
                    ax.plot(q, median, color=COLORS[family], lw=1, label=family.upper() if family == 'iaaft' else family.title())
                    ax.fill_between(q, lo, hi, color=COLORS[family], alpha=.13)
            ax.set(xlabel='Moment order q', ylabel='h(q)')
            ax.grid(alpha=.2)
        else:
            ax.text(.5, .5, 'Undefined: '+row['observed']['analysis']['status'].replace('_', ' '),
                    ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set(xticks=[], yticks=[])
    legend = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    if legend:
        fig.legend(legend.values(), legend.keys(), loc='outside lower center', ncols=4, fontsize=9)
    return fig


def plot_spectra(primary):
    fig, axes = panel_grid(len(primary), 'Legendre spectra over q ∈ [−3, 3] — finite-sample estimates')
    for row, ax in zip(primary, axes):
        ax.set_title(title_for(row), fontsize=9)
        drawn = False
        for order, color in (('1', '#2878b5'), ('2', '#c34c43')):
            part = row['observed']['analysis'].get('orders', {}).get(order)
            if part and part['fit']['spectrum_width'] is not None:
                fit = part['fit']
                ax.plot(fit['alpha'], fit['spectrum'], 'o-', ms=3, color=color, label=f'DFA{order}')
                drawn = True
        if not drawn:
            ax.text(.5, .5, 'Full spectrum undefined', ha='center', va='center', transform=ax.transAxes, color='#777777')
            ax.set(xticks=[], yticks=[])
        weak = [f'DFA{order}' for order, part in row['observed']['analysis'].get('orders', {}).items()
                if part['fit']['status'] == 'weak_scaling']
        if weak:
            ax.set_title(title_for(row)+'\nWeak scaling: '+', '.join(weak), fontsize=9)
        ax.set(xlabel='Singularity exponent α', ylabel='f(α)')
        ax.grid(alpha=.2)
    handles, labels = next((ax.get_legend_handles_labels() for ax in axes if ax.get_legend_handles_labels()[0]), ([], []))
    if handles:
        axes[0].legend(handles, labels, fontsize=7)
    return fig


def plot_contrasts(primary):
    fig, axes = plt.subplots(1, 2, figsize=(15, max(7, len(primary)*.4)), layout='constrained')
    for ax, metric, label in zip(axes, ('delta_h', 'spectrum_width'), ('Δh = h(−3) − h(3)', 'Spectrum width Δα')):
        for i, row in enumerate(primary):
            part = row['observed']['analysis'].get('orders', {}).get('1')
            value = part['fit'][metric] if part else None
            if value is not None:
                ax.plot(value, i, 'ko', ms=5, label='Observed')
            else:
                ax.text(.98, i, 'undefined', transform=ax.get_yaxis_transform(), ha='right', fontsize=7, color='#888888')
            for offset, family in zip((-.18, 0, .18), FAMILIES):
                comparison = row['controls'][family]['1']['fit']['statistics'][metric]
                if comparison['status'] == 'complete':
                    lo, median, hi = comparison['quantiles']
                    ax.errorbar(median, i+offset, xerr=[[median-lo], [hi-median]], fmt='|',
                                color=COLORS[family], capsize=2, label=family)
        ax.set_yticks(range(len(primary)), [title_for(r) for r in primary], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc='best', fontsize=8)
    count = primary[0]['controls']['shuffled']['1']['fit']['expected']
    length = primary[0]['window']['length']
    fig.suptitle(f'Observed statistics and complete {count}-realization control intervals — DFA1, N={length:,}')
    return fig


def plot_sensitivity(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['window']['label']].append(row)
    fig, axes = panel_grid(len(groups), 'Length and detrending sensitivity — nested windows are dependent')
    for (label, group), ax in zip(groups.items(), axes):
        ax.set_title(label, fontsize=10)
        for order, fit_name, style, color, caption in (
                ('1', 'fit', '-', '#2878b5', 'DFA1: full grid'),
                ('2', 'fit', '--', '#c34c43', 'DFA2: full grid'),
                ('1', 'common_fit', ':', '#666666', 'DFA1: scales 16–128')):
            by_seed = defaultdict(list)
            for row in group:
                part = row['observed']['analysis'].get('orders', {}).get(order)
                value = part[fit_name]['delta_h'] if part else None
                by_seed[row['window']['seed']].append((row['window']['length'], np.nan if value is None else value))
            for j, points in enumerate(by_seed.values()):
                points.sort()
                ax.plot([p[0] for p in points], [p[1] for p in points], style+'o', color=color,
                        alpha=.65, ms=3, label=caption if j == 0 else None)
        if not any(np.isfinite(line.get_ydata()).any() for line in ax.lines):
            ax.text(.5, .5, 'Δh undefined at all lengths', ha='center', transform=ax.transAxes, fontsize=8)
            ax.set_yticks([])
        ax.set(xscale='log', xlabel='Window length', ylabel='Δh')
        lengths = sorted({row['window']['length'] for row in group})
        ax.set_xscale('log', base=2)
        ax.set_xticks(lengths, [f'{n:,}' for n in lengths], fontsize=8)
        ax.set_xlim(lengths[0]/1.15, lengths[-1]*1.15)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    return fig


def number(value):
    return '—' if value is None else f'{value:.4f}'


def primary_groups(primary):
    groups = defaultdict(list)
    for row in primary:
        if row['window']['kind'] == 'simulation':
            groups[row['window']['label']].append(row)
    summaries = []
    for label, group in groups.items():
        summary = {'label': label, 'expected': len(group),
                   'seeds': [r['window']['seed'] for r in group], 'statistics': {}}
        for metric in ('delta_h', 'spectrum_width'):
            values = [r['observed']['analysis'].get('orders', {}).get('1', {}).get('fit', {}).get(metric)
                      for r in group]
            values = [v for v in values if v is not None and np.isfinite(v)]
            summary['statistics'][metric] = {
                'valid': len(values),
                'median': float(np.median(values)) if values else None,
                'minimum': float(np.min(values)) if values else None,
                'maximum': float(np.max(values)) if values else None}
        summaries.append(summary)
    return summaries


def findings_markdown(manifest, rows):
    primary = [r for r in rows if r['window']['length'] == manifest['settings']['primary_length']]
    overview = []
    for kind, caption in (('simulation', 'Simulations'), ('index', 'Indices')):
        subset = [r for r in primary if r['window']['kind'] == kind]
        counts = Counter(r['observed']['analysis'].get('orders', {}).get('1', {}).get('fit', {}).get(
                         'status', r['observed']['analysis']['status']) for r in subset)
        outcomes = ', '.join(f'{count} {status.replace("_", " ")}' for status, count in counts.items())
        comparisons = []
        for family in FAMILIES:
            complete = [r['controls'][family]['1']['fit']['statistics']['delta_h'] for r in subset
                        if r['controls'][family]['1']['fit']['statistics']['delta_h']['status'] == 'complete']
            result = (f'{sum(c["above_reference_interval"] for c in complete)}/{len(complete)}'
                      if complete else 'unavailable')
            comparisons.append(f'{family}: {result}')
        overview.append(f'{caption}: {outcomes}. Counts above the upper 95% control bound for Δh '
                        'among complete comparisons — '+', '.join(comparisons)+'.')
    lines = ['# Multifractal pilot findings', '',
             'Research question: do simulated and real stock-index returns exhibit comparable '
             'multifractal scaling, and how much survives stochastic and finite-length controls?', '',
             f'Analysis identity: `{manifest["analysis_id"]}`.', '',
             f'The fixed pilot contains {len(manifest["configurations"])*len(manifest["seeds"])} simulations, '
             f'{len(manifest["indices"])} indices, and {len(rows)} source/window combinations. '
             f'Primary windows contain {manifest["settings"]["primary_length"]:,} returns. Simulations retain '
             f'{manifest["days"]-manifest["burn_in"]:,} daily observations after {manifest["burn_in"]:,} '
             'discarded trading days. All windows are end-aligned.', '',
             '## Outcome overview', '', *overview, '',
             'These are descriptive finite-sample comparisons, not multiplicity-adjusted tests. '
             'Weak fits remain flagged, and unavailable comparisons provide no evidence either way. '
             'Similar spectrum widths alone do not establish ABM–index equivalence or deterministic chaos.', '',
             '## Primary observations', '',
             '| Source | Seed | Return SD | No-trade fraction | Δh | Δα | Scaling status |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in primary:
        a, w = row['observed']['analysis'], row['window']
        fit = a.get('orders', {}).get('1', {}).get('fit', {})
        no_trade = row['observed']['window_diagnostics'].get('zero_volume_fraction')
        lines.append(f'| {w["label"]} | {w["seed"] if w["seed"] is not None else "—"} | '
                     f'{number(a.get("return_sd"))} | {number(no_trade)} | '
                     f'{number(fit.get("delta_h"))} | {number(fit.get("spectrum_width"))} | {fit.get("status", a["status"])} |')
    lines += ['', '## Dispersion across simulation seeds', '',
              'Medians and ranges below use only finite primary DFA1 estimates. '
              'Coverage is shown explicitly; these ranges are descriptive, not population confidence intervals.', '',
              '| Configuration | Valid/expected | Median Δh | Δh range | Median Δα | Δα range |',
              '| --- | --- | --- | --- | --- | --- |']
    for group in primary_groups(primary):
        h, a = [group['statistics'][name] for name in ('delta_h', 'spectrum_width')]
        lines.append(f'| {group["label"]} | {h["valid"]}/{group["expected"]} | '
                     f'{number(h["median"])} | [{number(h["minimum"])}, {number(h["maximum"])}] | '
                     f'{number(a["median"])} | [{number(a["minimum"])}, {number(a["maximum"])}] |')
    lines += ['', '## Control comparisons', '',
              'Intervals below are 95% control reference intervals for Δh, not confidence intervals '
              'for the ABM population. Incomplete ensembles are retained as inconclusive. '
              'Weak-scaling control fits remain in the reference distribution when their statistic is finite.', '',
              '| Source | Seed | Control | Valid/expected | Observed − control median | Control 95% interval | Comparison |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for row in primary:
        for family in FAMILIES:
            comparison = row['controls'][family]['1']['fit']['statistics']['delta_h']
            lo, _, hi = comparison['quantiles']
            w = row['window']
            lines.append(f'| {w["label"]} | {w["seed"] if w["seed"] is not None else "—"} | {family} | '
                         f'{comparison["valid"]}/{comparison["expected"]} | '
                         f'{number(comparison["observed_minus_control_median"])} | '
                         f'[{number(lo)}, {number(hi)}] | {comparison["status"]} |')
    lines += ['', '## Interpretation and limitations', '',
              '- Report each seed separately; three seeds provide exploratory dispersion, not a precise population interval.',
              '- Nested windows share observations. Their agreement is a sensitivity check, not independent replication.',
              '- The short fitted scale ranges make asymptotic scaling uncertain even when R-squared is high.',
              '- The two detrending orders, full scale grids, and common 16–128 scale fits are retained in the tables and figures.',
              '- Undefined negative moments or full spectra are outcomes of zero or unresolved segment fluctuations. '
              'Inactive days were retained; no jitter or segment stitching was used.',
              '- Constant-dividend/zero-closing-noise variants remain stochastic through individual decisions. '
              'Their joint intervention cannot isolate the two forcing mechanisms individually.',
              '- Shuffling, IAAFT, and fitted GARCH address different null models. A large spectrum width alone '
              'does not establish deterministic chaos, a causal mechanism, or empirical equivalence.',
              '- A fitted GARCH control is unavailable when convergence or stationarity checks fail. '
              'IAAFT results above 1% relative Fourier-amplitude error are excluded from that control comparison, with counts retained.',
              '- Exchange closures and holidays are recorded as gaps. The index clock uses consecutive recorded observations.',
              '', '## Figures', '',
              '![Scaling curves](scaling_curves.png)', '', '![Generalized Hurst curves](hurst_curves.png)', '',
              '![Multifractal spectra](multifractal_spectra.png)', '',
              '![Control comparisons](control_contrasts.png)', '', '![Sensitivity](sensitivity.png)', '',
              '## Methods and reproduction', '',
              'MF-DFA: Kantelhardt et al. (2002), https://doi.org/10.1016/S0378-4371(02)01383-3. '
              'Software adaptations and licenses are recorded in THIRD_PARTY_NOTICES.md. '
              'The immutable study manifest records settings, source hashes, input snapshots and package versions. '
              'Resume with `experiments/run_multifractal.py study STUDY`; inspect with `status STUDY`; '
              'regenerate figures with `render STUDY`.', '']
    return '\n'.join(lines)


def render_study(root, *, limits=None, locked=False):
    root = Path(root).resolve()
    limits = limits or Limits()
    if not locked:
        with experiment_lock(root):
            return render_study(root, limits=limits, locked=True)
    manifest, rows = collect_study(root)
    output = root/'reports'/manifest['analysis_id']
    output.mkdir(parents=True, exist_ok=True)
    if disk_guard(root, limits, reserved=128*MIB):
        raise ValueError('Study resource budget prevents rendering')
    primary = [r for r in rows if r['window']['length'] == manifest['settings']['primary_length']]
    summary = {'analysis_id': manifest['analysis_id'], 'settings': manifest['settings'],
               'windows': rows, 'operational_failures': 0,
               'primary_simulation_groups': primary_groups(primary),
               'observed_status_counts': dict(Counter(r['observed']['analysis']['status'] for r in rows))}
    atomic_json(output/'summary.json', json_safe(summary))
    numeric = {'label': np.array([r['window']['label'] for r in rows]),
               'seed': np.array([-1 if r['window']['seed'] is None else r['window']['seed'] for r in rows]),
               'length': np.array([r['window']['length'] for r in rows]), 'q': np.array(manifest['settings']['q'])}
    for order in ('1', '2'):
        for fit_name in ('fit', 'common_fit'):
            for metric in ('h', 'alpha', 'spectrum', 'delta_h', 'spectrum_width'):
                values = []
                for row in rows:
                    fit = row['observed']['analysis'].get('orders', {}).get(order, {}).get(fit_name, {})
                    default = [np.nan]*len(numeric['q']) if metric in ('h', 'alpha', 'spectrum') else np.nan
                    values.append(fit.get(metric, default))
                numeric[f'order{order}_{fit_name}_{metric}'] = np.asarray(values, dtype=float)
    np.savez_compressed(output/'tables.npz', **numeric)
    (output/'findings.md').write_text(findings_markdown(manifest, rows), encoding='utf-8')
    for name, builder in (
            ('scaling_curves', lambda: plot_scaling(primary, manifest['settings'])),
            ('hurst_curves', lambda: plot_hurst(primary, manifest['settings'])),
            ('multifractal_spectra', lambda: plot_spectra(primary)),
            ('control_contrasts', lambda: plot_contrasts(primary)),
            ('sensitivity', lambda: plot_sensitivity(rows))):
        if disk_guard(root, limits, reserved=128*MIB):
            raise ValueError('Study resource budget prevents figure generation')
        fig = builder()
        fig.savefig(output/(name+'.png'), dpi=140)
        fig.savefig(output/(name+'.pdf'))
        plt.close(fig)
    atomic_json(output/'render.json', {'analysis_id': manifest['analysis_id'],
                                     'renderer_sha256': file_hash(ROOT/'src/multifractal_plotting.py'),
                                     'matplotlib': matplotlib.__version__,
                                     'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                     'files': {p.name: file_hash(p) for p in output.iterdir() if p.name != 'render.json'}})
    print(json.dumps({'event': 'rendered', 'directory': str(output), 'windows': len(rows)}), flush=True)
    return output

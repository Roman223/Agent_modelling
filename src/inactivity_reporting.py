"""Streaming study aggregation; no pooled daily tables or future pseudoreplication."""

import csv
import gzip
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.experiment_artifacts import digest, file_hash, read_json, write_json
from src.experiments import MIB, GIB, disk_guard, resident_bytes, memory_available, _remove_stage
from src.inactivity import ARMS, REASONS, spells
from src.inactivity_analysis import factorial_effects, survival, incidence

SERIES = ('execution_fraction', 'distinct_exchange_fraction', 'volume', 'self_volume', 'price',
          'price_to_decision_sma_minus_one', 'accepted_price_return', 'bid_count', 'ask_count',
          'relative_spread', 'both_book_sides_present', 'dropped_bids', 'dropped_asks',
          'cash_share_noise', 'cash_share_chartist', 'cash_share_fundamentalist',
          'asset_share_noise', 'asset_share_chartist', 'asset_share_fundamentalist',
          'buy_orders_noise', 'buy_orders_chartist', 'buy_orders_fundamentalist',
          'sell_orders_noise', 'sell_orders_chartist', 'sell_orders_fundamentalist',
          'fundamentalist_below_threshold_fraction', 'chartist_buy_order_fraction',
          'noise_no_holdings_fraction', 'active_noise', 'active_chartist', 'active_fundamentalist')


def daily_series(a):
    n = len(a['day'])
    total = lambda key: np.divide(a[key], a[key].sum(axis=1, keepdims=True),
                                 out=np.zeros((n, 3)), where=a[key].sum(axis=1, keepdims=True) != 0)
    both = (a['bid_count'] > 0) & (a['ask_count'] > 0)
    reasons = a['decision_counts']
    population = reasons.sum(axis=2)
    reason_fraction = lambda group, name: np.divide(reasons[:, group, REASONS.index(name)], population[:, group],
                                     out=np.zeros(n), where=population[:, group] != 0)
    return np.column_stack((a['volume'] > 0, a['distinct_volume'] > 0, a['volume'], a['same_owner_volume'],
        a['price'], a['previous_price']/a['decision_sma']-1,
        a['previous_price']/a['decision_previous_price']-1,
        a['bid_count'], a['ask_count'], np.where(both, (a['best_ask']-a['best_bid'])/a['price'], 0),
        both, a['match_counts'][:, :2], total('total_cash'), total('total_assets'),
        a['submissions'][:, :, 0], a['submissions'][:, :, 1],
        reason_fraction(2, 'excess_return_below_threshold'), reason_fraction(1, 'price_above_average'),
        reason_fraction(0, 'no_holdings'), a['active_count']))


def guard(root, limits):
    reason = disk_guard(root, limits, reserved=64*MIB)
    if reason or resident_bytes(os.getpid()) > limits.memory_gib*GIB or memory_available() < GIB:
        raise RuntimeError(reason or 'analysis_memory_pressure')


def validate_report(directory):
    directory = Path(directory)
    envelope = read_json(directory / 'complete.json')
    actual = {p.name for p in directory.iterdir() if p.is_file()} - {'complete.json'}
    if actual != set(envelope['files']) or any(file_hash(directory/k) != v for k, v in envelope['files'].items()):
        raise ValueError('Analysis report is corrupt')
    return read_json(directory / 'summary.json')


def analyze(root, limits):
    from src.inactivity_study import (load_study, reference_jobs, continuation_jobs,
                                      read_arrays, validate_artifact, fingerprint)
    study = load_study(root, current=True)
    refs = reference_jobs(study)
    jobs = continuation_jobs(root, study, refs)  # validates every reference, including no-stall runs
    for job in jobs:
        validate_artifact(root / 'continuation/runs' / job['run_id'], job, study['execution'])
    inputs = {f"{job['phase']}/{job['run_id']}": file_hash(root / job['phase'] / 'runs' / job['run_id'] / 'complete.json')
              for job in refs + jobs}
    identity = {'manifest_sha256': digest(study), 'inputs': inputs, 'execution': fingerprint()}
    destination = root / 'analysis' / digest(identity)
    if destination.exists():
        validate_report(destination)
        return {'directory': str(destination), 'reused': True}
    guard(root, limits)
    destination.parent.mkdir(exist_ok=True)
    stage = destination.parent / ('.staging-' + uuid4().hex)
    stage.mkdir()
    try:
        config_names = sorted(study['configurations'])
        days, settings = study['days'], study['analysis_settings']
        radius = settings['event_radius']
        reference_rows, effects = [], {}
        plots = {'day': np.arange(1, days+1), 'event_day': np.arange(-radius, radius+1)}
        by_cell = {}
        with gzip.open(stage / 'fixed_windows.csv.gz', 'wt', newline='') as stream, \
                gzip.open(stage / 'spells.jsonl.gz', 'wt', encoding='utf-8') as spell_stream:
            writer = csv.writer(stream)
            writer.writerow(['cell', 'seed', 'window', 'first_day', 'last_day', 'contributing_days', *SERIES])
            for cell_index, cell in enumerate(config_names):
                selected = [r for r in refs if r['cell'] == cell]
                total = np.zeros((days, len(SERIES)))
                event_sum = np.zeros((2*radius+1, len(SERIES)))
                event_count = np.zeros(2*radius+1, dtype=np.int32)
                durations, duration_events, cell_rows = [], [], []
                decision_total = np.zeros((3, len(REASONS)), dtype=np.int64)
                for ref in selected:
                    guard(root, limits)
                    path = root / 'reference/runs' / ref['run_id']
                    validate_artifact(path, ref, study['execution'], deep=True)
                    a = read_arrays(path / 'daily.npz', full=True)
                    series = daily_series(a)
                    total += series
                    decision_total += a['decision_counts'].sum(axis=0).astype(np.int64)
                    row = {'cell': cell, 'seed': ref['seed'], **read_json(path / 'summary.json')}
                    reference_rows.append(row)
                    cell_rows.append(row)
                    periods = read_json(path / 'spells.json')
                    for period in periods:
                        spell_stream.write(json.dumps({'cell': cell, 'seed': ref['seed'], **period}) + '\n')
                        durations.append(period['duration'])
                        duration_events.append(not period['right_censored'])
                    first = row['thresholds'][str(study['threshold'])]['first_onset']
                    if first is not None:
                        positions = np.arange(first-1-radius, first+radius)
                        valid = (positions >= 0) & (positions < days)
                        event_sum[valid] += series[positions[valid]]
                        event_count[valid] += 1
                    for window in settings['fixed_windows']:
                        for start in range(0, days, window):
                            end = min(days, start+window)
                            writer.writerow([cell, ref['seed'], window, start+1, end, end-start,
                                             *series[start:end].mean(axis=0)])
                    del a, series
                plots[f'cell{cell_index}_daily_mean'] = total/len(selected)
                plots[f'cell{cell_index}_event_mean'] = np.divide(event_sum, event_count[:, None],
                    out=np.zeros_like(event_sum), where=event_count[:, None] != 0)
                plots[f'cell{cell_index}_event_count'] = event_count
                plots[f'cell{cell_index}_decision_total'] = decision_total
                coverage = {}
                for threshold in settings['thresholds']:
                    times = [r['thresholds'][str(threshold)]['first_confirmation'] for r in cell_rows]
                    coverage[str(threshold)] = {'incidence': incidence([t is not None for t in times]),
                        'time_to_confirmation': survival([days if t is None else t for t in times],
                                                        [t is not None for t in times])}
                by_cell[cell] = {'n_references': len(selected), 'thresholds': coverage,
                    'mean_zero_volume_fraction': float(np.mean([r['zero_volume_fraction'] for r in cell_rows])),
                    'mean_distinct_day_fraction': float(np.mean([r['distinct_day_fraction'] for r in cell_rows])),
                    'spell_duration_descriptive': survival(durations, duration_events, confidence=False),
                    'spell_duration_note': 'Repeated spells are dependent; no iid spell confidence intervals.'}
        for cell_index, cell in enumerate(config_names):
            selected = [j for j in jobs if j['cell'] == cell]
            outcomes = np.zeros((len(selected), 8, study['futures']))
            recovery_curves = np.zeros((8, study['continuation_days']))
            exchange_curves = np.zeros_like(recovery_curves)
            secondary = {arm: [] for arm in ARMS}
            unique = 0
            for row, job in enumerate(selected):
                guard(root, limits)
                path = root / 'continuation/runs' / job['run_id']
                validate_artifact(path, job, study['execution'], deep=True)
                record = read_json(path / 'recovery.json')
                unique += record['unique_continuations']
                for k, arm in enumerate(ARMS):
                    summaries = [record['states'][s]['summary'] for s in record['arms'][arm]['futures']]
                    outcomes[row, k] = [s['tail_distinct_fraction'] for s in summaries]
                    for s in summaries:
                        for key, curves in (('first_execution', recovery_curves), ('first_distinct_exchange', exchange_curves)):
                            if s[key] is not None:
                                curves[k, s[key]-1:] += 1/study['futures']
                    secondary[arm].append({'seed': job['seed'], 'futures': summaries})
            effects[cell] = {'eligible_checkpoints': len(selected), 'unique_continuations': unique,
                'logical_continuations': len(selected)*8*study['futures'],
                'status': 'ok' if selected else 'not_applicable',
                'primary': factorial_effects(outcomes, seed=settings['analysis_seed']+1000*cell_index,
                                             resamples=settings['bootstrap_resamples']),
                'secondary': secondary}
            plots[f'cell{cell_index}_recovery_cdf'] = recovery_curves / max(1, len(selected))
            plots[f'cell{cell_index}_distinct_cdf'] = exchange_curves / max(1, len(selected))
            plots[f'cell{cell_index}_primary_outcomes'] = outcomes
        for name, rows in (('references', reference_rows),):
            with gzip.open(stage / f'{name}.jsonl.gz', 'wt', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
        summary = {'schema_version': 1, 'identity': identity, 'configurations': config_names,
                   'series': list(SERIES), 'settings': settings, 'days': days, 'threshold': study['threshold'],
                   'continuation_days': study['continuation_days'], 'futures': study['futures'],
                   'reference_count': len(refs), 'eligible_checkpoints': len(jobs),
                   'replay_verified_count': len(jobs), 'references': by_cell, 'recovery': effects,
                   'limitations': ['Recovery effects are conditional on first confirmed stalled checkpoints.',
                       'Futures share initial RNG states across arms, not necessarily later shocks.',
                       'Finite nonrecovery does not prove permanent absorption.',
                       'Intervals are pointwise; no multiplicity-adjusted confirmation claims.',
                       'Degenerate bootstrap intervals do not imply population certainty.',
                       'All sixteen cohorts are fresh; earlier studies are not pooled.',
                       'A missing book side encodes spread as zero; use both_book_sides_present when interpreting it.']}
        write_json(stage / 'summary.json', summary)
        np.savez_compressed(stage / 'tables.npz', **plots)
        lines = ['# Task 6: market inactivity and recovery', '',
                 f"Completed {len(refs)} references; {len(jobs)} eligible checkpoints, each replay-verified.", '',
                 'The primary recovery outcome is the fraction of distinct-trader exchange days in the final '
                 f"{settings['primary_tail_days']} continuation days. Futures are averaged within each checkpoint; "
                 'confidence intervals resample reference seeds.', '',
                 '| Configuration | Confirmed stalls | Mean no-trade days | Control tail activity | All-interventions tail activity |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for cell in config_names:
            ref, eff = by_cell[cell], effects[cell]
            count = ref['thresholds'][str(study['threshold'])]['incidence']
            fmt = lambda value: 'N/A' if value is None else f'{100*value:.1f}%'
            lines.append(f"| {cell} | {count['events']}/{count['n']} | {fmt(ref['mean_zero_volume_fraction'])} | "
                         f"{fmt(eff['primary']['arms']['000']['estimate'])} | {fmt(eff['primary']['arms']['111']['estimate'])} |")
        lines += ['', *summary['limitations'], '',
                  'Arm bits, in order: clear queues; reset shared accepted-price memory; redistribute active portfolios.',
                  'See summary.json for all eight arms, paired contrasts, factorial interactions, intervals and flags.',
                  'Time-to-confirmation survival includes finite-horizon right censoring. Operational failures are never censored.']
        (stage / 'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        if fingerprint() != study['execution']:
            raise ValueError('Execution changed during analysis')
        guard(root, limits)
        write_json(stage / 'complete.json', {'files': {p.name: file_hash(p) for p in sorted(stage.iterdir())}})
        os.rename(stage, destination)
        validate_report(destination)
        return {'directory': str(destination), 'reused': False}
    finally:
        if stage.exists():
            _remove_stage(stage)

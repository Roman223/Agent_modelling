"""Predeclared fidelity criteria, seed-level uncertainty, and frozen discovery."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from src.experiment_artifacts import digest, read_json
from src.heterogeneity import STRATEGIES, FEATURE_NAMES, LAGS

SETTINGS = {'version': 1, 'bootstrap_resamples': 2000, 'analysis_seed': 54001,
            'volume_offset': 1., 'volatility_offset': .0001,
            'tolerances': {'terminal_stall': .10, 'zero_volume_fraction': .05,
                           'mean_volume': float(np.log(1.2)), 'log_return_volatility': float(np.log(1.2)),
                           'cumulative_log_return': float(np.log(1.2)),
                           **{f'{resource}_fraction_{kind}': .05 for resource in ('cash', 'share') for kind in STRATEGIES}},
            'primary_min_horizon': 100, 'pca_dimensions': [2, 4, 8], 'cluster_counts': [2, 3, 4, 5, 6],
            'kmeans_n_init': 20, 'feature_stride': 10, 'future_horizon': 20,
            'uncertainty': '95% seed-bootstrap intervals for maximum normalized distribution distance; paired-binomial stall bound; exploratory, not simultaneous across all cells'}


def outcome_summary(arrays, horizon):
    r, v = arrays['log_return'][:horizon], arrays['volume'][:horizon]
    if len(v) != horizon or horizon < 1:
        raise ValueError('Invalid summary horizon')
    positive = r[v > 0]
    summary = {'terminal_stall': bool(np.all(v[-100:] == 0)) if horizon >= 100 else None,
               'zero_volume_fraction': float(np.mean(v == 0)), 'mean_volume': float(np.mean(v)),
               'log_return_volatility': float(np.std(r, ddof=1)) if horizon > 1 else None,
               'positive_volume_volatility': float(np.std(positive, ddof=1)) if len(positive) > 1 else None,
               'cumulative_log_return': float(np.sum(r)), 'subshare_volume_fraction': float(np.mean((v > 0) & (v < 1))),
               'return_quantiles': np.quantile(r, [.05, .5, .95]).tolist(),
               'within_strategy_volume_fraction': float(np.sum(arrays['within_strategy_volume'][:horizon]) / np.sum(v)) if np.sum(v) else None,
               'same_owner_volume_fraction': float(np.sum(arrays['same_owner_volume'][:horizon]) / np.sum(v)) if np.sum(v) else None}
    for name, field in (('cash', 'total_cash'), ('share', 'total_assets')):
        values = arrays[field][horizon-1]
        for kind, value in zip(STRATEGIES, values):
            summary[f'{name}_fraction_{kind}'] = float(value / np.sum(values)) if np.sum(values) else None
    return summary


def _matrix(rows, names):
    result = np.array([[r[name] for name in names] for r in rows], dtype=float)
    for j, name in enumerate(names):
        if name == 'mean_volume':
            result[:, j] = np.log1p(result[:, j])
        elif name == 'log_return_volatility':
            result[:, j] = np.log(result[:, j] + SETTINGS['volatility_offset'])
    return result


def paired_stall_interval(a, b):
    """Conservative 95% paired risk-difference interval, including zero cells."""
    n = len(a)
    if n < 2:
        return [-1., 1.]
    plus, minus = np.count_nonzero(a & ~b), np.count_nonzero(~a & b)
    # Two 97.5% exact binomial intervals jointly cover at least 95% by union bound.
    p = binomtest(int(plus), n).proportion_ci(confidence_level=.975, method='exact')
    m = binomtest(int(minus), n).proportion_ci(confidence_level=.975, method='exact')
    return [float(p.low - m.high), float(p.high - m.low)]


def compare_outcomes(rows, reference, *, seed, exact=False, resamples=2000):
    if len(rows) != len(reference) or not rows:
        raise ValueError('Comparisons require nonempty matched seed cohorts')
    names = [name for name in SETTINGS['tolerances'] if all(r.get(name) is not None for r in rows + reference)]
    if not names:
        return {'status': 'inconclusive', 'reason': 'No defined primary outcomes', 'n_seeds': len(rows)}
    a, b = _matrix(rows, names), _matrix(reference, names)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'status': 'inconclusive', 'reason': 'Nonfinite outcomes', 'n_seeds': len(rows)}
    tolerances = np.array([SETTINGS['tolerances'][name] for name in names])
    # Equal-weight 1-D empirical W1, vectorized across outcomes and bootstrap draws.
    distances = np.mean(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)), axis=0)
    score = float(np.max(distances / tolerances))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(resamples, len(rows)))
    samples = np.mean(np.abs(np.sort(a[indices], axis=1) - np.sort(b[indices], axis=1)), axis=1)
    bounds = np.quantile(np.max(samples / tolerances, axis=1), [.025, .975]).tolist()
    stall_interval = None
    if 'terminal_stall' in names:
        stall_interval = paired_stall_interval(np.array([r['terminal_stall'] for r in rows], dtype=bool),
                                               np.array([r['terminal_stall'] for r in reference], dtype=bool))
    if exact:
        if not np.array_equal(a, b):
            raise AssertionError('Exact endpoint summaries differ')
        bounds, stall_interval, state = [0., 0.], [0., 0.], 'within_tolerance'
    elif len(rows) < 2:
        state = 'inconclusive'
    elif bounds[0] > 1 or (stall_interval and (stall_interval[0] > .1 or stall_interval[1] < -.1)):
        state = 'outside_tolerance'
    elif bounds[1] <= 1 and (stall_interval is None or max(abs(x) for x in stall_interval) <= .1):
        state = 'within_tolerance'
    else:
        state = 'inconclusive'
    return {'status': state, 'n_seeds': len(rows), 'score': score, 'score_interval': bounds,
            'stall_difference_interval': stall_interval, 'distances': dict(zip(names, distances.tolist())),
            'normalized_distances': dict(zip(names, (distances/tolerances).tolist())),
            'undefined': [name for name in SETTINGS['tolerances'] if name not in names]}



def sampling_baseline(reference, seed, resamples):
    """Independent disjoint seed halves; descriptive sampling variability only."""
    names = [name for name in SETTINGS['tolerances'] if all(r.get(name) is not None for r in reference)]
    n = len(reference)//2
    if n < 2 or not names:
        return {'status': 'insufficient_seeds'}
    a, b = _matrix(reference[::2][:n], names), _matrix(reference[1::2][:n], names)
    rng = np.random.default_rng(seed)
    ia, ib = rng.integers(0, n, (resamples, n)), rng.integers(0, n, (resamples, n))
    distances = np.mean(np.abs(np.sort(a[ia], axis=1)-np.sort(b[ib], axis=1)), axis=1)
    scores = np.max(distances/np.array([SETTINGS['tolerances'][name] for name in names]), axis=1)
    return {'status': 'ok', 'seeds_per_half': n, 'score_quantiles': np.quantile(scores, [.025, .5, .975]).tolist(),
            'interpretation': 'ABM versus ABM, half-sized independent cohorts; not an equivalence tolerance'}

def select_threshold(comparisons, levels):
    """A passing level is insufficient if any finer tested level is unresolved."""
    ordered = sorted([k for k in levels if k != 'full']) + ['full']
    for i, level in enumerate(ordered):
        if all(comparisons[str(k)]['status'] == 'within_tolerance' for k in ordered[i:]):
            return {'level': level, 'status': 'no_supported_compression' if level == 'full' else 'candidate',
                    'interpretation': 'initial portfolio information at this horizon; not a persistent reduced simulator'}
    return {'level': None, 'status': 'inconclusive'}


def collect_fidelity(root, study, jobs, *, resamples=None):
    from src.heterogeneity_study import validate_artifact
    records, groups = [], defaultdict(dict)
    for job in jobs:
        path = root / 'continuation/runs' / job['run_id']
        validate_artifact(path, job, study['execution'])
        bundle = read_json(path / 'levels.json')
        for level, entry in bundle['levels'].items():
            for horizon, summary in bundle['states'][entry['state']]['summaries'].items():
                row = {'cell': job['cell'], 'cohort': job['cohort'], 'seed': job['seed'],
                       'checkpoint': job['checkpoint_day'], 'horizon': int(horizon), 'level': level,
                       'checkpoint_stalled': bundle['checkpoint_stalled'], 'summary': summary,
                       'detail': entry['compression']['groups'],
                       'exact_source': entry['compression']['compressed_sha256'] == entry['compression']['source_sha256']}
                records.append(row)
                key = job['cell'], job['cohort'], job['checkpoint_day'], int(horizon)
                groups[key].setdefault(level, []).append(row)
    comparisons, thresholds, baselines = [], [], []
    for key, levels in sorted(groups.items()):
        cell, cohort, checkpoint, horizon = key
        if horizon < SETTINGS['primary_min_horizon']:
            continue
        reference = sorted(levels['full'], key=lambda r: r['seed'])
        expected = sorted(study[f'{cohort}_seeds'])
        if [r['seed'] for r in reference] != expected:
            raise ValueError('Missing or duplicated seed coverage')
        seed = int(digest([SETTINGS['analysis_seed'], cohort, expected])[:8], 16)
        baseline = sampling_baseline([r['summary'] for r in reference], seed, resamples or SETTINGS['bootstrap_resamples'])
        baseline.update(cell=cell, cohort=cohort, checkpoint=checkpoint, horizon=horizon)
        baselines.append(baseline)
        local = {}
        for level, rows in levels.items():
            rows = sorted(rows, key=lambda r: r['seed'])
            if [r['seed'] for r in rows] != expected:
                raise ValueError('Unpaired comparison coverage')
            result = compare_outcomes([r['summary'] for r in rows], [r['summary'] for r in reference], seed=seed,
                                      exact=all(r['exact_source'] for r in rows),
                                      resamples=resamples or SETTINGS['bootstrap_resamples'])
            result.update(cell=cell, cohort=cohort, checkpoint=checkpoint, horizon=horizon, level=level)
            comparisons.append(result)
            local[level] = result
        threshold = select_threshold(local, study['levels'])
        threshold.update(cell=cell, cohort=cohort, checkpoint=checkpoint, horizon=horizon)
        thresholds.append(threshold)
    return {'records': records, 'comparisons': comparisons, 'thresholds': thresholds, 'abm_sampling_baselines': baselines}


def _feature_samples(arrays):
    """Frozen feature rows and subsequent 20-day outcomes, with no overlap leak."""
    indices = arrays['feature_days'] - arrays['day'][0]
    good = indices + 21 <= len(arrays['day'])
    indices = indices[good]
    outcomes = np.array([[np.sum(arrays['log_return'][i+1:i+21]),
                          np.mean(np.log1p(arrays['volume'][i+1:i+21])),
                          np.mean(arrays['volume'][i+1:i+21] == 0)] for i in indices], dtype=float).reshape(-1, 3)
    return arrays['features'][good], outcomes


def discover_structure(root, study, references, jobs):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from src.heterogeneity_study import read_arrays
    buckets = {c: {'x': [], 'y': [], 'seed': [], 'cell': []} for c in ('development', 'validation')}
    for job in references:
        arrays = read_arrays(root / 'reference/runs' / job['run_id'] / 'daily.npz')
        x, y = _feature_samples(arrays)
        bucket = buckets[job['cohort']]
        bucket['x'].append(x)
        bucket['y'].append(y)
        bucket['seed'].extend([job['seed']] * len(x))
        bucket['cell'].extend([job['cell']] * len(x))
    for bucket in buckets.values():
        bucket['x'] = np.concatenate(bucket['x'])
        bucket['y'] = np.concatenate(bucket['y'])
        bucket['seed'] = np.array(bucket['seed'])
    train, test = buckets['development'], buckets['validation']
    if len(train['x']) < 8 or not len(test['x']):
        return {'status': 'insufficient_observations', 'training_rows': len(train['x']), 'validation_rows': len(test['x'])}, {}
    if set(train['seed']) & set(test['seed']):
        raise ValueError('Discovery seed leakage')
    output, numeric, models = {}, {}, {}
    for mode, width in (('current', len(FEATURE_NAMES)), ('lagged', len(FEATURE_NAMES)*len(LAGS))):
        x, z = train['x'][:, :width], test['x'][:, :width]
        if not np.any(np.std(x, axis=0) > 0):
            output[mode] = {'status': 'constant_features'}
            continue
        scaler = StandardScaler().fit(x)
        tx, vx = scaler.transform(x), scaler.transform(z)
        dmax = min(8, len(tx)-1, width)
        pca = PCA(n_components=dmax, svd_solver='full').fit(tx)
        ts, vs = pca.transform(tx), pca.transform(vx)
        errors = {}
        for d in (2, 4, 8):
            if d > dmax:
                continue
            reconstruct = vs[:, :d] @ pca.components_[:d] + pca.mean_
            errors[str(d)] = {'validation_mse': float(np.mean((vx-reconstruct)**2)),
                              'development_variance_retained': float(pca.explained_variance_ratio_[:d].sum())}
        candidates, fitted = [], {}
        for k in SETTINGS['cluster_counts']:
            if k >= len(ts) or len(np.unique(ts, axis=0)) < k:
                continue
            model = KMeans(n_clusters=k, n_init=20, random_state=SETTINGS['analysis_seed']).fit(ts)
            score = silhouette_score(ts, model.labels_, sample_size=min(2000, len(ts)), random_state=SETTINGS['analysis_seed'])
            candidates.append({'clusters': k, 'silhouette': float(score)})
            fitted[k] = model
        if not candidates:
            output[mode] = {'status': 'insufficient_distinct_features', 'reconstruction': errors}
            continue
        best = max(candidates, key=lambda r: (r['silhouette'], -r['clusters']))['clusters']
        model = fitted[best]
        labels, vlabels = model.labels_, model.predict(vs)
        means = np.array([np.mean(train['y'][labels == k], axis=0) for k in range(best)])
        pred = means[vlabels]
        baseline = np.mean(train['y'], axis=0)
        seed_errors = np.array([np.mean(np.abs(pred[test['seed'] == seed]-test['y'][test['seed'] == seed])
                                       - np.abs(baseline-test['y'][test['seed'] == seed]), axis=0)
                                for seed in sorted(set(test['seed']))])
        rng = np.random.default_rng(SETTINGS['analysis_seed'])
        bootstrap = np.mean(seed_errors[rng.integers(0, len(seed_errors), (2000, len(seed_errors)))], axis=1)
        stability = []
        train_seeds = np.unique(train['seed'])
        for repeat in range(5):
            chosen = rng.choice(train_seeds, len(train_seeds), replace=True)
            selected = np.concatenate([np.flatnonzero(train['seed'] == seed) for seed in chosen])
            other = KMeans(n_clusters=best, n_init=20, random_state=SETTINGS['analysis_seed']+repeat+1).fit(ts[selected])
            stability.append(float(adjusted_rand_score(labels, other.predict(ts))))
        clusters = []
        for k in range(best):
            future = test['y'][vlabels == k]
            clusters.append({'cluster': k, 'validation_rows': len(future),
                             'development_rows': int(np.sum(labels == k)),
                             'future_median': np.median(future, axis=0).tolist() if len(future) else None,
                             'future_iqr': np.quantile(future, [.25, .75], axis=0).tolist() if len(future) else None})
        output[mode] = {'status': 'ok', 'training_rows': len(tx), 'validation_rows': len(vx),
                        'dimensions': dmax, 'reconstruction': errors, 'cluster_candidates': candidates,
                        'selected_clusters': best, 'bootstrap_cluster_ari': stability, 'clusters': clusters,
                        'future_targets': ['next_20_log_return', 'next_20_mean_log1p_volume', 'next_20_no_trade_fraction'],
                        'validation_future_mae': np.mean(np.abs(pred-test['y']), axis=0).tolist(),
                        'baseline_future_mae': np.mean(np.abs(baseline-test['y']), axis=0).tolist(),
                        'seed_mae_difference_interval': np.quantile(bootstrap, [.025, .975], axis=0).tolist(),
                        'training_seeds': train_seeds.tolist(), 'validation_seeds': sorted(set(test['seed'].tolist()))}
        numeric.update({f'{mode}_scaler_mean': scaler.mean_, f'{mode}_scaler_scale': scaler.scale_,
                        f'{mode}_pca_mean': pca.mean_, f'{mode}_components': pca.components_,
                        f'{mode}_variance': pca.explained_variance_ratio_, f'{mode}_cluster_centers': model.cluster_centers_,
                        f'{mode}_validation_projection': vs[:, :2], f'{mode}_validation_clusters': vlabels,
                        f'{mode}_validation_seeds': test['seed'], f'{mode}_future_means': means})
        models[mode] = scaler, pca, model, means, width
    # Apply frozen reference-trained models; never fit on intervention trajectories.
    accum = defaultdict(lambda: {'count': 0, 'mse_sum': 0., 'clusters': None, 'future_error': np.zeros(3)})
    for job in jobs:
        if job['cohort'] != 'validation':
            continue
        path = root / 'continuation/runs' / job['run_id']
        bundle = read_json(path / 'levels.json')
        for state_id, state in bundle['states'].items():
            x, y = _feature_samples(read_arrays(path / state['file']))
            if not len(x):
                continue
            levels = [level for level, entry in bundle['levels'].items() if entry['state'] == state_id]
            for mode, (scaler, pca, model, means, width) in models.items():
                transformed = scaler.transform(x[:, :width])
                projection = pca.transform(transformed)
                labels = model.predict(projection)
                mse = float(np.sum(np.mean((transformed-pca.inverse_transform(projection))**2, axis=1)))
                for level in levels:
                    a = accum[mode, job['cell'], job['checkpoint_day'], level]
                    a['count'] += len(x)
                    a['mse_sum'] += mse
                    counts = np.bincount(labels, minlength=len(means))
                    a['clusters'] = counts if a['clusters'] is None else a['clusters'] + counts
                    a['future_error'] += np.sum(np.abs(means[labels]-y), axis=0)
    output['compressed_projections'] = [{'mode': mode, 'cell': cell, 'checkpoint': day, 'level': level,
         'rows': a['count'], 'reconstruction_mse': a['mse_sum']/a['count'],
         'cluster_occupancy': (a['clusters']/a['count']).tolist(),
         'future_mae': (a['future_error']/a['count']).tolist()}
        for (mode, cell, day, level), a in sorted(accum.items())]
    output['status'] = 'ok' if models else 'undefined'
    return output, numeric

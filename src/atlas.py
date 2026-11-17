"""Task 3 design and chronological diagnostics, separate from legacy metrics."""

from collections import Counter
from copy import deepcopy
from itertools import product
import warnings

import numpy as np
from nolds import lyap_r, lyap_r_len, sampen

from src.experiment_artifacts import digest, summarize

SCHEMA = 1
SETTINGS = {
    'schema_version': SCHEMA,
    'window': '[burn_in, days), daily log returns including zero-volume days',
    'volatility_ddof': 1,
    'terminal_stall_days': 100,
    'sampen': {'emb_dim': 3, 'lag': 1, 'closed': False, 'tolerance_sd_fraction': 0.2},
    'rosenstein': {'emb_dim': 10, 'lag': 1, 'min_tsep': 20, 'trajectory_len': 20,
                   'fit_offset': 0, 'tau': 1, 'min_neighbors': 20, 'fit': 'RANSAC'},
    'fit_seed': 'first 8 hex digits of SHA256 canonical JSON [simulation_seed, estimator]',
}
METRIC_LABELS = {
    'final_price': 'Final price (currency/share)',
    'cumulative_log_return': 'Window log return',
    'log_return_volatility': 'Daily return standard deviation',
    'positive_volume_volatility': 'Return standard deviation on trading days',
    'maximum_drawdown': 'Maximum drawdown (fraction)',
    'mean_volume': 'Mean daily volume (shares)',
    'zero_volume_fraction': 'Zero-volume fraction',
    'longest_no_trade_streak': 'Longest no-trade streak (days in window)',
    'terminal_no_trade_streak': 'Terminal no-trade streak (days in window)',
    'terminal_stall': 'Terminal stall frequency (at least 100 days)',
    'sample_entropy': 'Sample Entropy (nats)',
    'rosenstein': 'Rosenstein estimate (per trading day)',
}
OPERATIONAL = {'estimator_error', 'timeout', 'worker_error', 'artifact_error'}


def axes():
    alpha = np.geomspace(0.025, 0.4, 17)
    alpha[0], alpha[8], alpha[-1] = 0.025, 0.1, 0.4
    gamma = np.geomspace(0.5, 8, 7)
    gamma[0], gamma[3], gamma[-1] = 0.5, 2.0, 8.0
    return alpha.tolist(), alpha[::2].tolist(), gamma.tolist()


def variants():
    """The named legacy stochastic reference comes first at every coordinate."""
    return [dict(closing_noise_scale=noise, dividend_mode=dividend, order_lifetime=orders)
            for noise, dividend, orders in product((0.05, 0.0), ('stochastic', 'constant'),
                                                   ('persistent', 'day'))]


def variant_name(scenario):
    return 'noise-{closing_noise_scale:g}_{dividend_mode}_{order_lifetime}'.format(**scenario)


def make_spec(*, seeds=(100, 101, 102, 103, 104), days=2000, burn_in=1000):
    """Emit schema-1 cells in seed/coordinate/variant order; expand_spec deduplicates."""
    sweep, grid, gamma = axes()
    coordinates = [('sweep', i, 0, a, 2.0) for i, a in enumerate(sweep)]
    coordinates += [('grid', i, j, a, g) for i, a in enumerate(grid) for j, g in enumerate(gamma)]
    cells = []
    for seed in seeds:
        for view, i, j, alpha, risk in coordinates:
            for v, controls in enumerate(variants()):
                cells.append({'name': f'{view}-{i}-{j}-v{v}-s{seed}', 'scenario': 'baseline',
                              'seeds': [seed], 'overrides': {
                                  'chartist_alpha': alpha, 'fundamentalist_gamma': risk,
                                  'chartist_window': 20, 'evolution': False,
                                  'circuit_breaker_asset': False, 'halt_check': False, **controls}})
    return {'schema_version': 1, 'name': 'task3-parameter-atlas', 'days': days,
            'burn_in': burn_in, 'cells': cells}


def fit_seed(seed, estimator):
    return int(digest([int(seed), estimator])[:8], 16)


def value_record(value=None, status='ok', reason=None, **extra):
    if value is not None:
        value = float(value)
        if not np.isfinite(value):
            value, status, reason = None, 'non_finite_result', 'Estimator returned a nonfinite result'
    return {'value': value, 'status': status, 'reason': reason, **extra}


def nonlinear(values, seed, settings=None):
    """Use the supplied chronological scalar series, without return preprocessing.

    Restores global RNG even on failure. Never removes samples or adds jitter.
    Known-system validation may supply level series and its own explicit settings.
    """
    settings = deepcopy(SETTINGS if settings is None else settings)
    values = np.asarray(values, dtype=float)
    result = {}
    for metric, key in (('sample_entropy', 'sampen'), ('rosenstein', 'rosenstein')):
        parameters = dict(settings[key])
        estimator_seed = fit_seed(seed, metric) if metric == 'rosenstein' else None
        common = {'parameters': parameters, 'fit_seed': estimator_seed,
                  'observations': len(values), 'warnings': []}
        minimum = (parameters['emb_dim'] + 2 if key == 'sampen'
                   else lyap_r_len(**parameters) + 1)
        if values.ndim != 1 or len(values) < minimum:
            result[metric] = value_record(status='insufficient_input', reason=f'Need {minimum} scalar samples', **common)
            continue
        if not np.all(np.isfinite(values)):
            result[metric] = value_record(status='non_finite_input', reason='Input contains nonfinite samples', **common)
            continue
        sd = float(np.std(values))
        constant = np.all(values == values[0]) or sd == 0
        if constant or not np.isfinite(sd):
            result[metric] = value_record(status='constant_series' if constant else 'non_finite_input',
                                          reason='Zero or nonfinite standard deviation', **common)
            continue
        state = np.random.get_state()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                if key == 'sampen':
                    parameters['tolerance'] = parameters.pop('tolerance_sd_fraction') * sd
                    estimate = sampen(values, **parameters)
                    record = value_record(estimate, **common)
                else:
                    np.random.seed(estimator_seed)
                    estimate, (steps, divergence, polynomial) = lyap_r(values, debug_data=True, **parameters)
                    finite_fit = np.isfinite(polynomial).all()
                    diagnostic = {'steps': steps.tolist(), 'mean_log_distance': divergence.tolist(),
                                  'polynomial': np.asarray(polynomial).tolist() if finite_fit else None}
                    record = value_record(estimate, fit_diagnostic=diagnostic, **common)
            except Exception as error:
                record = value_record(status='estimator_error', reason=f'{type(error).__name__}: {error}', **common)
            finally:
                np.random.set_state(state)
        record['warnings'] = [str(item.message) for item in caught]
        result[metric] = record
    return result


def analyze_arrays(arrays, burn_in, seed, settings=None):
    settings = SETTINGS if settings is None else settings
    summary = summarize(arrays, burn_in)
    metrics = {name: value_record(value, 'ok' if value is not None else 'undefined',
                                  summary['undefined'].get(name))
               for name, value in summary['values'].items()}
    returns, volume = arrays['log_return'][burn_in:], arrays['volume'][burn_in:]
    if not np.all(np.isfinite(returns)):
        metrics['cumulative_log_return'] = value_record(status='non_finite_input', reason='Nonfinite daily returns')
        metrics['log_return_volatility'] = value_record(status='non_finite_input', reason='Nonfinite daily returns')
    else:
        metrics['cumulative_log_return'] = value_record(np.sum(returns))
    traded = returns[volume > 0]
    metrics['positive_volume_volatility'] = (
        value_record(np.std(traded, ddof=1)) if len(traded) >= 2
        else value_record(status='insufficient_input', reason='Need two positive-volume days'))
    metrics['mean_volume'] = value_record(np.mean(volume))
    metrics['terminal_stall'] = value_record(summary['values']['terminal_no_trade_streak'] >= settings['terminal_stall_days'])
    metrics.update(nonlinear(returns, seed, settings))
    return {'burn_in': burn_in, 'observations': len(returns), 'positive_volume_days': len(traded), 'metrics': metrics}


def aggregate(manifest, records):
    """Aggregate scalar results only; expected runs always come from the manifest."""
    cells = {}
    for run in manifest['runs']:
        scenario = run['scenario']
        coordinate = (variant_name(scenario), scenario['chartist_alpha'], scenario['fundamentalist_gamma'])
        cell = cells.setdefault(coordinate, {'variant': coordinate[0], 'alpha': coordinate[1], 'gamma': coordinate[2],
                                           'expected': 0, 'analyzed': 0, 'seeds': [], 'metrics': {}})
        cell['expected'] += 1
        cell['seeds'].append(run['seed'])
        record = records.get(run['run_id'])
        if record and record.get('state') == 'complete':
            cell['analyzed'] += 1
        for name in METRIC_LABELS:
            bucket = cell['metrics'].setdefault(name, {'values': [], 'statuses': Counter()})
            metric = record.get('result', {}).get('metrics', {}).get(name) if record else None
            status = metric['status'] if metric else (record['state'] if record else 'missing')
            bucket['statuses'][status] += 1
            if metric and status == 'ok' and metric['value'] is not None:
                bucket['values'].append(metric['value'])
    for cell in cells.values():
        for name, bucket in cell['metrics'].items():
            values = bucket.pop('values')
            bucket['valid'] = len(values)
            bucket['expected'] = cell['expected']
            bucket['statuses'] = dict(bucket['statuses'])
            q = np.quantile(values, [0.25, 0.5, 0.75]) if values else [None] * 3
            bucket.update(zip(('q25', 'median', 'q75'), (float(x) if x is not None else None for x in q)))
            # Binary stall incidence is a proportion, not the median of indicators.
            if name == 'terminal_stall':
                bucket['frequency'] = float(np.mean(values)) if values else None
    return {'schema_version': SCHEMA, 'expected_runs': len(manifest['runs']),
            'analyzed_runs': sum(c['analyzed'] for c in cells.values()), 'cells': list(cells.values())}


def validation_controls():
    """Known-system fixtures use levels, explicitly separate from ABM log returns."""
    def logistic(r):
        x, trajectory = 0.123456789, []
        for step in range(2048):
            x = r * x * (1 - x)
            if step >= 1024:
                trajectory.append(x)
        return np.array(trajectory)

    chaotic = logistic(4.0)
    short_fit = deepcopy(SETTINGS)
    short_fit['rosenstein'].update(emb_dim=2, min_tsep=10, trajectory_len=5, fit='poly')
    estimate = nonlinear(chaotic, 90210, short_fit)['rosenstein']
    if estimate['status'] != 'ok' or abs(estimate['value'] - np.log(2)) > 0.12:
        raise AssertionError(f'Logistic short-fit validation failed: {estimate}')
    fixtures = {'logistic_r4': chaotic, 'logistic_r3.2': logistic(3.2),
                'gaussian_noise': np.random.default_rng(90210).normal(size=1000),
                'constant': np.ones(1000)}
    return {'description': 'Level-series controls; production ABM uses daily log returns. No chaos classifier.',
            'logistic_expected': float(np.log(2)), 'logistic_short_fit': estimate,
            'production_settings': SETTINGS,
            'production_results': {name: nonlinear(data, 90210) for name, data in fixtures.items()}}

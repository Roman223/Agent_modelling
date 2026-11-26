"""Finite-horizon stall estimates and checkpoint-paired factorial contrasts."""

from itertools import combinations
import warnings

import numpy as np
from scipy import stats

from src.inactivity import ARMS, spells, positive_integer

SETTINGS = {'thresholds': [20, 100, 250], 'fixed_windows': [100, 500],
            'event_radius': 500, 'primary_tail_days': 100, 'bootstrap_resamples': 2000,
            'confidence_level': .95, 'bootstrap_method': 'percentile',
            'small_sample_threshold': 10, 'analysis_seed': 6002026,
            'inference_unit': 'reference_seed', 'interval_scope': 'pointwise',
            'recovery_population': 'first confirmed stalled checkpoints only'}


def survival(times, observed, *, confidence=True):
    times, observed = np.asarray(times, float), np.asarray(observed, bool)
    if (times.ndim != 1 or times.shape != observed.shape or not np.isfinite(times).all()
            or np.any(times < 0)):
        raise ValueError('Invalid survival observations')
    if not len(times):
        return {'status': 'not_applicable', 'n': 0, 'time': [], 'survival': []}
    result = stats.ecdf(stats.CensoredData(uncensored=times[observed], right=times[~observed])).sf
    answer = {'status': 'ok', 'n': len(times), 'events': int(observed.sum()),
              'time': result.quantiles.tolist(), 'survival': result.probabilities.tolist(),
              'at_risk': [int(np.sum(times >= t)) for t in result.quantiles]}
    if confidence:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            interval = result.confidence_interval(confidence_level=.95)
        clean = lambda a: [float(x) if np.isfinite(x) else None for x in a]
        answer.update(low=clean(interval.low.probabilities), high=clean(interval.high.probabilities),
                      interval='pointwise Greenwood; undefined boundaries are null')
    return answer


def incidence(events):
    events = np.asarray(events, bool)
    if not len(events):
        return {'n': 0, 'events': 0, 'estimate': None, 'low': None, 'high': None}
    result = stats.binomtest(int(events.sum()), len(events))
    ci = result.proportion_ci(confidence_level=.95, method='exact')
    return {'n': len(events), 'events': int(events.sum()), 'estimate': float(events.mean()),
            'low': float(ci.low), 'high': float(ci.high)}


def mean_interval(values, *, seed, resamples=2000):
    values = np.asarray(values, float)
    positive_integer(resamples, 'resamples')
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Invalid bootstrap values')
    n = len(values)
    if not n:
        return {'n': 0, 'estimate': None, 'low': None, 'high': None, 'flags': ['not_applicable']}
    flags = ['small_sample'] if n < SETTINGS['small_sample_threshold'] else []
    if n == 1 or np.ptp(values) == 0:
        flags.append('degenerate_empirical_distribution')
        low = high = float(values.mean())
    else:
        # Each value is already a WITHIN-checkpoint paired contrast, averaged
        # over futures. Resampling these values retains all arm/future pairing.
        result = stats.bootstrap((values,), np.mean, vectorized=True, paired=True,
                                 n_resamples=resamples, batch=128, method='percentile',
                                 confidence_level=.95, random_state=np.random.default_rng(seed))
        low, high = map(float, result.confidence_interval)
    return {'n': n, 'estimate': float(values.mean()), 'low': low, 'high': high, 'flags': flags}


def factorial_effects(outcomes, *, seed=6002026, resamples=2000):
    """Input (reference checkpoints, eight arms, futures); never pool futures."""
    y = np.asarray(outcomes, float)
    if y.ndim != 3 or y.shape[1] != 8 or y.shape[2] < 1 or not np.isfinite(y).all():
        raise ValueError('Expected checkpoint x 8 arms x futures outcomes')
    y = y.mean(axis=2)
    result = {'arms': {}, 'versus_control': {}, 'factorial': {}}
    for i, arm in enumerate(ARMS):
        result['arms'][arm] = mean_interval(y[:, i], seed=seed+i, resamples=resamples)
        result['versus_control'][arm] = mean_interval(y[:, i]-y[:, 0], seed=seed+10+i, resamples=resamples)
    signs = np.array([[2*int(bit)-1 for bit in arm] for arm in ARMS])
    names = ('clear_queues', 'reset_shared_memory', 'redistribute_portfolios')
    for count in (1, 2, 3):
        for subset in combinations(range(3), count):
            weights = np.prod(signs[:, subset], axis=1) / 4
            key = ':'.join(names[i] for i in subset)
            result['factorial'][key] = mean_interval(y @ weights, seed=seed+100+sum(2**i for i in subset),
                                                    resamples=resamples)
    return result


def recovery_summary(arrays, *, threshold=100, tail_days=100):
    v, d = arrays['volume'], arrays['distinct_volume']
    if len(v) < tail_days:
        raise ValueError('Continuation must include its full primary tail window')
    first = lambda x: int(np.flatnonzero(x > 0)[0] + 1) if np.any(x > 0) else None
    periods = spells(v, threshold=threshold)
    # A zero spell starting on day 1 continues the selected checkpoint stall;
    # recurrent stalls must start after at least one execution in this branch.
    return {'tail_distinct_fraction': float(np.mean(d[-tail_days:] > 0)),
            'first_execution': first(v), 'first_distinct_exchange': first(d),
            'cumulative_volume': float(np.sum(v)), 'cumulative_distinct_volume': float(np.sum(d)),
            'execution_day_fraction': float(np.mean(v > 0)),
            'recurrent_confirmed_stalls': sum(p['onset'] > 1 and p['confirmation'] is not None for p in periods),
            'terminal_zero_spell': periods[-1]['duration'] if periods and periods[-1]['right_censored'] else 0}


def reference_summary(arrays, *, thresholds=(20, 100, 250)):
    v = arrays['volume']
    result = {'days': len(v), 'zero_volume_fraction': float(np.mean(v == 0)),
              'distinct_day_fraction': float(np.mean(arrays['distinct_volume'] > 0)),
              'positive_volume_zero_return_days': int(np.sum((v > 0) & (arrays['price'] == arrays['previous_price']))),
              'thresholds': {}}
    for threshold in thresholds:
        periods = spells(v, threshold=threshold)
        confirmed = [p for p in periods if p['confirmation'] is not None]
        result['thresholds'][str(threshold)] = {'first_onset': confirmed[0]['onset'] if confirmed else None,
            'first_confirmation': confirmed[0]['confirmation'] if confirmed else None,
            'confirmed_spells': len(confirmed), 'terminal_censored': bool(periods and periods[-1]['right_censored'])}
    return result

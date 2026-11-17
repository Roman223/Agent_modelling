"""Finite-sample MF-DFA with explicit zero-segment and scaling diagnostics.

The forward/backward segmentation follows Kantelhardt et al. (2002) and the
NumPy core of MFDFA 0.4.3 (MIT; see THIRD_PARTY_NOTICES.md). This adaptation uses
orthonormal polynomial projections, logarithmic moment accumulation, and the
q=0 limit. Input is a scalar series, never implicitly transformed into returns.
"""

from copy import deepcopy
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

SETTINGS = {
    'schema_version': 1,
    'q': [float(x) for x in np.arange(-3, 3.01, 0.5)],
    'orders': [1, 2],
    'minimum_scale': 16, 'scale_octave_step': 0.25, 'maximum_scale_divisor': 8,
    'common_maximum_scale': 128, 'minimum_fit_scales': 12, 'weak_r_squared': 0.95,
    'numerical_zero_factor': 64,
    'replicates': 99, 'chunk_size': 25, 'analysis_seed': 20261117,
    'iaaft_iterations': 1000, 'iaaft_max_spectral_error': 0.01,
    'garch_burn_in': 2048, 'garch_max_iterations': 2000,
    'simulation_lengths': [1024, 2048, 4096, 8192], 'index_lengths': [1024, 2048],
    'primary_length': 2048,
    'window': 'end-aligned daily log returns, including no-trade days',
    'normalization': 'demean and divide by population standard deviation',
}


def validate_settings(settings):
    if set(settings) != set(SETTINGS) or settings['schema_version'] != 1:
        raise ValueError('Unsupported or incomplete multifractal settings')
    q = np.asarray(settings['q'], dtype=float)
    if (q.ndim != 1 or len(q) < 3 or not np.isfinite(q).all()
            or not np.all(np.diff(q) > 0) or not np.any(q == 0)):
        raise ValueError('q must be finite, strictly increasing, and include zero')
    if settings['orders'] != [1, 2]:
        raise ValueError('The protocol requires detrending orders 1 and 2')
    for name in ('replicates', 'chunk_size', 'minimum_scale', 'maximum_scale_divisor',
                 'common_maximum_scale', 'minimum_fit_scales', 'iaaft_iterations',
                 'garch_burn_in', 'garch_max_iterations', 'primary_length', 'analysis_seed',
                 'numerical_zero_factor'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if settings['chunk_size'] > 25:
        raise ValueError('At most 25 control realizations per chunk')
    for name in ('scale_octave_step', 'weak_r_squared', 'iaaft_max_spectral_error'):
        if not np.isfinite(settings[name]) or not 0 < settings[name] <= 1:
            raise ValueError(f'Invalid {name}')
    for name in ('simulation_lengths', 'index_lengths'):
        lengths = settings[name]
        if not lengths or len(set(lengths)) != len(lengths) or any(type(n) is not int or n < 32 for n in lengths):
            raise ValueError(f'Invalid {name}')
        if settings['primary_length'] not in lengths:
            raise ValueError('Primary window must occur in both length lists')
    return settings


def scales_for(length, settings=None):
    settings = SETTINGS if settings is None else settings
    lo, hi = settings['minimum_scale'], length / settings['maximum_scale_divisor']
    if hi < lo:
        return np.array([], dtype=int)
    return np.unique(np.rint(2 ** np.arange(np.log2(lo), np.log2(hi) + 1e-10,
                                          settings['scale_octave_step'])).astype(int))


@lru_cache(maxsize=128)
def polynomial_basis(scale, order):
    x = np.linspace(-1., 1., scale)
    basis = np.linalg.qr(np.polynomial.polynomial.polyvander(x, order))[0]
    basis.flags.writeable = False
    return basis


def segment_variances(values, scales, order, zero_factor=64):
    """Return variances and diagnostics, retaining every segment, including zeros."""
    x = np.asarray(values, dtype=float)
    y = np.cumsum(x - np.mean(x))
    n = len(x)
    answer = []
    for scale in scales:
        scale = int(scale)
        count = n // scale
        segments = np.concatenate((y[:count*scale].reshape(count, scale),
                                   y[n % scale:].reshape(count, scale)))
        raw = np.concatenate((x[:count*scale].reshape(count, scale),
                              x[n % scale:].reshape(count, scale)))
        # A constant sequence of increments makes a linear profile. Identify it
        # from the original increments, before cancellation can create residue.
        exact_zero = np.all(raw[:, 1:] == raw[:, 1:2], axis=1)
        segments = segments - segments[:, :1]
        basis = polynomial_basis(scale, order)
        residual = segments - (segments @ basis) @ basis.T
        variance = np.mean(residual**2, axis=1)
        floor = (zero_factor*np.finfo(float).eps
                 * np.maximum(1., np.sqrt(np.mean(segments**2, axis=1))))**2
        numerical_zero = (variance <= floor) & ~exact_zero
        variance[exact_zero | numerical_zero] = 0.
        answer.append((variance, {'segments_per_direction': count,
                                 'zero_segments': int(exact_zero.sum()),
                                 'numerically_degenerate_segments': int(numerical_zero.sum()),
                                 'duplicate_directions': n % scale == 0}))
    return answer


def log_fluctuations(variance, q):
    """Natural log F_q; nonpositive moments of zero segments remain undefined."""
    positive = variance > 0
    result = np.full(len(q), np.nan)
    if not positive.any():
        return result
    logv = np.log(variance[positive])
    for i, power in enumerate(q):
        if power <= 0 and not positive.all():
            continue
        result[i] = (.5*np.mean(logv) if power == 0 else
                     (logsumexp(power*.5*logv) - np.log(len(variance))) / power)
    return result


def fit_scaling(scales, logf, q, settings):
    h, r2, status = np.full(len(q), np.nan), np.full(len(q), np.nan), []
    residuals = np.full_like(logf, np.nan)
    for j in range(len(q)):
        if len(scales) < settings['minimum_fit_scales']:
            status.append('insufficient_scales')
        elif not np.isfinite(logf[:, j]).all():
            status.append('undefined_fluctuations')
        else:
            slope, intercept = np.polyfit(np.log(scales), logf[:, j], 1)
            residuals[:, j] = logf[:, j] - (slope*np.log(scales) + intercept)
            total = np.sum((logf[:, j] - logf[:, j].mean())**2)
            h[j] = slope
            r2[j] = 1 - np.sum(residuals[:, j]**2)/total if total > 0 else 0.
            status.append('ok' if r2[j] >= settings['weak_r_squared'] else 'weak_scaling')
    tau = q*h - 1
    complete = np.isfinite(h).all()
    alpha = np.gradient(tau, q, edge_order=2) if complete else np.full(len(q), np.nan)
    spectrum = q*alpha - tau
    delta_h = float(h[0] - h[-1]) if np.isfinite(h[[0, -1]]).all() else None
    return {'h': h.tolist(), 'r_squared': r2.tolist(), 'status_by_q': status,
            'log_residuals': residuals.tolist(), 'tau': tau.tolist(), 'alpha': alpha.tolist(),
            'spectrum': spectrum.tolist(), 'delta_h': delta_h,
            'spectrum_width': float(np.ptp(alpha)) if complete else None,
            'nonconcave_tau': bool(np.any(np.diff(alpha) > 1e-8)) if complete else None,
            'scales': scales.tolist(),
            'scale_decades': float(np.log10(scales[-1]/scales[0])) if len(scales) > 1 else 0.,
            'status': ('undefined' if not complete else
                       'weak_scaling' if 'weak_scaling' in status else 'ok')}


def analyze_series(returns, *, settings=None, seed=None):
    """Analyze one scalar series; deterministic and independent of all RNG streams."""
    settings = deepcopy(SETTINGS if settings is None else settings)
    validate_settings(settings)
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1:
        raise ValueError('Expected a one-dimensional return series')
    result = {'observations': len(x), 'q': settings['q'], 'orders': {}, 'seed': seed,
              'normalization': settings['normalization']}
    if not len(x) or not np.isfinite(x).all():
        return dict(result, status='non_finite_input' if len(x) else 'insufficient_input')
    sd = float(np.std(x))
    result.update(return_mean=float(np.mean(x)), return_sd=sd,
                  zero_return_fraction=float(np.mean(x == 0)))
    if np.all(x == x[0]) or sd == 0:
        return dict(result, status='constant_series')
    if not np.isfinite(sd):
        return dict(result, status='non_finite_input')
    scales, q = scales_for(len(x), settings), np.array(settings['q'])
    if len(scales) < settings['minimum_fit_scales']:
        return dict(result, status='insufficient_scales')
    normalized = (x - np.mean(x)) / sd
    for order in settings['orders']:
        segments = segment_variances(normalized, scales, order, settings['numerical_zero_factor'])
        logf = np.array([log_fluctuations(v, q) for v, _ in segments])
        common = scales <= settings['common_maximum_scale']
        result['orders'][str(order)] = {
            'log_fluctuations': logf.tolist(), 'scales': scales.tolist(),
            'segment_diagnostics': [info for _, info in segments],
            'fit': fit_scaling(scales, logf, q, settings),
            'common_fit': fit_scaling(scales[common], logf[common], q, settings)}
    return dict(result, status='complete')

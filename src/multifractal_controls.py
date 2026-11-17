"""Seeded shuffled, IAAFT and Student-t GARCH controls for Task 7.

IAAFT adapts the NumPy rank/spectral projection in pyunicorn 0.8.0 (BSD-3-Clause).
See THIRD_PARTY_NOTICES.md. Only the original amplitude values are returned.
"""

import warnings

import numpy as np

from src.experiment_artifacts import digest
from src.multifractal import SETTINGS


def generator(identity, family, replicate, base_seed=20261117):
    token = digest([base_seed, identity, family, int(replicate)])
    words = [int(token[i:i+8], 16) for i in range(0, len(token), 8)]
    return np.random.default_rng(np.random.SeedSequence(words)), token


def iaaft(values, rng, settings=None):
    settings = SETTINGS if settings is None else settings
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all() or np.all(x == x[0]):
        return None, {'status': 'constant_or_invalid_input'}
    sorted_x = np.sort(x)
    target = np.abs(np.fft.rfft(x - x.mean()))
    norm = np.linalg.norm(target[1:])
    if norm == 0:
        return None, {'status': 'constant_or_invalid_input'}
    current = rng.permutation(sorted_x)
    best, best_error, stable = current.copy(), float('inf'), False
    for iteration in range(settings['iaaft_iterations']):
        phases = np.exp(1j*np.angle(np.fft.rfft(current - current.mean())))
        projected = np.fft.irfft(target*phases, n=len(x))
        ranks = np.argsort(projected, kind='stable')
        updated = np.empty_like(x)
        updated[ranks] = sorted_x
        error = float(np.linalg.norm(np.abs(np.fft.rfft(updated - updated.mean()))[1:]
                                     - target[1:]) / norm)
        if error < best_error:
            best, best_error = updated.copy(), error
        stable = np.array_equal(current, updated)
        if stable:
            break
        current = updated
    return best, {'status': 'ok' if best_error <= settings['iaaft_max_spectral_error'] else 'spectral_mismatch',
                  'relative_spectral_error': best_error, 'iterations': iteration+1,
                  'rank_stable': stable, 'exact_amplitudes': bool(np.array_equal(np.sort(best), sorted_x))}


def fit_garch(values, settings=None):
    settings = SETTINGS if settings is None else settings
    x = np.asarray(values, dtype=float)
    if len(x) < 32 or not np.isfinite(x).all() or np.std(x) == 0:
        return {'status': 'constant_or_invalid_input'}
    from arch import arch_model
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            fit = arch_model(100*x, mean='Constant', vol='GARCH', p=1, o=0, q=1,
                             dist='StudentsT', rescale=False).fit(
                                 disp='off', update_freq=0,
                                 options={'maxiter': settings['garch_max_iterations'], 'ftol': 1e-8})
            params = {str(k): float(v) for k, v in fit.params.items()}
            admissible = (np.isfinite(list(params.values())).all() and params['omega'] > 0
                          and params['alpha[1]'] >= 0 and params['beta[1]'] >= 0
                          and params['alpha[1]'] + params['beta[1]'] < 1 and params['nu'] > 2)
            result = {'status': 'ok' if fit.convergence_flag == 0 and admissible else 'unavailable_fit',
                      'parameters': params, 'parameter_order': list(fit.params.index),
                      'convergence_flag': int(fit.convergence_flag), 'admissible': bool(admissible),
                      'units': 'percentage log returns', 'log_likelihood': float(fit.loglikelihood)}
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            result = {'status': 'unavailable_fit', 'reason': f'{type(error).__name__}: {error}'}
    result['warnings'] = [str(w.message) for w in caught]
    return result


def simulate_garch(fit, length, rng, settings=None):
    from arch.univariate import ConstantMean, GARCH, StudentsT
    settings = SETTINGS if settings is None else settings
    if fit['status'] != 'ok':
        return None
    model = ConstantMean(None, volatility=GARCH(p=1, o=0, q=1), distribution=StudentsT(seed=rng))
    parameters = [fit['parameters'][name] for name in fit['parameter_order']]
    return model.simulate(parameters, nobs=length, burn=settings['garch_burn_in'])['data'].to_numpy()/100


def control(values, family, identity, replicate, *, fit=None, settings=None):
    settings = SETTINGS if settings is None else settings
    rng, token = generator(identity, family, replicate, settings['analysis_seed'])
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or not len(x) or np.std(x) == 0:
        return None, {'status': 'constant_or_invalid_input', 'seed_token': token}
    if family == 'shuffled':
        return rng.permutation(x), {'status': 'ok', 'seed_token': token}
    if family == 'iaaft':
        values, diagnostic = iaaft(x, rng, settings)
        return values, dict(diagnostic, seed_token=token)
    if family == 'garch':
        values = simulate_garch(fit, len(x), rng, settings)
        return values, {'status': 'ok' if values is not None else 'unavailable_fit', 'seed_token': token}
    raise ValueError(f'Unknown control family: {family}')

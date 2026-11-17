"""Known-answer validation, performed before market multifractal analysis."""

from copy import deepcopy

import numpy as np

from src.multifractal import SETTINGS, analyze_series, log_fluctuations
from src.multifractal_controls import control, fit_garch, generator, iaaft, simulate_garch


def lognormal_cascade(seed, levels=15, intermittency=0.1):
    """Independent binary lognormal weights, E[W]=1/2; normalized total mass.

    sigma²=lambda²*log(2), tau(q)=q-1-lambda²*q*(q-1)/2. The selected q range
    is inside the theoretical moment branch for lambda²=0.1. These are measure
    increments, not prices or an empirical asset-return generator.
    """
    rng = np.random.default_rng(seed)
    variance = intermittency*np.log(2.)
    weights = np.ones(1)
    for _ in range(levels):
        weights = (weights[:, None]*rng.lognormal(-np.log(2.)-variance/2,
                                                  np.sqrt(variance), (len(weights), 2))).ravel()
    return weights / np.mean(weights)


def validate_estimators(settings=None):
    from MFDFA import MFDFA, fgn
    settings = deepcopy(SETTINGS if settings is None else settings)
    checks, evidence = {}, {}
    state = np.random.get_state()
    try:
        rng = np.random.default_rng(700000)
        x = rng.normal(size=4096)
        answer = analyze_series(x, settings=settings)
        q = np.array(settings['q'])
        nonzero = q != 0
        differential = []
        for order in settings['orders']:
            part = answer['orders'][str(order)]
            _, expected = MFDFA((x-x.mean())/x.std(), np.array(part['scales']),
                                order=order, q=q[nonzero])
            actual = np.exp(np.array(part['log_fluctuations'])[:, nonzero])
            differential.append(float(np.max(np.abs(actual-expected)/expected)))
        checks['reference_MFDFA'] = max(differential) <= 1e-10
        evidence['reference_relative_errors'] = differential
        v = np.array([1., 4., 16.])
        limits = log_fluctuations(v, np.array([-1e-5, 0., 1e-5]))
        checks['q_zero_limit'] = bool(abs(limits[1]-np.log(2.)) < 1e-12 and
                                     abs(limits[[0, 2]].mean()-limits[1]) < 1e-8)
        scaled = analyze_series(100*x+2, settings=settings)
        checks['scale_invariance'] = bool(np.allclose(answer['orders']['1']['fit']['h'],
                                                     scaled['orders']['1']['fit']['h'], atol=1e-10))
        checks['constant'] = analyze_series(np.zeros(2048), settings=settings)['status'] == 'constant_series'
        checks['non_finite'] = analyze_series([0., np.nan], settings=settings)['status'] == 'non_finite_input'
        mixed = analyze_series(np.r_[rng.normal(size=1024), np.zeros(1024)], settings=settings)
        checks['stalls_preserved'] = (mixed['orders']['1']['fit']['delta_h'] is None and
                                     mixed['orders']['1']['fit']['spectrum_width'] is None)
        for hurst in (0.3, 0.5, 0.7):
            curves = []
            for repeat in range(8):
                np.random.seed(700100 + int(100*hurst)*10 + repeat)
                values = fgn(32768, float(hurst))
                curves.append(analyze_series(values, settings=settings)['orders']['1']['fit']['h'])
            median = np.median(curves, axis=0)
            error = float(np.max(np.abs(median-hurst)))
            h2_error = float(abs(median[np.flatnonzero(q == 2)[0]] - hurst))
            checks[f'fgn_H_{hurst}'] = error < .12 and h2_error < .08
            evidence[f'fgn_H_{hurst}'] = {'h_curves': curves, 'median': median.tolist(),
                                         'maximum_median_error': error, 'h2_error': h2_error}
        curves = [analyze_series(lognormal_cascade(701000+i), settings=settings)['orders']['1']['fit']['h']
                  for i in range(8)]
        expected = 1 - .1/2*(q-1)
        error = float(np.max(np.abs(np.median(curves, axis=0)-expected)))
        checks['lognormal_cascade'] = error < .10
        evidence['lognormal_cascade'] = {'h_curves': curves, 'expected_h': expected.tolist(),
                                          'maximum_median_error': error, 'intermittency': .1,
                                          'levels': 15, 'seeds': list(range(701000, 701008))}
        surrogate, diagnostics = iaaft(x[:2048], np.random.default_rng(702000), settings)
        checks['iaaft_amplitudes'] = np.array_equal(np.sort(surrogate), np.sort(x[:2048]))
        checks['iaaft_spectrum'] = diagnostics['status'] == 'ok'
        evidence['iaaft'] = diagnostics
        tied = np.round(x[:2048], 1)
        surrogate, _ = iaaft(tied, np.random.default_rng(702001), settings)
        checks['iaaft_ties'] = np.array_equal(np.sort(surrogate), np.sort(tied))
        shuffled, _ = control(x, 'shuffled', 'validation', 0, settings=settings)
        checks['shuffle'] = np.array_equal(np.sort(shuffled), np.sort(x))
        known = {'status': 'ok', 'parameters': {'mu': 0., 'omega': .05, 'alpha[1]': .05,
                                                'beta[1]': .9, 'nu': 8.},
                 'parameter_order': ['mu', 'omega', 'alpha[1]', 'beta[1]', 'nu']}
        a = simulate_garch(known, 8192, generator('validation', 'garch', 0)[0], settings)
        b = simulate_garch(known, 8192, generator('validation', 'garch', 0)[0], settings)
        checks['garch_reproducibility'] = np.array_equal(a, b)
        checks['garch_known_variance'] = bool(.6 < np.var(100*a) < 1.4)
        fitted = fit_garch(a, settings)
        checks['garch_fit'] = fitted['status'] == 'ok'
        checks['garch_degenerate'] = fit_garch(np.zeros(2048), settings)['status'] != 'ok'
        evidence['garch'] = {'variance_percentage_units': float(np.var(100*a)), 'fitted': fitted}
    finally:
        np.random.set_state(state)
    return {'passed': all(checks.values()), 'checks': {k: bool(v) for k, v in checks.items()},
            'evidence': evidence, 'settings': settings,
            'interpretation': 'Known-answer numerical validation; not market evidence'}

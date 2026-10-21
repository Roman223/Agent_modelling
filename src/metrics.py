"""Legacy-compatible nonlinear estimators with optional structured diagnostics."""

from dataclasses import dataclass
from importlib.metadata import version
import inspect

import numpy as np
from nolds import hurst_rs, lyap_r, sampen


@dataclass(frozen=True)
class MetricResult:
    """One estimator invocation, including failures without a second invocation."""

    value: float
    status: str
    reason: str | None
    estimator: str
    parameters: dict
    library_defaults: dict
    input_count: int
    return_count: int | None
    nolds_version: str


def _log_returns(price_series):
    """Preserve the notebook's NaN filtering and return calculation."""
    log_returns = np.diff(np.log(price_series))
    return log_returns[~np.isnan(log_returns)]


def _defaults(estimator, supplied):
    defaults = {}
    for name, parameter in inspect.signature(estimator).parameters.items():
        if name in supplied or parameter.default is inspect.Parameter.empty:
            continue
        value = parameter.default
        if callable(value):
            value = f'{value.__module__}.{value.__name__}'
        defaults[name] = value
    return defaults


def metric_result(name, price_series, *, emb_dim=None, lag=1):
    """Calculate hurst, lyapunov, or sampen with the legacy preprocessing rules.

    Defaults not explicitly supplied to nolds are recorded separately. For short
    price series, return_count is None because legacy code skips preprocessing.
    Input conversion errors still propagate, as in the scalar API.
    """
    if name == 'hurst':
        estimator, label, min_prices, min_returns = hurst_rs, 'nolds.hurst_rs (R/S)', 30, 10
        parameters = {}
    elif name == 'lyapunov':
        estimator, label, min_prices, min_returns = lyap_r, 'nolds.lyap_r (Rosenstein)', 50, 30
        parameters = {'emb_dim': 5 if emb_dim is None else emb_dim, 'lag': lag}
    elif name == 'sampen':
        estimator, label, min_prices, min_returns = sampen, 'nolds.sampen (Sample Entropy, nats)', 100, 50
        parameters = {'emb_dim': 3 if emb_dim is None else emb_dim}
    else:
        raise ValueError(f'Unknown metric: {name}')
    defaults = _defaults(estimator, parameters)
    return_count = None

    def result(value, status, reason=None):
        return MetricResult(value, status, reason, label, dict(parameters), defaults,
                            len(price_series), return_count, version('nolds'))

    if len(price_series) < min_prices:
        return result(np.nan, 'insufficient_input', f'Need at least {min_prices} price observations')
    log_returns = _log_returns(price_series)
    return_count = len(log_returns)
    if return_count < min_returns:
        return result(np.nan, 'insufficient_input', f'Need at least {min_returns} log-returns after NaN filtering')
    if name == 'sampen':
        tolerance = 0.2 * np.std(log_returns)
        if tolerance == 0:
            tolerance = 0.01
        parameters['tolerance'] = float(tolerance)
        defaults.pop('tolerance', None)
    try:
        value = estimator(log_returns, **parameters)
    except Exception as error:
        return result(np.nan, 'estimator_error', f'{type(error).__name__}: {error}')
    if not np.isfinite(value):
        return result(value, 'non_finite', f'Estimator returned {value}')
    return result(value, 'ok')


def calculate_hurst_exponent(price_series):
    """Estimate Hurst with R/S analysis; retain the historical scalar return."""
    return metric_result('hurst', price_series).value


def calculate_lyapunov_exponent(price_series, emb_dim=5, lag=1):
    """Estimate Rosenstein's Lyapunov exponent; retain the historical scalar return."""
    return metric_result('lyapunov', price_series, emb_dim=emb_dim, lag=lag).value


def calculate_sample_entropy(price_series, emb_dim=3):
    """Estimate Sample Entropy (nats), using the notebook's tolerance convention."""
    return metric_result('sampen', price_series, emb_dim=emb_dim).value


calculate_kolmogorov_entropy = calculate_sample_entropy

"""Legacy-compatible nonlinear estimators computed on price log-returns."""

import numpy as np
from nolds import hurst_rs, lyap_r, sampen


def _log_returns(price_series):
    """Preserve the notebook's NaN filtering and return calculation."""
    log_returns = np.diff(np.log(price_series))
    return log_returns[~np.isnan(log_returns)]


def calculate_hurst_exponent(price_series):
    """Estimate the Hurst exponent with R/S analysis; return NaN if unavailable."""
    if len(price_series) < 30:
        return np.nan
    log_returns = _log_returns(price_series)
    if len(log_returns) < 10:
        return np.nan
    try:
        return hurst_rs(log_returns)
    except Exception:
        return np.nan


def calculate_lyapunov_exponent(price_series, emb_dim=5, lag=1):
    """Estimate Rosenstein's Lyapunov exponent; return NaN if unavailable."""
    if len(price_series) < 50:
        return np.nan
    log_returns = _log_returns(price_series)
    if len(log_returns) < 30:
        return np.nan
    try:
        return lyap_r(log_returns, emb_dim=emb_dim, lag=lag)
    except Exception:
        return np.nan


def calculate_sample_entropy(price_series, emb_dim=3):
    """Estimate Sample Entropy (nats), using the notebook's tolerance convention."""
    if len(price_series) < 100:
        return np.nan
    log_returns = _log_returns(price_series)
    if len(log_returns) < 50:
        return np.nan
    tolerance = 0.2 * np.std(log_returns)
    if tolerance == 0:
        tolerance = 0.01
    try:
        return sampen(log_returns, emb_dim=emb_dim, tolerance=tolerance)
    except Exception:
        return np.nan


# Historical import compatibility; this estimator has always been Sample Entropy.
calculate_kolmogorov_entropy = calculate_sample_entropy

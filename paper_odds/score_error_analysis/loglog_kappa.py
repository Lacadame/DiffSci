"""Global log-magnitude least squares for the draft's variance-clock exponent.

log |f| = log |epsilon| - kappa*log(V/V_ref). Sign changes are diagnosed, not
removed from the data by keeping only one sign. This is a magnitude envelope.
"""

from __future__ import annotations

import numpy as np

from .score_error_modes import integration_weights


def fit_loglog(distance, profile, *, relative_cutoff=1e-6, weighting='log_variance'):
    """Fit one global slope; no numerical derivatives, smoothing, or kappa bounds.

    distance is log(V/V_ref), already a logarithm: do not take log(distance).
    The default weights approximate uniform measure in log variance on the
    existing hybrid grid. 'points' gives ordinary unweighted least squares on
    the saved nodes. Omitted near-zero nodes retain their original quadrature
    weights; gaps are not bridged by recomputing trapezoidal weights.
    """
    d, y = np.asarray(distance, float), np.asarray(profile, float)
    if d.ndim != 1 or len(d) < 3 or d.shape != y.shape:
        raise ValueError('Require matching one-dimensional profiles and at least three samples')
    if not np.isfinite(d).all() or not np.isfinite(y).all() or np.any(np.diff(d) <= 0):
        raise ValueError('Require finite samples on a strictly increasing variance clock')
    if not np.isfinite(relative_cutoff) or not 0 <= relative_cutoff < 1:
        raise ValueError('relative_cutoff must be in [0,1)')
    if weighting not in ('log_variance', 'points'):
        raise ValueError('weighting must be log_variance or points')
    clock_weights = integration_weights(d)
    peak = float(np.max(abs(y)))
    mask = (abs(y) > relative_cutoff*peak) & (y != 0)
    nonzero = y[y != 0]
    sign_changes = int(np.count_nonzero(nonzero[1:]*nonzero[:-1] < 0))
    result = dict(valid=False, kappa=np.nan, amplitude=np.nan, log_amplitude=np.nan,
                  slope=np.nan, log_r2=np.nan, log_rmse=np.nan, magnitude_relative_rmse=np.nan,
                  sign_changes=sign_changes, mixed_sign=bool(np.any(y > 0) and np.any(y < 0)),
                  retained_count=int(mask.sum()), excluded_count=int((~mask).sum()),
                  retained_clock_fraction=float(clock_weights[mask].sum()/clock_weights.sum()),
                  relative_cutoff=relative_cutoff, absolute_cutoff=relative_cutoff*peak, weighting=weighting)
    if mask.sum() < 3:
        return result
    weights = clock_weights[mask] if weighting == 'log_variance' else np.ones(mask.sum())
    weights = weights/weights.sum()
    x, z = d[mask], np.log(abs(y[mask]))
    x_mean, z_mean = weights@x, weights@z
    x_variance = weights@(x-x_mean)**2
    if x_variance <= 0:
        return result
    slope = float((weights@((x-x_mean)*(z-z_mean)))/x_variance)
    intercept = float(z_mean-slope*x_mean)
    log_residual = z-(intercept+slope*x)
    mse = float(weights@log_residual**2)
    variance = float(weights@(z-z_mean)**2)
    with np.errstate(over='ignore', under='ignore'):
        amplitude = float(np.exp(intercept))
        magnitude_fit = np.exp(intercept+slope*d)
        relative_rmse = float(np.sqrt((clock_weights@(abs(y)-magnitude_fit)**2)/(clock_weights@y**2)))
    result.update(valid=bool(np.isfinite(amplitude) and amplitude > 0), kappa=-slope,
                  amplitude=amplitude, log_amplitude=intercept, slope=slope,
                  log_r2=float(1-mse/variance) if variance > 1e-28 else np.nan,
                  log_rmse=float(np.sqrt(mse)), magnitude_relative_rmse=relative_rmse)
    return result

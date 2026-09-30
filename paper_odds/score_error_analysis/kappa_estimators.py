"""Sensitivity estimators for signed profiles on the log-variance clock.

The original pointwise least-squares estimator remains unchanged. Cumulative
fits use the ODE mode kernels: 1 for shape, exp(-d/2) for mean.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar

from .score_error_modes import integration_weights


def check_profile(distance, profile):
    d, y = np.asarray(distance, float), np.asarray(profile, float)
    if d.ndim != 1 or len(d) < 3 or d.shape != y.shape or d[0] != 0:
        raise ValueError('Profiles require at least three samples starting at d=0')
    if not np.isfinite(d).all() or not np.isfinite(y).all() or np.any(np.diff(d) <= 0):
        raise ValueError('Profiles must be finite on a strictly increasing clock')
    return d, y


def primitive_at(distance, profile, points):
    """Exact integral of the piecewise-linear supplied profile from 0 to points."""
    d, y = check_profile(distance, profile)
    points = np.asarray(points, float)
    if np.any(points < 0) or np.any(points > d[-1]):
        raise ValueError('Primitive evaluation outside the supplied clock')
    integral = cumulative_trapezoid(y, d, initial=0.)
    index = np.clip(np.searchsorted(d, points, side='right')-1, 0, len(d)-2)
    delta = points-d[index]
    slope = (y[index+1]-y[index])/(d[index+1]-d[index])
    return integral[index]+delta*y[index]+.5*delta**2*slope


def smooth_profile(distance, profile, bandwidth, points_per_bandwidth=8):
    """Area-conserving Gaussian smoothing on uniformly spaced d bins.

    Conservatively rebin the piecewise-linear signed profile before smoothing;
    interpolating sparse point samples onto a uniform grid can erase a narrow
    terminal peak. Reflection prevents signed area leaking through either end.
    bandwidth is the Gaussian standard deviation in log-variance units.
    """
    d, y = check_profile(distance, profile)
    if not np.isfinite(bandwidth) or bandwidth <= 0 or points_per_bandwidth < 4:
        raise ValueError('Positive bandwidth and at least four points per bandwidth required')
    count = max(32, int(np.ceil(points_per_bandwidth*d[-1]/bandwidth)))
    edges = np.linspace(0, d[-1], count+1)
    width = edges[1]
    values = np.diff(primitive_at(d, y, edges))/width
    smoothed = gaussian_filter1d(values, bandwidth/width, mode='reflect', truncate=5.)
    return dict(distance=(edges[:-1]+edges[1:])/2, profile=smoothed,
                weights=np.full(count, width), edges=edges,
                original_area=float(np.trapz(y, d)), smoothed_area=float(width*smoothed.sum()))


def cumulative_basis(rate, points):
    points = np.asarray(points, float)
    return points.copy() if rate == 0 else -np.expm1(-rate*points)/rate


def fit_cumulative(distance, profile, *, mode, bounds=(-4., 1e6), loss_tolerance=.01):
    """Fit the cumulative signed ODE response, not derivatives or log magnitudes.

    Minimize integral [int_0^d w_mode(s) y(s) ds - amplitude*F(k+beta,d)]^2 dd,
    with beta=0 (shape) or 1/2 (mean). The constant mean prefactor cancels in
    this mode-wise fit. The loss envelope is an objective-sensitivity diagnostic,
    NOT a statistical confidence interval. An envelope reaching the upper search
    bound indicates that temporal localization is not resolved by this objective.
    """
    d, y = check_profile(distance, profile)
    if mode not in ('shape', 'mean'):
        raise ValueError('mode must be shape or mean')
    if not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or loss_tolerance <= 0:
        raise ValueError('Finite increasing bounds and positive loss tolerance required')
    beta = 0. if mode == 'shape' else .5
    target = cumulative_trapezoid(np.exp(-beta*d)*y, d, initial=0.)
    weights = integration_weights(d); weights /= weights.sum()
    energy = float(weights@target**2)
    if energy <= np.finfo(float).tiny:
        return dict(kappa=np.nan, amplitude=0., relative_rmse=0., at_bound=False,
                    loss_envelope_low=np.nan, loss_envelope_high=np.nan,
                    upper_unresolved=True, cancellation_ratio=0.)

    def solve(k):
        basis = cumulative_basis(k+beta, d)
        amplitude = (weights@(basis*target))/(weights@basis**2)
        mse = float(weights@(target-amplitude*basis)**2)
        return mse, float(amplitude)

    grid = np.unique(np.r_[bounds, np.linspace(max(bounds[0], -4.), min(bounds[1], 8.), 241),
                           np.sinh(np.linspace(*np.arcsinh(bounds), 481))])
    grid = grid[(grid >= bounds[0]) & (grid <= bounds[1])]
    scores = np.array([solve(k)[0] for k in grid])
    candidates = [(scores[0], grid[0]), (scores[-1], grid[-1])]
    for i in range(1, len(grid)-1):
        if scores[i] <= scores[i-1] and scores[i] <= scores[i+1] and (scores[i] < scores[i-1] or scores[i] < scores[i+1]):
            fit = minimize_scalar(lambda k: solve(k)[0], bounds=(grid[i-1], grid[i+1]),
                                  method='bounded', options={'xatol': 1e-10})
            candidates.append((fit.fun, fit.x))
    mse, kappa = min(candidates)
    plausible = grid[scores <= mse+loss_tolerance*energy]
    envelope = np.r_[plausible, kappa]
    absolute_area = np.trapz(np.exp(-beta*d)*abs(y), d)
    return dict(kappa=float(kappa), amplitude=solve(kappa)[1],
                relative_rmse=float(np.sqrt(mse/energy)),
                at_bound=bool(min(abs(kappa-bounds[0]), abs(kappa-bounds[1])) < 1e-5),
                loss_envelope_low=float(envelope.min()), loss_envelope_high=float(envelope.max()),
                upper_unresolved=bool(bounds[1] in plausible),
                cancellation_ratio=float(abs(target[-1])/absolute_area) if absolute_area else 0.)


def smoothed_mode_response(smoothed, gammas, *, mode):
    """Integrate the piecewise-constant smoothed bins against exact mode kernels."""
    gammas = np.asarray(gammas, float)
    if mode not in ('shape', 'mean') or np.any(gammas < 0):
        raise ValueError('Require a known mode and nonnegative gammas')
    rates = gammas if mode == 'shape' else (1+gammas)/2
    factors = 1+gammas if mode == 'shape' else (1+gammas)/2
    edges, y = smoothed['edges'], smoothed['profile']
    integrals = [np.diff(cumulative_basis(rate, edges))@y for rate in rates]
    return factors*np.asarray(integrals)

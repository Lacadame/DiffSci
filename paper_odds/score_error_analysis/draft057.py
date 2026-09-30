"""CPU-only response and odds calculations for the September 2026 draft.

The score coordinates are a=VC and u=Vb/sqrt(Vref), including a>=1.
All integrals stop at the measured positive-noise endpoint.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from .score_error_modes import integration_weights


def response_weights(distance, gammas):
    """Rows of quadrature weights for L_gamma and M_gamma, Eqs. 2.13–14."""
    d, g = np.asarray(distance, float), np.atleast_1d(np.asarray(gammas, float))
    w = integration_weights(d)
    if not np.isclose(d[0], 0) or not np.isfinite(g).all() or np.any(g < 0):
        raise ValueError('Use an endpoint-shifted distance and finite nonnegative gammas.')
    return ((1+g[:, None])*np.exp(-g[:, None]*d)*w,
            (1+g[:, None])/2*np.exp(-(1+g[:, None])*d/2)*w)


def responses(distance, a, u, gammas):
    ka, ku = response_weights(distance, gammas)
    L, M = np.asarray(a) @ ka.T, np.asarray(u) @ ku.T
    return dict(L=L, M=M, kl=L**2/4+M**2/2)


def bounded_change(h, h0):
    """(h-h0)/(h+h0); a double zero is undefined, not a win or tie."""
    h, h0 = np.broadcast_arrays(np.asarray(h, float), np.asarray(h0, float))
    if np.any(h < 0) or np.any(h0 < 0):
        raise ValueError('KL values must be nonnegative.')
    scale = np.maximum(h, h0)
    hs = np.divide(h, scale, out=np.zeros_like(h), where=scale > 0)
    bs = np.divide(h0, scale, out=np.zeros_like(h0), where=scale > 0)
    return np.divide(hs-bs, hs+bs, out=np.full_like(hs, np.nan), where=hs+bs > 0)


def bounded_from_log10(log_ratio):
    return np.tanh(np.log(10)*np.asarray(log_ratio)/2)


def two_amplitude_odds(delta_shape, delta_mean):
    """Independent unit Gaussian mode amplitudes; includes the 1/4, 1/2 KL factors.

    Inputs are L_gamma^2-L_0^2 and M_gamma^2-M_0^2. This is a strict
    improvement probability. Independent random signs alone do not give this law.
    """
    a, b = np.broadcast_arrays(np.asarray(delta_shape, float)/4,
                               np.asarray(delta_mean, float)/2)
    out = np.zeros_like(a)
    out[(a <= 0) & (b <= 0) & ((a < 0) | (b < 0))] = 1
    mask = (a < 0) & (b > 0)
    out[mask] = 2/np.pi*np.arctan(np.sqrt(-a[mask]/b[mask]))
    mask = (b < 0) & (a > 0)
    out[mask] = 2/np.pi*np.arctan(np.sqrt(-b[mask]/a[mask]))
    return out


def _bvn_cdf(h, k, rho):
    """Deterministic bivariate standard-normal CDF, including rank-one limits."""
    rho = float(np.clip(rho, -1, 1))
    if rho >= 1-1e-12:
        return float(ndtr(min(h, k)))
    if rho <= -1+1e-12:
        return float(max(0, ndtr(h)-ndtr(-k)))
    den = np.sqrt(1-rho*rho)
    value = quad(lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
                 * ndtr((k-rho*z)/den), -np.inf, h,
                 epsabs=1e-10, epsrel=1e-9)[0]
    return float(value)


def shape_gaussian_odds(mean, covariance):
    """P(Z_gamma^2 < Z_0^2), Theorem 7.1, with nonzero mean and singular limits."""
    mean, cov = np.asarray(mean, float), np.asarray(covariance, float)
    if mean.shape != (2,) or cov.shape != (2, 2):
        raise ValueError('Expected mean and covariance of [L0, Lgamma].')
    transform = np.array([[1., -1.], [1., 1.]])
    mu, c = transform @ mean, transform @ cov @ transform.T
    var = np.maximum(np.diag(c), 0)
    tol = max(float(np.max(np.abs(cov)))*1e-13, np.finfo(float).tiny)
    if np.all(var <= tol):
        return float(mu[0]*mu[1] > 0)
    for j in (0, 1):
        if var[j] <= tol:
            return float(ndtr(np.sign(mu[j])*mu[1-j]/np.sqrt(var[1-j]))) if mu[j] != 0 else 0.
    nu = mu/np.sqrt(var)
    rho = c[0, 1]/np.sqrt(var.prod())
    return float(np.clip(_bvn_cdf(*nu, rho)+_bvn_cdf(*(-nu), rho), 0, 1))


def gaussian_joint_odds(mean, covariance, *, seed=42, power=14, repeats=4):
    """Both-mode Gaussian quadratic-form probability (Sec. 7.4), scrambled Sobol.

    Vector order: [L0, Lgamma, M0, Mgamma]. The reported SE is across
    independent numerical scrambles, not across training runs or checkpoints.
    """
    mean, cov = np.asarray(mean, float), np.asarray(covariance, float)
    if mean.shape != (4,) or cov.shape != (4, 4) or repeats < 2:
        raise ValueError('Expected a four-dimensional Gaussian and >=2 scrambles.')
    eigenvalues, eigenvectors = np.linalg.eigh((cov+cov.T)/2)
    if eigenvalues.min() < -1e-10*max(np.max(np.abs(eigenvalues)), 1e-30):
        raise ValueError('Covariance is not positive semidefinite.')
    factor = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0))
    estimates = []
    for repeat in range(repeats):
        uniform = qmc.Sobol(4, scramble=True, seed=seed+repeat).random_base2(power)
        z = ndtri(np.clip(uniform, np.finfo(float).eps, 1-np.finfo(float).eps)) @ factor.T+mean
        change = (z[:, 1]**2-z[:, 0]**2)/4+(z[:, 3]**2-z[:, 2]**2)/2
        estimates.append(np.mean(change < 0))
    return dict(probability=float(np.mean(estimates)),
                numerical_se=float(np.std(estimates, ddof=1)/np.sqrt(repeats)),
                draws=repeats*2**power)


def fit_log_noise_gp(sigma, profiles, *, grid_points=257, max_lag_fraction=1/3):
    """Estimate m, population SD, and an OU correlation length in log sigma.

    Interpolate to a uniform log-noise grid before forming the ACF, to avoid
    overweighting the hybrid cache grid. Standardize pointwise, then pool
    products over checkpoints and positions. Fit exp(-lag/ell) by pair-count
    weighted LS through one-third of the observed log-noise span. Negative
    correlations remain in the fit diagnostics. This is descriptive calibration.
    """
    sigma, f = np.asarray(sigma, float), np.asarray(profiles, float)
    if f.ndim != 2 or f.shape[1] != len(sigma) or len(f) < 2 or not np.isfinite(f).all():
        raise ValueError('Need at least two finite checkpoint profiles.')
    x = np.log(sigma)
    integration_weights(x)
    xu = np.linspace(x[0], x[-1], grid_points)
    fu = np.stack([np.interp(xu, x, row) for row in f])
    mean, sd = f.mean(axis=0), f.std(axis=0)
    sdu = fu.std(axis=0)
    if np.any(sd == 0) or np.any(sdu == 0):
        raise ValueError('A stationary standardized GP requires positive pointwise SD.')
    z = (fu-fu.mean(axis=0))/sdu
    max_lag = int((grid_points-1)*max_lag_fraction)
    lag = np.arange(max_lag+1)*(xu[1]-xu[0])
    acf = np.array([1.] + [np.mean(z[:, :-k]*z[:, k:]) for k in range(1, max_lag+1)])
    weight = (grid_points-np.arange(1, max_lag+1)).astype(float)
    weight /= weight.sum()
    bounds = (np.log((xu[1]-xu[0])/20), np.log(100*(x[-1]-x[0])))
    objective = lambda log_length: float(np.sum(weight*(np.exp(-lag[1:]/np.exp(log_length))-acf[1:])**2))
    result = minimize_scalar(objective, bounds=bounds, method='bounded')
    if not result.success:
        raise RuntimeError(result.message)
    length = float(np.exp(result.x))
    return dict(mean=mean, sd=sd, length=length, lag=lag, acf=acf,
                acf_rmse=float(np.sqrt(result.fun)),
                length_at_bound=bool(min(result.x-bounds[0], bounds[1]-result.x) < .01),
                standardized=z, log_grid=xu)


def gp_response_distribution(distance, sigma, mean, amplitude, length, gammas, *, mode='a'):
    """Finite-grid projection of m + A*GP with OU covariance in log sigma."""
    if not np.isfinite(length) or length <= 0 or mode not in ('a', 'u'):
        raise ValueError('Use a positive finite length and mode a or u.')
    amplitude, mean = np.asarray(amplitude), np.asarray(mean)
    if np.any(amplitude < 0) or not np.isfinite(amplitude).all():
        raise ValueError('Amplitudes must be finite and nonnegative.')
    kernel = np.exp(-np.abs(np.log(sigma)[:, None]-np.log(sigma)[None, :])/length)
    weights = response_weights(distance, gammas)[0 if mode == 'a' else 1]
    weighted = weights*amplitude
    return weights @ mean, weighted @ kernel @ weighted.T


def affine_moments(distance, a, u, gammas, *, refinement=4):
    """Nonperturbative moments with exact constant-coefficient midpoint steps.

    Linear interpolation of measured a,u within each interval, refined before
    midpoint freezing. Converges at order two. This is a numerical integration
    of the exact moment equations, not an exact solve of time-varying profiles.
    Returns KL directions [q||p, p||q], r, v; the exact prior has r=0,v=1.
    """
    d = np.asarray(distance, float)
    integration_weights(d)
    a, u = np.asarray(a, float), np.asarray(u, float)
    if a.shape != u.shape or a.shape[-1] != len(d) or refinement < 1:
        raise ValueError('Profiles must match the clock and refinement must be positive.')
    g = np.atleast_1d(np.asarray(gammas, float))
    if not np.isfinite(a).all() or not np.isfinite(u).all() or np.any(g < 0):
        raise ValueError('Finite profiles and nonnegative gammas required.')
    r = np.zeros(a.shape[:-1]+(len(g),))
    v = np.ones_like(r)
    # exprel(z) avoids cancellation and handles zero drift.
    from scipy.special import exprel
    for i in range(len(d)-1, 0, -1):
        h = (d[i]-d[i-1])/refinement
        for j in range(refinement):
            t = (j+.5)/refinement
            at = ((1-t)*a[..., i]+t*a[..., i-1])[..., None]
            ut = ((1-t)*u[..., i]+t*u[..., i-1])[..., None]
            ar = -(1+g)/2*(1-at)
            av = -g+(1+g)*at
            r = np.exp(ar*h)*r+h*exprel(ar*h)*(1+g)/2*ut
            v = np.exp(av*h)*v+h*exprel(av*h)*g
    if np.any(v <= 0) or not np.isfinite(v).all():
        raise FloatingPointError('Invalid terminal variance.')
    delta = v-1
    logv = np.log1p(delta)
    kl = np.stack([.5*(delta-logv+r*r), .5*(logv-delta/v+r*r/v)], axis=-1)
    return np.maximum(kl, 0), r, v

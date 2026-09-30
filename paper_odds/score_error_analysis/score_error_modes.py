"""Spatial score-error projections and Gaussian phase coordinates.

Only the checkpoint inference module requires PyTorch. The numerical routines
here use NumPy/SciPy, and work with signed errors (not their absolute values).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, roots_hermitenorm


def integration_weights(t):
    """Trapezoidal weights for a strictly increasing one-dimensional grid."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Integration grid must contain at least two increasing finite points.")
    w = np.empty_like(t)
    w[0], w[-1] = (t[1] - t[0]) / 2, (t[-1] - t[-2]) / 2
    w[1:-1] = (t[2:] - t[:-2]) / 2
    return w


def affine_projection(x, error, weights=None, center=None):
    """Minimize E_w ||error - b - C(x-center)||² with a free intercept.

    Inputs have shape [sample, dimension]; C has [output, input] axes.
    Center defaults to the weighted sample mean. With a supplied population
    mean the intercept is fitted jointly, so finite-sample residuals still
    satisfy both normal equations. The energy decomposition is orthogonal
    about the weighted sample mean; the returned b is at the supplied center.
    """
    x, error = np.asarray(x, float), np.asarray(error, float)
    if x.ndim != 2 or error.shape != x.shape or len(x) < x.shape[1] + 1:
        raise ValueError("x and error must have matching [sample, dimension] shapes.")
    if not np.isfinite(x).all() or not np.isfinite(error).all():
        raise ValueError("Projection inputs must be finite.")
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, and have positive sum.")
    w = w / w.sum()
    mu = w @ x
    center = mu if center is None else np.asarray(center, float)
    if center.shape != (x.shape[1],) or not np.isfinite(center).all():
        raise ValueError("center must be a finite vector of the input dimension.")
    z = x - mu
    mean_error = w @ error
    # Column scaling avoids ill conditioning across diffusion noise scales.
    std = np.sqrt(w @ (z*z))
    if np.any(std <= np.finfo(float).eps):
        raise ValueError("Projection requires nonzero variance in every coordinate.")
    design = z / std
    coef, _, rank, _ = np.linalg.lstsq(
        design * np.sqrt(w[:, None]),
        (error - mean_error) * np.sqrt(w[:, None]), rcond=None,
    )
    if rank != x.shape[1]:
        raise ValueError("Projection covariance is rank deficient.")
    C = (coef / std[:, None]).T
    b = mean_error - C @ (mu - center)
    linear = z @ C.T
    residual = error - mean_error - linear
    covariance = (z.T * w) @ z
    d = x.shape[1]
    isotropic = np.trace(C) / d * np.eye(d)
    symmetric_traceless = (C + C.T) / 2 - isotropic
    skew = (C - C.T) / 2
    energy = lambda values: float(w @ np.sum(values**2, axis=-1))
    return dict(
        b=b, C=C, center=center, sample_mean=mu, covariance=covariance,
        C_isotropic=isotropic, C_symmetric_traceless=symmetric_traceless,
        C_skew=skew, total_energy=energy(error),
        mean_energy=float(mean_error @ mean_error), affine_energy=energy(linear),
        residual_energy=energy(residual),
        # These matrix submode energies need not add under anisotropic p_t.
        isotropic_energy=energy(z @ isotropic.T),
        symmetric_traceless_energy=energy(z @ symmetric_traceless.T),
        skew_energy=energy(z @ skew.T),
        residual_mean_norm=float(np.linalg.norm(w @ residual)),
        residual_cross_norm=float(np.linalg.norm((z.T * w) @ residual)),
    )


@dataclass(frozen=True)
class GaussianMixture1D:
    means: tuple = (-1.0, 0.1)
    scales: tuple = (0.2, 0.1)
    probabilities: tuple = (0.1, 0.9)

    def __post_init__(self):
        m, s, p = map(lambda a: np.asarray(a, float), (self.means, self.scales, self.probabilities))
        if m.ndim != 1 or m.size == 0 or s.shape != m.shape or p.shape != m.shape:
            raise ValueError("Mixture parameters must be nonempty matching vectors.")
        if not all(np.isfinite(a).all() for a in (m, s, p)) or np.any(s <= 0) or np.any(p <= 0):
            raise ValueError("Mixture parameters must be finite, with positive scales and weights.")
        if not np.isclose(p.sum(), 1):
            raise ValueError("Mixture probabilities must sum to one.")

    @property
    def mean(self):
        return float(np.dot(self.probabilities, self.means))

    @property
    def variance(self):
        return float(np.dot(self.probabilities, np.asarray(self.scales)**2
                            + (np.asarray(self.means) - self.mean)**2))

    def quadrature(self, sigma, order=256):
        """Positive weighted nodes integrating p_sigma exactly for polynomials.

        Gauss-Hermite quadrature is applied separately to each mixture component.
        At finite order the nonpolynomial network/score integrals are approximate.
        """
        sigma = np.asarray(sigma, float)
        if sigma.ndim != 1 or not np.isfinite(sigma).all() or np.any(sigma < 0) or order < 2:
            raise ValueError("Use a nonnegative finite sigma vector and order >= 2.")
        z, weights = roots_hermitenorm(order)
        std = np.sqrt(np.asarray(self.scales)[None, :]**2 + sigma[:, None]**2)
        x = np.asarray(self.means)[None, :, None] + std[:, :, None] * z[None, None, :]
        w = np.asarray(self.probabilities)[:, None] * weights[None, :] / np.sqrt(2*np.pi)
        return x.reshape(len(sigma), -1), w.ravel()

    def score(self, x, sigma):
        """Exact Gaussian-mixture score, with log-sum-exp responsibilities."""
        x, sigma = np.broadcast_arrays(np.asarray(x, float), np.asarray(sigma, float))
        var = np.asarray(self.scales)**2 + sigma[..., None]**2
        delta = x[..., None] - np.asarray(self.means)
        logits = np.log(self.probabilities) - 0.5*np.log(var) - delta**2/(2*var)
        responsibilities = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
        return np.sum(-responsibilities * delta / var, axis=-1)


def variance_clock(mixture, sigma_min=0.002, sigma_max=80.0, time_points=161, sigma_points=129):
    """Hybrid grid resolving both log variance and very small positive sigma.

    d = Lambda - ell = log(V(t)/V_ref) increases toward high noise. The
    reference endpoint is sigma_min, matching the saved evaluation's endpoint.
    """
    if not 0 < sigma_min < sigma_max or time_points < 3 or sigma_points < 3:
        raise ValueError("Require 0 < sigma_min < sigma_max and at least three grid points.")
    vref = mixture.variance + sigma_min**2
    horizon = np.log((mixture.variance + sigma_max**2) / vref)
    d_uniform = np.linspace(0, horizon, time_points)
    sigma_clock = np.sqrt(np.maximum(vref * np.expm1(d_uniform) + sigma_min**2, 0))
    sigma = np.unique(np.r_[sigma_clock[1:-1], np.geomspace(sigma_min, sigma_max, sigma_points)])
    V = mixture.variance + sigma**2
    d = np.log(V/vref)
    return dict(sigma=sigma, variance=V, reference_variance=vref,
                distance=d, ell=horizon-d, horizon=horizon)


def normalized_profiles(b, C, variance, reference_variance):
    """Gaussian tangent profiles, and exact Gaussian-closure parameters.

    Eq. (7): a_lin=VC, u_lin=Vb/sqrt(V_ref). Exactly, if 1-VC>0,
    log(alpha)=-log(1-VC), u=u_lin/(1-VC). For a mixture these are
    moment-matched Gaussian surrogate coordinates, not exact dynamics.
    """
    a = np.asarray(variance) * np.asarray(C)
    u = np.asarray(variance) * np.asarray(b) / np.sqrt(reference_variance)
    valid = 1-a > 0
    log_alpha, u_exact = np.full_like(a, np.nan), np.full_like(u, np.nan)
    log_alpha[valid] = -np.log1p(-a[valid])
    u_exact[valid] = u[valid]/(1-a[valid])
    return dict(a_linear=a, u_linear=u, log_alpha=log_alpha, u_exact=u_exact,
                exact_mapping_valid=valid)


def fit_exponential(distance, profile, bounds=(-4.0, 1e6), weights=None):
    """Signed least-squares fit profile(d) = amplitude * exp(-kappa*d).

    The amplitude is eliminated analytically for each kappa. A grid search and
    refinement of every local minimum avoid relying on a single unimodal fit.
    Integration weights give uniform weight per unit log variance, independent
    of the hybrid grid's density. An identically zero mode has undefined kappa.
    """
    d, y = np.asarray(distance, float), np.asarray(profile, float)
    w = integration_weights(d) if weights is None else np.asarray(weights, float)
    if y.shape != d.shape or w.shape != d.shape or not np.isfinite(y).all():
        raise ValueError("Finite profile and weights must match the distance grid.")
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Fit weights must be nonnegative and finite with positive sum.")
    if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("Fit bounds must be finite and increasing.")
    w = w/w.sum()
    energy = float(w @ (y*y))
    if energy <= np.finfo(float).tiny:
        return dict(amplitude=0., kappa=np.nan, relative_rmse=0., r2=np.nan,
                    at_bound=False, sign_changes=0)

    def solution(kappa):
        exponent = -kappa*d
        shift = exponent.max()
        basis = np.exp(exponent-shift)
        amplitude_scaled = (w @ (basis*y)) / (w @ (basis*basis))
        sse = float(w @ (y-amplitude_scaled*basis)**2)
        return sse, amplitude_scaled, shift

    # Resolve ordinary phase exponents near zero AND narrow low-noise profiles
    # with kappa in the hundreds/thousands, without imposing the old plot limits.
    near_lo, near_hi = max(bounds[0], -4.), min(bounds[1], 8.)
    near_grid = np.linspace(near_lo, near_hi, 241) if near_lo < near_hi else []
    grid = np.unique(np.r_[bounds, near_grid,
                           np.sinh(np.linspace(*np.arcsinh(bounds), 481))])
    scores = np.array([solution(k)[0] for k in grid])
    candidates = [(scores[0], grid[0]), (scores[-1], grid[-1])]
    for i in range(1, len(grid)-1):
        if scores[i] <= scores[i-1] and scores[i] <= scores[i+1] and (scores[i] < scores[i-1] or scores[i] < scores[i+1]):
            result = minimize_scalar(lambda k: solution(k)[0], bounds=(grid[i-1], grid[i+1]),
                                     method='bounded', options={'xatol': 1e-10})
            candidates.append((result.fun, result.x))
    sse, kappa = min(candidates)
    _, amplitude_scaled, shift = solution(kappa)
    log_amplitude = np.log(abs(amplitude_scaled))-shift if amplitude_scaled != 0 else -np.inf
    amplitude = float(np.sign(amplitude_scaled)*np.exp(log_amplitude)) if log_amplitude < np.log(np.finfo(float).max) else float(np.copysign(np.inf, amplitude_scaled))
    variance = float(w @ (y - w@y)**2)
    nonzero = y[np.abs(y) > np.max(np.abs(y))*1e-8]
    return dict(amplitude=amplitude, kappa=float(kappa),
                relative_rmse=float(np.sqrt(sse/energy)),
                r2=float(1-sse/variance) if variance > energy*1e-14 else np.nan,
                at_bound=bool(min(abs(kappa-bounds[0]), abs(kappa-bounds[1])) < 1e-5),
                sign_changes=int(np.count_nonzero(nonzero[1:]*nonzero[:-1] < 0)))


def profile_response(distance, a, u, gammas):
    """Finite-horizon Gaussian first-order kernels, Eqs. (26), (70), (83).

    These retain sign cancellation and do not require an exponential profile.
    """
    d, gammas = np.asarray(distance), np.asarray(gammas, float)
    if gammas.ndim != 1 or not np.isfinite(gammas).all() or np.any(gammas < 0):
        raise ValueError("gammas must be a finite nonnegative vector.")
    w = integration_weights(d)
    shape = (1+gammas) * (np.exp(-gammas[:, None]*d) @ (w*np.asarray(a)))
    mean = (1+gammas)/2 * (np.exp(-(1+gammas[:, None])/2*d) @ (w*np.asarray(u)))
    return dict(shape_response=shape, mean_response=mean, kl=shape**2/4+mean**2/2)


def exponential_response(kappa_a, kappa_m, epsilon_a, epsilon_m, gammas, horizon=np.inf):
    """Eq. (83) at actual amplitudes; asymptotic domain is shared with the plots."""
    gammas = np.asarray(gammas, float)
    def integral(rate):
        rate = np.asarray(rate, float)
        if np.isinf(horizon):
            result = np.full_like(rate, np.nan)
            np.divide(1, rate, out=result, where=rate > 0)
            return result
        result = np.full_like(rate, horizon)
        np.divide(-np.expm1(-rate*horizon), rate, out=result, where=rate != 0)
        return result
    # Undefined localization for a zero mode must not contaminate the other.
    shape = np.zeros_like(gammas) if epsilon_a == 0 else epsilon_a*(1+gammas)*integral(kappa_a+gammas)
    mean = np.zeros_like(gammas) if epsilon_m == 0 else epsilon_m*(1+gammas)/2*integral(kappa_m+(1+gammas)/2)
    kl = shape**2/4 + mean**2/2
    if np.isinf(horizon) and ((epsilon_a != 0 and kappa_a <= 0) or (epsilon_m != 0 and kappa_m <= -0.5)):
        kl[:] = np.nan
    return kl

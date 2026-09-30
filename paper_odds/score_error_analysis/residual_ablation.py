"""Paired residual ablation for the 1D Gaussian-mixture sampling experiment.

Uses s_lambda = exact_score + affine_error + lambda * nonlinear_residual.
The sampling clock is ell=log(V_max/V); z=(x-mu)/sqrt(V) satisfies
dz = [z/2 + (1+gamma)*sqrt(V)*s_lambda/2] d ell + sqrt(gamma) dW.
Euler/Euler-Maruyama and additive-noise stochastic Heun use the same initial
states and Brownian increments for every lambda and gamma. Gamma=0 is the
probability-flow ODE; both methods advance the standardized log-variance clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.special import ndtr

from .score_error_modes import GaussianMixture1D


def sampling_grid(mixture, sigma_min=.002, sigma_max=80., steps=600):
    """Positive EDM rho=7 noise grid, ending at the measured target noise."""
    if steps < 2 or not 0 < sigma_min < sigma_max:
        raise ValueError('Require at least two steps and 0 < sigma_min < sigma_max')
    sigma = np.linspace(sigma_max**(1/7), sigma_min**(1/7), steps+1)**7
    sigma[0], sigma[-1] = sigma_max, sigma_min
    variance = mixture.variance + sigma**2
    ell = np.log(variance[0]/variance)
    return dict(sigma=sigma, variance=variance, sqrt_variance=np.sqrt(variance), ell=ell)


def mixture_cdf(x, mixture, sigma):
    x = np.asarray(x, float)
    std = np.sqrt(np.asarray(mixture.scales)**2+sigma**2)
    return np.sum(np.asarray(mixture.probabilities)*ndtr((x[..., None]-mixture.means)/std), axis=-1)


def target_partition(mixture, sigma, bins=64):
    """Exact-target quantile bins, including both infinite tails; p mass=1/bins."""
    if bins < 4:
        raise ValueError('Use at least four bins')
    std = np.sqrt(np.asarray(mixture.scales)**2+sigma**2)
    lo = float(np.min(np.asarray(mixture.means)-12*std))
    hi = float(np.max(np.asarray(mixture.means)+12*std))
    edges = [-np.inf]
    for q in np.arange(1, bins)/bins:
        edges.append(brentq(lambda x: mixture_cdf(x, mixture, sigma)-q, lo, hi, xtol=1e-13))
    return np.r_[edges, np.inf]


def binned_kl(counts, pseudocount=.5):
    """Both KL directions against exact uniform target bin masses.

    This is the KL of the fixed partition, not an unbiased continuous-density
    estimate. The pseudocount is a count per bin and is also applied to controls.
    """
    counts = np.asarray(counts, float)
    if counts.ndim < 1 or counts.shape[-1] < 4 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('Counts must be finite and nonnegative, with at least four bins')
    if not np.isfinite(pseudocount) or pseudocount <= 0 or np.any(counts.sum(axis=-1) <= 0):
        raise ValueError('Require a positive pseudocount and nonempty samples')
    q = (counts+pseudocount)/(counts.sum(axis=-1, keepdims=True)+pseudocount*counts.shape[-1])
    p = 1/counts.shape[-1]
    return np.stack([np.sum(q*np.log(q/p), axis=-1), np.mean(np.log(p/q), axis=-1)], axis=-1)


def shared_initial_states(mixture, sigma_max, nsamples, seeds, prior='exact'):
    """Initial samples are reused across all arms, gammas, and checkpoints."""
    if nsamples < 1 or len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError('Require positive sample count and distinct seeds')
    z = []
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1701]))
        if prior == 'exact':
            component = rng.choice(len(mixture.means), size=nsamples, p=mixture.probabilities)
            std = np.sqrt(np.asarray(mixture.scales)[component]**2+sigma_max**2)
            x = np.asarray(mixture.means)[component]+std*rng.standard_normal(nsamples)
        elif prior == 'gaussian':
            x = sigma_max*rng.standard_normal(nsamples)
        else:
            raise ValueError('Prior must be exact or gaussian')
        z.append((x-mixture.mean)/np.sqrt(mixture.variance+sigma_max**2))
    return np.array(z)


def brownian_generators(seeds):
    return [np.random.default_rng(np.random.SeedSequence([int(seed), 2903])) for seed in seeds]


def uniform_lookup(z, table, z_min, z_max):
    """Linear interpolation on a regular grid. Extrapolation is forbidden."""
    z = np.asarray(z)
    if np.any(z < z_min) or np.any(z > z_max):
        raise ValueError('Requested a point outside the tabulated score domain')
    u = (z-z_min)*((len(table)-1)/(z_max-z_min))
    i = np.minimum(u.astype(np.int64), len(table)-2)
    fraction = u-i
    return table[i]+fraction*(table[i+1]-table[i])


@dataclass
class AblationField:
    mixture: GaussianMixture1D
    clock: dict
    b: np.ndarray
    C: np.ndarray
    learned_table: np.ndarray | None = None
    model: object = None
    z_limit: float = 12.
    fallback_points: int = 0
    max_abs_z: float = 0.

    def score(self, z, index, lam):
        """Return sqrt(V)*score at time index. None selects the exact control."""
        z = np.asarray(z)
        self.max_abs_z = max(self.max_abs_z, float(np.max(np.abs(z))))
        sigma = self.clock['sigma'][index]
        sv = self.clock['sqrt_variance'][index]
        x = self.mixture.mean + sv*z
        # For two components use the exact stable logistic responsibility.
        # Analytic exact scores are inexpensive and never interpolated.
        if len(self.mixture.means) == 2:
            means, var = np.asarray(self.mixture.means), np.asarray(self.mixture.scales)**2+sigma**2
            delta0, delta1 = x-means[0], x-means[1]
            logits = np.log(self.mixture.probabilities[1]/self.mixture.probabilities[0])-.5*np.log(var[1]/var[0])
            logits = logits-delta1**2/(2*var[1])+delta0**2/(2*var[0])
            from scipy.special import expit
            responsibility1 = expit(logits)
            exact = sv*(-(1-responsibility1)*delta0/var[0]-responsibility1*delta1/var[1])
        else:
            exact = sv*self.mixture.score(x, sigma)
        if lam is None:
            return exact
        affine = exact+sv*self.b[index]+self.clock['variance'][index]*self.C[index]*z
        if lam == 0:
            return affine
        if self.learned_table is None:
            if self.model is None:
                raise ValueError('Nonzero lambda requires a learned score')
            learned = sv*self.model(x, sigma)
        else:
            outside = (z < -self.z_limit) | (z > self.z_limit)
            # Outside points are evaluated directly, never clipped or dropped.
            learned = np.empty_like(z)
            inside = ~outside
            learned[inside] = uniform_lookup(z[inside], self.learned_table[index], -self.z_limit, self.z_limit)
            if np.any(outside):
                if self.model is None:
                    raise ValueError('Score-table overflow needs a direct model evaluator')
                self.fallback_points += int(outside.sum())
                learned[outside] = sv*self.model(x[outside], sigma)
        return learned if lam == 1 else affine+lam*(learned-affine)


def sample_paired(field, gammas, lambdas, initial, seeds, refinement=1, integrator="heun"):
    """Euler/Euler-Maruyama or Heun, with shared additive increments.

    ``euler`` uses one drift evaluation per step: explicit Euler when gamma=0
    and Euler-Maruyama otherwise, in the existing log-variance time coordinate.

    z has [gamma, lambda, seed, particle] axes. Refinement is reserved for callers
    providing a correspondingly finer field; generators are common per level.
    """
    if integrator not in ('euler', 'heun'):
        raise ValueError('Integrator must be euler or heun')
    gammas = np.asarray(gammas, float)
    if gammas.ndim != 1 or not np.isfinite(gammas).all() or np.any(gammas < 0):
        raise ValueError('Gammas must be finite and nonnegative')
    if len(lambdas) < 1 or any(lam is not None and (not np.isfinite(lam) or not 0 <= lam <= 1) for lam in lambdas):
        raise ValueError('Lambdas must lie in [0,1], or None for the exact control')
    if refinement != 1:
        raise ValueError('Pass an explicitly refined field instead of a refinement argument')
    z = np.broadcast_to(initial, (len(gammas), len(lambdas), *initial.shape)).copy()
    rngs = brownian_generators(seeds)
    gamma = gammas[:, None, None]
    sqrt_gamma = np.sqrt(gamma)
    for i, h in enumerate(np.diff(field.clock['ell'])):
        if h <= 0:
            raise ValueError('The sampling clock must increase strictly')
        noise = np.stack([rng.standard_normal(initial.shape[-1]) for rng in rngs])*np.sqrt(h)
        increment = sqrt_gamma*noise
        for j, lam in enumerate(lambdas):
            state = z[:, j]
            drift0 = state/2 + (1+gamma)/2*field.score(state, i, lam)
            predictor = state+h*drift0+increment
            if integrator == 'euler':
                z[:, j] = predictor
            else:
                drift1 = predictor/2+(1+gamma)/2*field.score(predictor, i+1, lam)
                z[:, j] = state + h/2*(drift0+drift1)+increment
        if not np.isfinite(z).all():
            raise FloatingPointError(f'Nonfinite samples at step {i}; no samples were discarded')
    return field.mixture.mean+field.clock['sqrt_variance'][-1]*z


def histogram_counts(samples, edges):
    """Count all particles in a target-defined partition, retaining every tail."""
    samples = np.asarray(samples)
    if not np.isfinite(samples).all():
        raise ValueError('Samples must all be finite')
    shape = samples.shape[:-1]
    counts = np.array([np.histogram(row, edges)[0] for row in samples.reshape(-1, samples.shape[-1])])
    counts = counts.reshape(*shape, len(edges)-1)
    if not np.all(counts.sum(axis=-1) == samples.shape[-1]):
        raise AssertionError('Histogram did not account for every particle')
    return counts


def gaussian_moment_kl(clock, b, C, gammas, initial_r=0., initial_v=1., integrator="adaptive"):
    """Nonperturbative moment-matched Gaussian closure, using the same b,C.

    This isolates the small-amplitude approximation from the Gaussian-reference
    approximation. The resulting KL is still a Gaussian surrogate for the mixture.
    The equations remain defined when 1-VC<=0: use the affine drift directly.
    ``euler`` takes exactly one explicit step per clock interval. The adaptive
    reference solver remains available for reproducing older experiments.
    """
    gammas = np.asarray(gammas, float)
    ell, V = clock['ell'], clock['variance']
    a = V*np.asarray(C)
    u = V*np.asarray(b)/np.sqrt(V[-1])
    def rhs(t, state):
        at, ut = np.interp(t, ell, a), np.interp(t, ell, u)
        r, v = state[:len(gammas)], state[len(gammas):]
        return np.r_[-(1+gammas)/2*((1-at)*r-ut), (-gammas+(1+gammas)*at)*v+gammas]
    state = np.r_[np.full(len(gammas), initial_r), np.full(len(gammas), initial_v)].astype(float)
    if integrator == 'euler':
        for t, h in zip(ell[:-1], np.diff(ell)):
            if h <= 0:
                raise ValueError('The moment clock must increase strictly')
            state += h*rhs(t, state)
    elif integrator == 'adaptive':
        solution = solve_ivp(rhs, (0, ell[-1]), state,
                             rtol=1e-8, atol=1e-11, max_step=float(np.max(np.diff(ell))))
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.y[:, -1]
    else:
        raise ValueError('Moment integrator must be euler or adaptive')
    r, v = np.split(state, 2)
    if np.any(v <= 0) or not np.isfinite(state).all():
        raise FloatingPointError('Invalid Gaussian closure moments')
    kl = np.stack([.5*(v-np.log(v)-1+r*r), .5*(np.log(v)+1/v-1+r*r/v)], axis=-1)
    return np.maximum(kl, 0), r, v

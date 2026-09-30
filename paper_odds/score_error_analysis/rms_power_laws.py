"""Continuous monotone power-law envelopes for positive checkpoint RMS curves.

Fit 1--3 segments with free breakpoints. Log-space slopes are solved exactly
for each breakpoint choice; magnitude-space amplitudes are solved analytically.
Seeded, restarted differential evolution searches the remaining parameters.
Both magnitude and log-magnitude objectives use the same quadrature weights.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar, nnls


def evaluate_power_law(t, fit, *, log=False):
    """Evaluate a fitted envelope, continuous at each saved breakpoint."""
    x = np.log(np.asarray(t, dtype=float)/fit['t_min'])
    p = np.asarray(fit['exponents'])
    knots = np.log(np.asarray(fit['breakpoints'])/fit['t_min'])
    log_value = np.log(fit['amplitude'])-p[0]*x
    for knot, change in zip(knots, np.diff(p)):
        log_value -= change*np.maximum(x-knot, 0)
    return log_value if log else np.exp(log_value)


def fit_piecewise_power_laws(t, rms, weights=None, *, max_segments=3,
                            fit_space='log', seed=42, restarts=4, initial_fits=None):
    """Return the best numerical fit found for each number of segments.

    R(t) = B exp(-sum_j p_j length([0, log(t/t_min)] intersect segment_j)),
    with B>0, p_j>=0 and 0--2 free, ordered breakpoints. Zero powers allow a
    plateau. No minimum segment width is imposed. ``fit_space='magnitude'``
    minimizes normalized squared magnitude error; ``'log'`` minimizes squared
    natural-log error. Normalization does not change the magnitude optimum.

    A smaller family is embedded in each larger one, so its objective cannot
    worsen. A numerical rate bound starts at 32 and expands when approached;
    ``rate_bound_hit`` and optimizer convergence are returned for inspection.
    For log fits, nonnegative least squares solves the powers exactly for each
    breakpoint choice. ``initial_fits`` can supply envelopes fitted with another
    objective as extra starting candidates. Multiple searches improve robustness
    but do not prove global optimality.
    """
    t, rms = np.asarray(t, dtype=float), np.asarray(rms, dtype=float)
    if max_segments not in (1, 2, 3) or restarts < 1:
        raise ValueError('Use 1--3 segments and at least one search restart.')
    if fit_space not in ('magnitude', 'log'):
        raise ValueError("fit_space must be 'magnitude' or 'log'.")
    if (t.ndim != 1 or rms.shape != t.shape or len(t) < 2*max_segments+1
            or not np.all(np.isfinite(t)) or not np.all(t > 0)
            or not np.all(np.diff(t) > 0)
            or not np.all(np.isfinite(rms)) or not np.all(rms > 0)):
        raise ValueError('Supply enough increasing positive times and finite positive RMS values.')
    x = np.log(t/t[0])
    if weights is None:
        weights = np.r_[(x[1]-x[0])/2, (x[2:]-x[:-2])/2, (x[-1]-x[-2])/2]
    w = np.asarray(weights, dtype=float)
    if w.shape != t.shape or not np.all(np.isfinite(w)) or not np.all(w > 0):
        raise ValueError('Quadrature weights must be finite and strictly positive.')
    w = w/w.sum()
    log_rms, span = np.log(rms), x[-1]
    energy = np.sum(w*rms**2)
    centered_energy = np.sum(w*(rms-np.sum(w*rms))**2)
    fits = []

    for segments in range(1, max_segments+1):
        def calculate(parameters):
            # Vectorized optimizer convention: parameters have shape (dimension, population).
            theta = np.atleast_2d(np.asarray(parameters).T)
            powers = theta[:, :segments]
            knots = np.sort(theta[:, segments:], axis=1)*span
            edges = np.c_[np.zeros(len(theta)), knots, np.full(len(theta), span)]
            lengths = np.minimum(np.maximum(x[None, :, None]-edges[:, None, :-1], 0),
                                 np.diff(edges)[:, None, :])
            log_basis = -np.sum(powers[:, None, :]*lengths, axis=2)
            if fit_space == 'log':
                log_amplitude = np.sum(w*(log_rms-log_basis), axis=1)
                objective = np.sum(w*(log_amplitude[:, None]+log_basis-log_rms)**2, axis=1)
            else:
                basis = np.exp(log_basis)
                amplitude = np.sum(w*basis*rms, axis=1)/np.sum(w*basis**2, axis=1)
                log_amplitude = np.log(amplitude)
                objective = np.sum(w*(amplitude[:, None]*basis-rms)**2, axis=1)/energy
            return objective, log_amplitude

        if segments == 1:
            if fit_space == 'log':
                dx, dy = x-np.sum(w*x), log_rms-np.sum(w*log_rms)
                power = max(0., -np.sum(w*dx*dy)/np.sum(w*dx**2))
                theta_best = np.array([power])
            else:
                rate_max = 40/x[1]
                grid = np.r_[0., np.geomspace(1e-8/span, rate_max, 513)]
                losses = calculate(grid[None, :])[0]
                candidates = [0., rate_max]
                for i in np.flatnonzero((losses[1:-1] < losses[:-2]) &
                                        (losses[1:-1] <= losses[2:]))+1:
                    solution = minimize_scalar(lambda p: calculate([p])[0][0], method='bounded',
                                               bounds=(grid[i-1], grid[i+1]),
                                               options={'xatol': 1e-11})
                    if not solution.success:
                        raise RuntimeError(solution.message)
                    candidates.append(solution.x)
                theta_best = np.array([min(candidates, key=lambda p: calculate([p])[0][0])])
            search_losses = [float(calculate(theta_best)[0][0])]
            converged, bound_hit = True, bool(fit_space == 'magnitude' and theta_best[0] == rate_max)
        else:
            # Reproduce the previous fit exactly by splitting its widest segment.
            previous = fits[-1]
            old_knots = np.log(previous['breakpoints']/t[0])
            old_edges = np.r_[0., old_knots, span]
            split = np.argmax(np.diff(old_edges))
            midpoint = (old_edges[split]+old_edges[split+1])/2
            nested_knots = np.insert(old_knots, split, midpoint)
            nested_powers = np.insert(previous['exponents'], split, previous['exponents'][split])
            theta_best = np.r_[nested_powers, nested_knots/span]
            best_loss = float(calculate(theta_best)[0][0])
            converged = previous['search_converged']
            if initial_fits is not None:
                initial = initial_fits[segments-1]
                candidate = np.r_[initial['exponents'], np.log(initial['breakpoints']/t[0])/span]
                if calculate(candidate)[0][0] < best_loss:
                    theta_best, best_loss = candidate, float(calculate(candidate)[0][0])
                    converged = initial['search_converged']
            rate_bound = max(32., 2*max(theta_best[:segments]))
            search_losses = []
            if fit_space == 'log':
                # Variable projection: the inner slope problem is convex. Search only
                # the one or two breakpoint coordinates, avoiding joint-slope local minima.
                sqrt_w = np.sqrt(w)
                log_mean = np.sum(w*log_rms)
                target = -sqrt_w*(log_rms-log_mean)

                def solve_powers(fractions):
                    knots = np.sort(fractions)*span
                    edges = np.r_[0., knots, span]
                    durations = np.minimum(np.maximum(x[:, None]-edges[:-1], 0), np.diff(edges))
                    centered = sqrt_w[:, None]*(durations-np.sum(w[:, None]*durations, axis=0))
                    powers, residual = nnls(centered, target)
                    return residual**2, np.r_[powers, np.sort(fractions)]

                # A coarse mesh supplies an independent start before continuous searches.
                grid = np.linspace(0.01, 0.99, 25)
                starts = ([[q] for q in grid] if segments == 2 else
                          [[a, b] for i, a in enumerate(grid) for b in grid[i+1:]])
                for start in starts:
                    loss, candidate = solve_powers(start)
                    if loss < best_loss:
                        best_loss, theta_best = loss, candidate
                        converged = False
                for restart in range(restarts):
                    solution = differential_evolution(
                        lambda fractions: solve_powers(fractions)[0],
                        [(0., 1.)]*(segments-1), x0=theta_best[segments:],
                        seed=int(seed)+restart, popsize=20, maxiter=1800,
                        tol=1e-10, atol=1e-13, polish=True)
                    loss, candidate = solve_powers(solution.x)
                    search_losses.append(loss)
                    if loss < best_loss:
                        best_loss, theta_best = loss, candidate
                        converged = bool(solution.success)
                bound_hit = False
            for expansion in range(5 if fit_space == 'magnitude' else 0):
                for restart in range(restarts):
                    solution = differential_evolution(
                        lambda theta: calculate(theta)[0],
                        [(0., rate_bound)]*segments+[(0., 1.)]*(segments-1),
                        x0=theta_best, seed=int(seed)+restart, popsize=20,
                        maxiter=1800, tol=1e-10, atol=1e-13,
                        vectorized=True, updating='deferred', polish=True)
                    loss = float(solution.fun)
                    search_losses.append(loss)
                    if loss < best_loss:
                        best_loss, theta_best = loss, solution.x.copy()
                        converged = bool(solution.success)
                bound_hit = bool(np.max(theta_best[:segments]) > 0.95*rate_bound)
                if not bound_hit:
                    break
                rate_bound *= 2

        objective, log_amplitude = calculate(theta_best)
        fit = dict(n_exponents=segments, t_min=float(t[0]), fit_space=fit_space,
                   amplitude=float(np.exp(log_amplitude[0])),
                   exponents=theta_best[:segments].copy(),
                   breakpoints=t[0]*np.exp(np.sort(theta_best[segments:])*span),
                   objective=float(objective[0]), search_converged=converged,
                   search_objectives=np.asarray(search_losses), rate_bound_hit=bound_hit)
        log_prediction = evaluate_power_law(t, fit, log=True)
        fit['prediction'] = np.exp(log_prediction)
        mse = np.sum(w*(fit['prediction']-rms)**2)
        fit.update(relative_RMSE=float(np.sqrt(mse/energy)),
                   log10_RMSE=float(np.sqrt(np.sum(w*(log_prediction-log_rms)**2))/np.log(10)),
                   R2=float(1-mse/centered_energy) if centered_energy > 0 else np.nan)
        fits.append(fit)
    return fits

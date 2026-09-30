"""Checkpoint-backed, paired time-dependent stochasticity experiment.

Uses the same standardized log-variance Euler scheme as residual_ablation.py.
All schedules share a grid split at every edge, stability refinements, initial
states, and per-seed Brownian increments. GPU inference, when requested, is
restricted by the caller to device 6; sampling and checked lookup tables use CPU.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, exprel

from .score_error_modes import GaussianMixture1D
from .residual_ablation import (sampling_grid, shared_initial_states, brownian_generators,
                               target_partition, binned_kl, histogram_counts, uniform_lookup)


def gamma_of(spec, sigma):
    sigma = np.asarray(sigma, float)
    if spec['kind'] == 'const':
        return np.full(sigma.shape, spec['gamma'], float)
    if spec['kind'] != 'window':
        raise ValueError('Resolve tailored schedules before evaluation.')
    return np.where((sigma >= spec['lo']) & (sigma <= spec['hi']), spec['gamma'], 0.)


def split_profile(sigma, a, u, variance0, spec, refinement=1):
    """Ascending distance cells, split exactly at schedule edges.

    Interpolate the measured a,u linearly in distance; midpoint freezing is
    second-order accurate for profiles. Gamma is exactly constant per cell.
    """
    sigma = np.asarray(sigma, float)
    d = np.log((variance0+sigma*sigma)/(variance0+sigma[0]**2))
    if np.any(np.diff(d) <= 0) or refinement < 1:
        raise ValueError('Require increasing noise levels and positive refinement.')
    points = d.copy()
    if spec['kind'] == 'window':
        edges = [s for s in (spec['lo'], spec['hi']) if sigma[0] < s < sigma[-1]]
        points = np.unique(np.r_[points, np.log((variance0+np.square(edges))/(variance0+sigma[0]**2))])
    grid = np.unique(np.concatenate([np.linspace(lo, hi, refinement+1) for lo, hi in zip(points[:-1], points[1:])]))
    midpoint = (grid[:-1]+grid[1:])/2
    sigmid = np.sqrt(np.maximum((variance0+sigma[0]**2)*np.exp(midpoint)-variance0, 0))
    return grid, np.interp(midpoint, d, a), np.interp(midpoint, d, u), gamma_of(spec, sigmid)


def cell_response(grid, a, u, gamma):
    """Exact response of cellwise constant profiles and stochasticity."""
    width = np.diff(grid)
    eta = np.r_[0., np.cumsum(gamma*width)[:-1]]
    L = np.sum(np.exp(-eta)*(1+gamma)*width*exprel(-gamma*width)*a)
    M = np.sum(np.exp(-(grid[:-1]+eta)/2)*(-np.expm1(-(1+gamma)*width/2))*u)
    return float(L), float(M), float(L*L/4+M*M/2)


def cell_moments(grid, a, u, gamma):
    """Exact constant-coefficient updates, avoiding exp(z)-1 cancellation."""
    r, v = 0., 1.
    for h, at, ut, g in zip(np.diff(grid)[::-1], a[::-1], u[::-1], gamma[::-1]):
        c = -g+(1+g)*at
        k = -(1+g)/2*(1-at)
        v = v*np.exp(c*h)+g*h*exprel(c*h)
        r = r*np.exp(k*h)+(1+g)/2*ut*h*exprel(k*h)
    if not np.isfinite([r, v]).all() or v <= 0:
        raise FloatingPointError('Invalid Gaussian moments.')
    delta, lv = v-1, np.log(v)
    kl = np.maximum([.5*(delta-lv+r*r), .5*(lv-delta/v+r*r/v)], 0)
    return kl


def predict_schedule(sigma, a, u, variance0, spec, refinement=2):
    grid, aa, uu, gg = split_profile(sigma, a, u, variance0, spec, refinement)
    L, M, h = cell_response(grid, aa, uu, gg)
    moments = cell_moments(grid, aa, uu, gg)
    return dict(first_order=h, moment_q_p=float(moments[0]), moment_p_q=float(moments[1]), L=L, M=M)


def common_clock(mixture, sigma_min, sigma_max, steps, specifications,
                 max_step_contraction=.1, stiffness_aware=True, refinement=1):
    """EDM rho=7 base grid plus every edge and a common stability refinement.

    Refinement is common to ALL schedules/checkpoints, so matching does not rely
    on schedule-specific Brownian bridges. With no refinement/edges this is
    exactly residual_ablation.sampling_grid. Gamma is sampled at cell interiors.
    """
    if max_step_contraction <= 0 or refinement < 1:
        raise ValueError('Require positive contraction limit and refinement.')
    base = sampling_grid(mixture, sigma_min, sigma_max, steps)
    maximum_variance = base['variance'][0]
    edges = []
    for spec in specifications:
        if spec['kind'] == 'window':
            edges.extend(s for s in (spec['lo'], spec['hi']) if sigma_min < s < sigma_max)
    edge_ell = np.log(maximum_variance/(mixture.variance+np.square(edges)))
    clock_points = np.unique(np.r_[base['ell'], edge_ell])
    nodes = [0.]
    for lo, hi in zip(clock_points[:-1], clock_points[1:]):
        var_hi, var_lo = maximum_variance*np.exp(-lo), maximum_variance*np.exp(-hi)
        mid_sigma = np.sqrt(max(maximum_variance*np.exp(-(lo+hi)/2)-mixture.variance, sigma_min**2))
        gmax = max(float(gamma_of(spec, mid_sigma)) for spec in specifications)
        local_var = min(mixture.scales)**2+max(var_lo-mixture.variance, 0) if stiffness_aware else var_lo
        contraction_bound = (.5+(1+gmax)*var_hi/(2*local_var))*(hi-lo)
        n = max(1, int(np.ceil(contraction_bound/max_step_contraction)))*refinement
        nodes.extend(np.linspace(lo, hi, n+1)[1:])
    ell = np.asarray(nodes)
    variance = maximum_variance*np.exp(-ell)
    sigma = np.sqrt(np.maximum(variance-mixture.variance, sigma_min**2))
    sigma[[0, -1]] = [sigma_max, sigma_min]
    variance = mixture.variance+sigma*sigma
    return dict(sigma=sigma, variance=variance, sqrt_variance=np.sqrt(variance), ell=ell)


def schedule_matrix(clock, mixture, specs):
    ellmid = (clock['ell'][:-1]+clock['ell'][1:])/2
    sigma = np.sqrt(np.maximum(clock['variance'][0]*np.exp(-ellmid)-mixture.variance, 0))
    return np.stack([gamma_of(spec, sigma) for spec in specs])


def sample_schedules(field, gamma_steps, lambdas, initial, seeds, noise_increments=None):
    """Paired Euler/EM in z=(x-mu)/sqrt(V), just as sample_paired.

    Each seed is an independent replicate. Identical seeded increments are
    reused for every schedule, arm, checkpoint and exact-score control.
    """
    gg = np.asarray(gamma_steps, float)
    if gg.shape[1] != len(field.clock['ell'])-1 or np.any(gg < 0):
        raise ValueError('Schedule matrix must match nonnegative interval gammas.')
    z = np.broadcast_to(initial, (len(gg), len(lambdas), *initial.shape)).copy()
    if noise_increments is not None and np.shape(noise_increments) != (gg.shape[1], *initial.shape):
        raise ValueError('Brownian increments must have [interval, seed, particle] shape.')
    rngs = brownian_generators(seeds)
    for i, h in enumerate(np.diff(field.clock['ell'])):
        noise = (np.stack([rng.standard_normal(initial.shape[-1]) for rng in rngs])*np.sqrt(h)
                 if noise_increments is None else noise_increments[i])
        g = gg[:, i, None, None]
        increment = np.sqrt(g)*noise
        for j, lam in enumerate(lambdas):
            state = z[:, j]
            drift = state/2+(1+g)/2*field.score(state, i, lam)
            z[:, j] = state+h*drift+increment
        if not np.isfinite(z).all():
            raise FloatingPointError(f'Nonfinite particles at step {i}; none discarded.')
    return field.mixture.mean+field.clock['sqrt_variance'][-1]*z


def content_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(value):
    # Specs use None to represent an unbounded edge in persisted JSON.
    def normalize(x):
        if isinstance(x, dict):
            return {str(k): normalize(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [normalize(v) for v in x]
        if isinstance(x, np.ndarray):
            return normalize(x.tolist())
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return 'Infinity' if x > 0 else '-Infinity' if x < 0 else 'NaN'
        if isinstance(x, np.generic):
            return x.item()
        return x
    return json.dumps(normalize(value), sort_keys=True, indent=2, allow_nan=False)


def prepare_cache(out, configuration, *, force=False):
    """Never mix settings or silently overwrite the provenance of existing rows."""
    digest = hashlib.sha256(canonical_json(configuration).encode()).hexdigest()
    destination = Path(out)/digest[:12]
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/'metadata.json'
    if path.exists():
        metadata = json.loads(path.read_text())
        if metadata['fingerprint'] != digest:
            raise ValueError('Cache fingerprint mismatch.')
    elif list(destination.iterdir()):
        raise ValueError('Nonempty cache without metadata.')
    metadata = dict(fingerprint=digest, configuration=json.loads(canonical_json(configuration)))
    path.write_text(json.dumps(metadata, indent=2)+'\n')
    if force:
        # Force is only for this exact configuration, never previous experiments.
        for file in destination.glob('*.npz'):
            file.unlink()
    return destination, digest


def build_field(record, config, clock, device='cpu'):
    """Fresh spatial projections at every sampler node; validated learned tables."""
    import torch
    from .checkpoint_score import CheckpointScore
    from .fit_checkpoint_modes import project_grid
    from .residual_ablation import AblationField
    torch.set_num_threads(1)
    mixture = GaussianMixture1D(**config['mixture'])
    path = Path(config['root'])/record['checkpoint']
    if content_hash(path) != record['checkpoint_sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    model = CheckpointScore(path, device=device)
    if model.epoch != record['epoch']:
        raise ValueError('Checkpoint epoch mismatch.')
    projection = project_grid(mixture, clock['sigma'], model, config['quadrature_order'], 8192)
    nodes = config['table_nodes']
    for attempt in range(3):
        zgrid = np.linspace(-config['z_limit'], config['z_limit'], nodes)
        x = mixture.mean+clock['sqrt_variance'][:, None]*zgrid
        table = model(x, clock['sigma'][:, None])*clock['sqrt_variance'][:, None]
        indices = np.unique(np.linspace(0, len(clock['sigma'])-1, 17).round().astype(int))
        check_x, weights = mixture.quadrature(clock['sigma'][indices], config['quadrature_order'])
        reference = model(check_x, clock['sigma'][indices, None])
        exact = mixture.score(check_x, clock['sigma'][indices, None])
        errors = []
        for j, index in enumerate(indices):
            z = (check_x[j]-mixture.mean)/clock['sqrt_variance'][index]
            inside = np.abs(z) <= config['z_limit']
            approximate = reference[j].copy()
            approximate[inside] = uniform_lookup(z[inside], table[index], -config['z_limit'], config['z_limit'])/clock['sqrt_variance'][index]
            errors.append(float(np.sqrt(weights @ ((approximate-reference[j])**2)/max(weights @ ((reference[j]-exact[j])**2), 1e-30))))
        if max(errors) <= config['table_tolerance']:
            # Sampling normally remains inside the table. A CPU loader provides
            # exact inference for any escaped particle without another GPU context.
            field = AblationField(mixture, clock, projection['b'], projection['C'], table, model, config['z_limit'])
            return field, dict(table_nodes=nodes, table_error=max(errors))
        nodes = 2*nodes-1
    raise RuntimeError(f'Score interpolation failed tolerance: {max(errors)}')


def summarize_samples(samples, mixture, config):
    out = dict(sample_mean=samples.mean(axis=-1), sample_variance=samples.var(axis=-1))
    for bins in config['bins']:
        out[f'counts_{bins}'] = histogram_counts(samples, target_partition(mixture, config['sigma_min'], bins))
    return out


def run_checkpoint(record, config, clock, specs, destination, fingerprint):
    """Picklable CPU worker: inference tables, both sampler arms, atomic results."""
    import torch
    torch.set_num_threads(1)
    dest = Path(destination)/f"{record['run']}_epoch{record['epoch']:02d}.npz"
    if dest.exists():
        with np.load(dest) as saved:
            if str(saved['fingerprint']) != fingerprint or str(saved['checkpoint_sha256']) != record['checkpoint_sha256']:
                raise ValueError('Stale checkpoint cache.')
        return dict(run=record['run'], epoch=record['epoch'], reused=True)
    start = time.perf_counter()
    field, validation = build_field(record, config, clock, config.get('inference_device', 'cpu'))
    mixture = field.mixture
    initial = shared_initial_states(mixture, config['sigma_max'], config['particles'], config['seeds'], 'exact')
    gamma = schedule_matrix(clock, mixture, specs)
    samples = sample_schedules(field, gamma, config['lambdas'], initial, config['seeds'])
    arrays = summarize_samples(samples, mixture, config)
    # Responses of the SAME newly projected node profiles used in sampling.
    s = clock['sigma'][::-1]
    a = (clock['variance']*field.C)[::-1]
    u = (clock['variance']*field.b/np.sqrt(clock['variance'][-1]))[::-1]
    prediction = [predict_schedule(s, a, u, mixture.variance, spec, refinement=2) for spec in specs]
    arrays.update(sigma=clock['sigma'], b=field.b, C=field.C,
        profile_kl=np.array([p['first_order'] for p in prediction]),
        moment_kl=np.array([[p['moment_q_p'],p['moment_p_q']] for p in prediction]),
        fingerprint=fingerprint, checkpoint_sha256=record['checkpoint_sha256'],
        actual_steps=len(clock['ell'])-1, fallback_points=field.fallback_points,
        **validation, seconds=time.perf_counter()-start)
    temp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, dest)
    return dict(run=record['run'], epoch=record['epoch'], seconds=float(arrays['seconds']), **validation)


def run_exact(config, clock, named_specs, destination, fingerprint):
    """Exact-score control per distinct schedule, on the same common grid."""
    from .residual_ablation import AblationField
    dest = Path(destination)/'exact_controls.npz'
    if dest.exists():
        with np.load(dest) as saved:
            if str(saved['fingerprint']) != fingerprint:
                raise ValueError('Stale exact-score cache.')
        return
    mixture = GaussianMixture1D(**config['mixture'])
    initial = shared_initial_states(mixture, config['sigma_max'], config['particles'], config['seeds'], 'exact')
    field = AblationField(mixture, clock, np.zeros(len(clock['ell'])), np.zeros(len(clock['ell'])))
    names = list(named_specs)
    arrays_per_batch = []
    for start in range(0, len(names), 16):
        specs = [named_specs[key] for key in names[start:start+16]]
        gamma = schedule_matrix(clock, mixture, specs)
        samples = sample_schedules(field, gamma, [None], initial, config['seeds'])
        arrays_per_batch.append(summarize_samples(samples, mixture, config))
    arrays = {key: np.concatenate([a[key] for a in arrays_per_batch]) for key in arrays_per_batch[0]}
    arrays.update(names=np.asarray(names), fingerprint=fingerprint, actual_steps=len(clock['ell'])-1)
    temp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, dest)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def result_rows(records, config, specs_by_cp, destination, fingerprint):
    """Produce pooled results while retaining per-seed/bin counts in NPZ caches."""
    destination = Path(destination)
    rows = []
    with np.load(destination/'exact_controls.npz') as saved:
        if str(saved['fingerprint']) != fingerprint:
            raise ValueError('Exact control provenance mismatch.')
        for j, name in enumerate(saved['names']):
            run, epoch, key = str(name).split('|')
            for bins in config['bins']:
                kl = binned_kl(saved[f'counts_{bins}'][j, 0].sum(axis=0), config['pseudocount'])
                rows.append(dict(arm='exact', run=run, epoch=int(epoch), schedule=key, bins=bins,
                    kl_q_p=float(kl[0]), kl_p_q=float(kl[1]), n_samples=config['particles']*len(config['seeds']),
                    n_steps=len(config['clock_ell'])-1, profile_kl=np.nan, moment_q_p=np.nan, moment_p_q=np.nan))
    for record in records:
        cp = (record['run'], record['epoch'])
        with np.load(destination/f'{cp[0]}_epoch{cp[1]:02d}.npz') as saved:
            if str(saved['fingerprint']) != fingerprint:
                raise ValueError('Checkpoint provenance mismatch.')
            for j, key in enumerate(specs_by_cp[cp]):
                for k, lam in enumerate(config['lambdas']):
                    for bins in config['bins']:
                        kl = binned_kl(saved[f'counts_{bins}'][j, k].sum(axis=0), config['pseudocount'])
                        rows.append(dict(arm='full' if lam == 1 else 'affine', run=cp[0], epoch=cp[1],
                            schedule=key, bins=bins, kl_q_p=float(kl[0]), kl_p_q=float(kl[1]),
                            n_samples=config['particles']*len(config['seeds']), n_steps=int(saved['actual_steps']),
                            profile_kl=float(saved['profile_kl'][j]), moment_q_p=float(saved['moment_kl'][j, 0]),
                            moment_p_q=float(saved['moment_kl'][j, 1])))
    write_csv(destination/'results.csv', rows)
    return rows

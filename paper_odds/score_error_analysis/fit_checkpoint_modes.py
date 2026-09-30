#!/usr/bin/env python3
"""Fit score-error modes for default3/4/5 and compare Gaussian predictions.

From the repository root:
    python -m paper_odds.score_error_analysis.fit_checkpoint_modes
Use --help for quadrature, time grid, fit bounds, and smoke-run options.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from .checkpoint_score import CheckpointScore
from .score_error_modes import (
    GaussianMixture1D, affine_projection, exponential_response, fit_exponential,
    integration_weights, normalized_profiles, profile_response, variance_clock,
)

PAPER = Path(__file__).resolve().parents[1]
ROOT = next(p for p in PAPER.parents if (p / 'diffsci').is_dir())
DEFAULT_OUTPUT = PAPER / 'outputs/score_error_modes'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to write to {path}')
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_checkpoints(root, runs):
    """Match saved statistics to checkpoints by (run, epoch), never list order."""
    records = []
    for run in runs:
        stats_path = root / f'stochasticity_paper/stats/output_{run}/results.npy'
        saved = np.load(stats_path, allow_pickle=True).item()
        dirs = list((root/'savedmodels/production').glob(f'*-bs=16-*-nlm={run}/checkpoints'))
        if len(dirs) != 1:
            raise ValueError(f'Expected one production checkpoint directory for {run}, found {dirs}')
        checkpoints = {}
        for path in dirs[0].glob('*.ckpt'):
            match = re.search(r'epoch=(\d+)', path.name)
            if match:
                epoch = int(match[1])
                if epoch in checkpoints:
                    raise ValueError(f'Duplicate epoch {epoch} in {dirs[0]}')
                checkpoints[epoch] = path
        if set(checkpoints) != set(saved['entropies']) or set(checkpoints) != set(saved['errors']):
            raise ValueError(f'Checkpoint/statistics epoch mismatch for {run}')
        for epoch in sorted(checkpoints):
            ent = saved['entropies'][epoch]
            gamma = np.asarray(ent['gamma_values'], float)
            if gamma.ndim != 1 or np.any(np.diff(gamma) <= 0) or gamma[0] < 0:
                raise ValueError(f'Invalid saved gamma grid for {run}:{epoch}')
            if any(np.asarray(ent[key]).shape != gamma.shape for key in ('sde_entropies', 'inv_sde_entropies')):
                raise ValueError(f'KL/gamma length mismatch for {run}:{epoch}')
            records.append(dict(run=run, epoch=epoch, checkpoint=checkpoints[epoch],
                                stats_path=stats_path, entropy=ent))
    return records


def project_grid(mixture, sigma, model, order, batch_size):
    x, w = mixture.quadrature(sigma, order)
    exact = mixture.score(x, sigma[:, None])
    learned = model(x, sigma[:, None], batch_size=batch_size)
    projections = [affine_projection(xi[:, None], ei[:, None], w, [mixture.mean])
                   for xi, ei in zip(x, learned-exact)]
    arrays = {key: np.array([p[key] for p in projections]) for key in projections[0]}
    arrays['b'] = arrays['b'][:, 0]
    arrays['C'] = arrays['C'][:, 0, 0]
    # Independent Stein identities: E[s_exact]=0, E[(X-mu)s_exact]=-1.
    arrays['stein_mean_error'] = exact @ w
    arrays['stein_linear_error'] = ((x-mixture.mean)*exact) @ w + 1
    return arrays


def compute_checkpoint(record, mixture, clock, args):
    model = CheckpointScore(record['checkpoint'], device=args.device)
    if model.epoch != record['epoch']:
        raise ValueError(f"Epoch in checkpoint disagrees with filename: {record['checkpoint']}")
    arrays = project_grid(mixture, clock['sigma'], model, args.quadrature_order, args.batch_size)
    arrays.update(normalized_profiles(arrays['b'], arrays['C'], clock['variance'], clock['reference_variance']))
    # Double the order on a sparse grid for EVERY checkpoint.
    indices = np.unique(np.r_[np.linspace(0, len(clock['sigma'])-1, 21).round().astype(int),
                               np.argmax(arrays['total_energy']),
                               np.argmax(np.abs(arrays['a_linear'])),
                               np.argmax(np.abs(arrays['u_linear']))])
    check = project_grid(mixture, clock['sigma'][indices], model, args.quadrature_order*2, args.batch_size)
    arrays['quadrature_check_indices'] = indices
    for key in ('b', 'C', 'total_energy', 'residual_energy'):
        arrays[f'quadrature_check_{key}'] = check[key]
    return arrays


def summarize(record, arrays, clock, args):
    d = clock['distance']
    iw = integration_weights(d)
    lo, hi = args.fit_distance
    selected = (d >= lo) & (d <= hi)
    if selected.sum() < 3:
        raise ValueError('Fit distance range contains fewer than three points')
    fits = {name: fit_exponential(d[selected], arrays[key][selected], args.kappa_bounds)
            for name, key in (('a', 'a_linear'), ('m', 'u_linear'))}
    a, m = fits['a'], fits['m']
    ratio = abs(m['amplitude']/a['amplitude']) if a['amplitude'] != 0 else np.inf
    total = float(iw @ arrays['total_energy'])
    fraction = lambda key: float(iw @ arrays[key] / total) if total > 0 else 0.
    row = dict(run=record['run'], epoch=record['epoch'], checkpoint=str(record['checkpoint'].relative_to(args.root)),
               kappa_a=a['kappa'], kappa_m=m['kappa'], epsilon_a=a['amplitude'], epsilon_m=m['amplitude'],
               amplitude_ratio=ratio, shape_fit_relative_rmse=a['relative_rmse'],
               mean_fit_relative_rmse=m['relative_rmse'], shape_fit_r2=a['r2'], mean_fit_r2=m['r2'],
               shape_fit_at_bound=a['at_bound'], mean_fit_at_bound=m['at_bound'],
               shape_sign_changes=a['sign_changes'], mean_sign_changes=m['sign_changes'],
               mean_energy_fraction=fraction('mean_energy'), affine_energy_fraction=fraction('affine_energy'),
               nonlinear_energy_fraction=fraction('residual_energy'),
               total_energy_integral=total,
               max_abs_a_linear=float(np.max(np.abs(arrays['a_linear']))),
               max_abs_u_linear=float(np.max(np.abs(arrays['u_linear']))),
               exact_mapping_valid_fraction=float(arrays['exact_mapping_valid'].mean()),
               infinite_horizon_valid=bool(a['kappa'] > 0 and m['kappa'] > -0.5),
               horizon=float(clock['horizon']), fit_distance_min=float(d[selected][0]),
               fit_distance_max=float(d[selected][-1]))
    # Exact parameter mapping is a sensitivity check; never silently drop invalid times.
    for name, key in (('a', 'log_alpha'), ('m', 'u_exact')):
        if np.isfinite(arrays[key][selected]).all():
            f = fit_exponential(d[selected], arrays[key][selected], args.kappa_bounds)
        else:
            f = dict(kappa=np.nan, amplitude=np.nan, relative_rmse=np.nan)
        row[f'exact_kappa_{name}'] = f['kappa']
        row[f'exact_epsilon_{name}'] = f['amplitude']
        row[f'exact_{name}_fit_relative_rmse'] = f['relative_rmse']
    # Fit-window sensitivity: the final two log-variance units and the rest.
    for window, mask in (('late', d <= 2), ('early', d >= 2)):
        for name, key in (('a', 'a_linear'), ('m', 'u_linear')):
            f = fit_exponential(d[mask], arrays[key][mask], args.kappa_bounds) if mask.sum() >= 3 else {'kappa': np.nan}
            row[f'{window}_kappa_{name}'] = f['kappa']
    error = arrays['total_energy']-arrays['mean_energy']-arrays['affine_energy']-arrays['residual_energy']
    row['energy_identity_relative_error'] = float(np.max(np.abs(error)/np.maximum(arrays['total_energy'], 1e-30)))
    row['max_stein_mean_error'] = float(np.max(np.abs(arrays['stein_mean_error'])))
    row['max_stein_linear_error'] = float(np.max(np.abs(arrays['stein_linear_error'])))
    ix = arrays['quadrature_check_indices']
    # L² size of the change in the fitted affine field, divided by total error RMS.
    delta_b = arrays['b'][ix]-arrays['quadrature_check_b']
    delta_C = arrays['C'][ix]-arrays['quadrature_check_C']
    row['quadrature_affine_relative_rms_max'] = float(np.max(np.sqrt(
        (delta_b**2+clock['variance'][ix]*delta_C**2)
        / np.maximum(arrays['quadrature_check_total_energy'], 1e-30))))
    row['quadrature_total_relative_error_max'] = float(np.max(np.abs(
        arrays['total_energy'][ix]-arrays['quadrature_check_total_energy'])
        / np.maximum(arrays['quadrature_check_total_energy'], 1e-30)))
    return row


def compare_kl(record, arrays, row, clock):
    ent = record['entropy']
    saved_gamma = np.asarray(ent['gamma_values'], float)
    gamma = np.unique(np.r_[0., saved_gamma])
    baseline = int(np.flatnonzero(gamma == saved_gamma[0])[0])
    response = profile_response(clock['distance'], arrays['a_linear'], arrays['u_linear'], gamma)
    fitted = exponential_response(row['kappa_a'], row['kappa_m'], row['epsilon_a'], row['epsilon_m'], gamma, clock['horizon'])
    infinite = exponential_response(row['kappa_a'], row['kappa_m'], row['epsilon_a'], row['epsilon_m'], gamma)
    arrays.update(prediction_gamma=gamma, profile_kl=response['kl'], profile_shape_response=response['shape_response'],
                  profile_mean_response=response['mean_response'], exponential_finite_kl=fitted,
                  exponential_infinite_kl=infinite)
    for key in ('gamma_values', 'sde_entropies', 'inv_sde_entropies'):
        arrays[f'empirical_{key}'] = np.asarray(ent[key])
    rows = []
    for i, g in enumerate(saved_gamma):
        j = int(np.flatnonzero(gamma == g)[0])
        item = dict(run=record['run'], epoch=record['epoch'], gamma=float(g),
                    empirical_baseline_gamma=float(saved_gamma[0]))
        for short, key in (('q_p', 'sde_entropies'), ('p_q', 'inv_sde_entropies')):
            values = np.asarray(ent[key], float)
            item[f'empirical_kl_{short}'] = float(values[i])
            item[f'empirical_log10_ratio_{short}'] = float(np.log10(values[i]/values[0]))
        for label, values in (('profile', response['kl']), ('exponential_finite', fitted), ('exponential_infinite', infinite)):
            item[f'{label}_kl'] = float(values[j])
            item[f'{label}_log10_ratio_recorded_baseline'] = float(np.log10(values[j]/values[baseline])) if values[baseline] > 0 else np.nan
            item[f'{label}_log10_ratio_ode'] = float(np.log10(values[j]/values[0])) if values[0] > 0 else np.nan
        rows.append(item)
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--runs', nargs='+', default=['default3', 'default4', 'default5'])
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--quadrature-order', type=int, default=256, help='Nodes per mixture component; also checked at twice this order')
    p.add_argument('--time-points', type=int, default=161, help='Uniform log-variance grid size')
    p.add_argument('--sigma-points', type=int, default=129, help='Additional log-sigma grid size')
    p.add_argument('--sigma-min', type=float, default=0.002)
    p.add_argument('--sigma-max', type=float, default=80.)
    p.add_argument('--kappa-bounds', nargs=2, type=float, default=[-4., 1e6])
    p.add_argument('--fit-distance', nargs=2, type=float, default=[0., np.inf])
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, help='Only evaluate the first N checkpoints, for a smoke run')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    if args.quadrature_order < 8 or args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        p.error('Require quadrature order >= 8 and positive batch size, threads, and limit')
    if len(set(args.runs)) != len(args.runs):
        p.error('Run names must be unique')
    if args.fit_distance[0] < 0 or args.fit_distance[0] >= args.fit_distance[1]:
        p.error('Fit range must satisfy 0 <= MIN < MAX')
    return args


def main():
    args = parse_args()
    args.root, args.output_dir = args.root.resolve(), args.output_dir.resolve()
    torch.set_num_threads(args.threads)
    mixture = GaussianMixture1D()
    clock = variance_clock(mixture, args.sigma_min, args.sigma_max, args.time_points, args.sigma_points)
    records = discover_checkpoints(args.root, args.runs)
    expected_count = len(records)
    if args.limit is not None:
        records = records[:args.limit]
    out = args.output_dir
    (out/'profiles').mkdir(parents=True, exist_ok=True)
    configuration = dict(schema_version=1, runs=args.runs, mixture=mixture.__dict__,
                         quadrature_order=args.quadrature_order, time_points=args.time_points,
                         sigma_points=args.sigma_points, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                         kappa_bounds=args.kappa_bounds, fit_distance=[args.fit_distance[0],
                         args.fit_distance[1] if np.isfinite(args.fit_distance[1]) else 'infinity'],
                         dtype='float64', score_convention='learned minus exact',
                         source_hashes={name: sha256(Path(__file__).parent/name) for name in
                                        ('score_error_modes.py', 'checkpoint_score.py', 'fit_checkpoint_modes.py')})
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    metadata_path = out/'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError(f'{out} contains a different configuration/code version; choose a new --output-dir')
    metadata = dict(configuration, fingerprint=fingerprint, expected_checkpoints=expected_count,
                    completed_checkpoints=0, complete=False, root=str(args.root),
                    source_pdf_sha256=sha256(args.root/'paper_odds/The_odds_favor_noise_in_generative_diffusion_sampling.pdf'),
                    reference_mean=mixture.mean, data_variance=mixture.variance,
                    reference_variance=clock['reference_variance'], horizon=clock['horizon'],
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    approximation='moment-matched Gaussian, first-order response, second-order KL, exact prior',
                    empirical_baseline='smallest recorded gamma, NOT an ODE unless that gamma is zero',
                    limitations=['Mixture dynamics need not follow Gaussian kernels.',
                                 'Large amplitudes, signed/nonexponential profiles and prior/discretization/histogram error can affect agreement.',
                                 '144 checkpoints come from 3 training runs, not 144 independent training replicates.',
                                 'Residual energy is a diagnostic, not a causal sampler ablation.'])
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    np.savez_compressed(out/'clock.npz', **clock)
    summaries, comparisons = [], []
    start = time.perf_counter()
    for i, record in enumerate(records):
        dest = out/'profiles'/f"{record['run']}_epoch{record['epoch']:02d}.npz"
        checkpoint_hash = sha256(record['checkpoint'])
        stats_hash = sha256(record['stats_path'])
        if dest.exists():
            with np.load(dest) as saved:
                arrays = {key: saved[key] for key in saved.files}
            if str(arrays['checkpoint_sha256']) != checkpoint_hash or str(arrays['fingerprint']) != fingerprint or str(arrays['stats_sha256']) != stats_hash:
                raise ValueError(f'Cached profile provenance mismatch: {dest}')
        else:
            arrays = compute_checkpoint(record, mixture, clock, args)
        row = summarize(record, arrays, clock, args)
        comparison = compare_kl(record, arrays, row, clock)
        arrays.update(checkpoint_sha256=checkpoint_hash, stats_sha256=stats_hash, fingerprint=fingerprint)
        temp = dest.with_suffix('.tmp.npz')
        np.savez_compressed(temp, **arrays)
        os.replace(temp, dest)
        summaries.append(row)
        comparisons.extend(comparison)
        print(f"[{i+1:3d}/{len(records)}] {record['run']} epoch {record['epoch']:02d}: "
              f"kappa=({row['kappa_a']:+.3f}, {row['kappa_m']:+.3f}), "
              f"residual={row['nonlinear_energy_fraction']:.1%}; {time.perf_counter()-start:.1f}s", flush=True)
    write_csv(out/'checkpoint_summary.csv', summaries)
    write_csv(out/'kl_comparison.csv', comparisons)
    metadata.update(completed_checkpoints=len(summaries), complete=len(summaries) == expected_count,
                    elapsed_seconds=time.perf_counter()-start)
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    if not args.no_plots:
        from .plot_checkpoint_modes import create_plots
        create_plots(out, args.root/'paper_odds')
    print(f'Wrote {len(summaries)} checkpoint profiles and summaries to {out}', flush=True)


if __name__ == '__main__':
    main()

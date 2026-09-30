"""Run paired nonlinear-residual ablations for all 144 mixture checkpoints.

From the repository root:
python -m paper_odds.score_error_analysis.run_residual_ablation
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint_score import CheckpointScore
from .fit_checkpoint_modes import ROOT, PAPER, DEFAULT_OUTPUT, project_grid, sha256, write_csv
from .score_error_modes import GaussianMixture1D, profile_response
from .residual_ablation import (
    AblationField, binned_kl, gaussian_moment_kl, histogram_counts, sampling_grid,
    sample_paired, shared_initial_states, target_partition, uniform_lookup,
)

DEFAULT_ABLATION_OUTPUT = PAPER/'outputs/residual_ablation'


def build_field(model, mixture, config):
    """Fresh least squares at every integration node, plus a checked score table."""
    clock = sampling_grid(mixture, config['sigma_min'], config['sigma_max'], config['steps'])
    projection = project_grid(mixture, clock['sigma'], model, config['quadrature_order'], 8192)
    nodes = config['table_nodes']
    for attempt in range(3):
        zgrid = np.linspace(-config['z_limit'], config['z_limit'], nodes)
        x = mixture.mean+clock['sqrt_variance'][:, None]*zgrid
        learned = model(x, clock['sigma'][:, None])*clock['sqrt_variance'][:, None]
        field = AblationField(mixture, clock, projection['b'], projection['C'], learned, model, config['z_limit'])
        selected = np.unique(np.linspace(0, config['steps'], 11).round().astype(int))
        check_x, weights = mixture.quadrature(clock['sigma'][selected], config['quadrature_order'])
        true_learned = model(check_x, clock['sigma'][selected, None])
        exact = mixture.score(check_x, clock['sigma'][selected, None])
        errors = []
        for j, index in enumerate(selected):
            z = (check_x[j]-mixture.mean)/clock['sqrt_variance'][index]
            mask = np.abs(z) <= config['z_limit']
            approx = true_learned[j].copy()
            approx[mask] = uniform_lookup(z[mask], learned[index], -config['z_limit'], config['z_limit'])/clock['sqrt_variance'][index]
            rms2 = float(weights@((true_learned[j]-exact[j])**2))
            errors.append(float(np.sqrt(weights@((approx-true_learned[j])**2)/max(rms2, 1e-30))))
        if max(errors) <= config['table_tolerance']:
            return field, projection, dict(nodes=nodes, relative_rms_error_max=max(errors), checked_time_indices=selected.tolist())
        nodes = 2*nodes-1
    raise RuntimeError(f'Score-table error {max(errors):g} exceeds {config["table_tolerance"]}; increase --table-nodes')


def summarize_samples(samples, mixture, sigma_min, bins):
    result = dict(sample_mean=samples.mean(axis=-1), sample_variance=samples.var(axis=-1))
    for nbin in bins:
        edges = target_partition(mixture, sigma_min, nbin)
        result[f'counts_{nbin}'] = histogram_counts(samples, edges)
    return result


def compute_record(record, config, fingerprint, destination):
    torch.set_num_threads(1)
    destination = Path(destination)
    checkpoint = Path(config['root'])/record['checkpoint']
    checkpoint_hash = sha256(checkpoint)
    if destination.exists():
        with np.load(destination) as old:
            if str(old['fingerprint']) != fingerprint or str(old['checkpoint_sha256']) != checkpoint_hash:
                raise ValueError(f'Cached ablation provenance mismatch: {destination}')
        return dict(run=record['run'], epoch=record['epoch'], reused=True)
    start = time.perf_counter()
    mixture = GaussianMixture1D(**config['mixture'])
    model = CheckpointScore(checkpoint)
    if model.epoch != record['epoch']:
        raise ValueError('Checkpoint epoch differs from the requested epoch')
    field, projection, validation = build_field(model, mixture, config)
    initial = shared_initial_states(mixture, config['sigma_max'], config['particles'], config['seeds'], config['prior'])
    samples = sample_paired(field, config['gammas'], config['lambdas'], initial, config['seeds'],
                            integrator=config.get('sampler_integrator', 'heun'))
    arrays = summarize_samples(samples, mixture, config['sigma_min'], config['bins'])
    V = field.clock['variance']
    d = field.clock['ell'][-1]-field.clock['ell'][::-1]
    predicted = profile_response(d, (V*projection['C'])[::-1],
                                 (V*projection['b']/np.sqrt(V[-1]))[::-1], config['gammas'])
    initial_r = 0. if config['prior']=='exact' else -mixture.mean/np.sqrt(V[-1])
    initial_v = 1. if config['prior']=='exact' else config['sigma_max']**2/V[0]
    gamma = np.asarray(config['gammas'])
    horizon = field.clock['ell'][-1]
    predicted['shape_response'] += (initial_v-1)*np.exp(-gamma*horizon)
    predicted['mean_response'] += initial_r*np.exp(-(1+gamma)/2*horizon)
    predicted['kl'] = predicted['shape_response']**2/4+predicted['mean_response']**2/2
    moment_kl, r, v = gaussian_moment_kl(field.clock, projection['b'], projection['C'],
                                       config['gammas'], initial_r, initial_v,
                                       integrator=config.get('moment_integrator', 'adaptive'))
    arrays.update(b=projection['b'], C=projection['C'], sigma=field.clock['sigma'],
                  gaussian_profile_kl=predicted['kl'], gaussian_moment_kl=moment_kl,
                  gaussian_terminal_mean_error=r, gaussian_terminal_relative_variance=v,
                  table_nodes=validation['nodes'], table_relative_rms_error_max=validation['relative_rms_error_max'],
                  fallback_points=field.fallback_points, max_abs_z=field.max_abs_z,
                  checkpoint_sha256=checkpoint_hash, fingerprint=fingerprint,
                  elapsed_seconds=time.perf_counter()-start)
    temp = destination.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, destination)
    return dict(run=record['run'], epoch=record['epoch'], seconds=round(time.perf_counter()-start, 2),
                table_error=validation['relative_rms_error_max'], fallback_points=field.fallback_points)


def summarize_results(records, config, out):
    """Pooled estimates plus seed-level paired jackknife Monte Carlo errors."""
    gammas = np.asarray(config['gammas'])
    baseline = int(np.flatnonzero(gammas == 0)[0])
    affine = config['lambdas'].index(0.)
    full = config['lambdas'].index(1.)
    seeds = len(config['seeds'])
    result_rows, effect_rows, replicate_rows = [], [], []
    with np.load(out/'exact_control.npz') as control:
        exact = {k: control[k] for k in control.files}
    for record in records:
        with np.load(out/'checkpoints'/f"{record['run']}_epoch{record['epoch']:02d}.npz") as f:
            data = {k: f[k] for k in f.files}
        for nbin in config['bins']:
            counts = data[f'counts_{nbin}']  # gamma, lambda, seed, bin
            pooled_counts = counts.sum(axis=2)
            kl = binned_kl(pooled_counts, config['pseudocount'])
            exact_kl = binned_kl(exact[f'counts_{nbin}'][:, 0].sum(axis=1), config['pseudocount'])
            log_ratio = np.log10(kl/kl[baseline])
            profile = data['gaussian_profile_kl']
            prediction = np.log10(profile/profile[baseline])
            moment = data['gaussian_moment_kl']
            moment_prediction = np.log10(moment/moment[baseline])
            leave_one = binned_kl(pooled_counts[:, :, None, :]-counts, config['pseudocount'])
            leave_ratio = np.log10(leave_one/leave_one[baseline])
            leave_effect = leave_ratio[:, full]-leave_ratio[:, affine]
            effect_se = np.sqrt((seeds-1)/seeds*np.sum((leave_effect-leave_effect.mean(axis=1, keepdims=True))**2, axis=1))
            leave_gain = abs(leave_ratio[:, full]-prediction[:, None, None])-abs(leave_ratio[:, affine]-prediction[:, None, None])
            gain_se = np.sqrt((seeds-1)/seeds*np.sum((leave_gain-leave_gain.mean(axis=1, keepdims=True))**2, axis=1))
            per_seed_kl = binned_kl(counts, config['pseudocount'])
            for i, g in enumerate(gammas):
                for j, lam in enumerate(config['lambdas']):
                    row = dict(run=record['run'], epoch=record['epoch'], gamma=float(g), residual_lambda=lam,
                               bins=nbin, pooled_particles=config['particles']*seeds)
                    for k, direction in enumerate(('q_p', 'p_q')):
                        row[f'kl_{direction}'] = float(kl[i,j,k])
                        row[f'log10_ratio_ode_{direction}'] = float(log_ratio[i,j,k])
                        row[f'exact_control_kl_{direction}'] = float(exact_kl[i,k])
                        row[f'exact_control_ode_kl_{direction}'] = float(exact_kl[baseline,k])
                        row[f'gaussian_profile_log10_ratio_ode_{direction}'] = float(prediction[i])
                        row[f'gaussian_moment_log10_ratio_ode_{direction}'] = float(moment_prediction[i,k])
                        row[f'above_control_floor_{direction}'] = bool(kl[i,j,k] > 3*exact_kl[i,k] and kl[baseline,j,k] > 3*exact_kl[baseline,k])
                    row.update(table_relative_rms_error_max=float(data['table_relative_rms_error_max']),
                               table_nodes=int(data['table_nodes']), fallback_points=int(data['fallback_points']))
                    result_rows.append(row)
                    for rep, seed in enumerate(config['seeds']):
                        replicate_rows.append(dict(run=record['run'], epoch=record['epoch'], gamma=float(g),
                                                   residual_lambda=lam, bins=nbin, seed=seed,
                                                   kl_q_p=float(per_seed_kl[i,j,rep,0]), kl_p_q=float(per_seed_kl[i,j,rep,1])))
                for k, direction in enumerate(('q_p', 'p_q')):
                    full_gap = abs(log_ratio[i,full,k]-prediction[i])
                    affine_gap = abs(log_ratio[i,affine,k]-prediction[i])
                    effect_rows.append(dict(run=record['run'], epoch=record['epoch'], gamma=float(g), bins=nbin,
                                            kl_direction=direction, full_log10_ratio=float(log_ratio[i,full,k]),
                                            affine_log10_ratio=float(log_ratio[i,affine,k]),
                                            nonlinear_effect_log10_ratio=float(log_ratio[i,full,k]-log_ratio[i,affine,k]),
                                            nonlinear_effect_mc_se=float(effect_se[i,k]),
                                            full_profile_disagreement=float(full_gap), affine_profile_disagreement=float(affine_gap),
                                            disagreement_reduction=float(full_gap-affine_gap), disagreement_reduction_mc_se=float(gain_se[i,k]),
                                            affine_gaussian_moment_disagreement=float(abs(log_ratio[i,affine,k]-moment_prediction[i,k])),
                                            gaussian_profile_log10_ratio=float(prediction[i]), gaussian_moment_log10_ratio=float(moment_prediction[i,k]),
                                            both_arms_above_control_floor=bool(np.all(kl[[i,baseline]][:,[affine,full],k] > 3*exact_kl[[i,baseline],None,k]))))
    write_csv(out/'ablation_results.csv', result_rows)
    write_csv(out/'paired_effects.csv', effect_rows)
    write_csv(out/'replicate_results.csv', replicate_rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--profiles-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_ABLATION_OUTPUT)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--gammas', type=float, nargs='+', default=[0., .01, .2, 1., 5.])
    p.add_argument('--lambdas', type=float, nargs='+', default=[0., 1.])
    p.add_argument('--seeds', type=int, nargs='+', default=[170, 271, 372, 473])
    p.add_argument('--particles', type=int, default=4096, help='Particles per seed and arm')
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--integrator', choices=['heun', 'euler'], default='heun',
                   help='euler uses Euler for the ODE and Euler-Maruyama for the SDE; also Euler for Gaussian moments')
    p.add_argument('--table-nodes', type=int, default=1025)
    p.add_argument('--table-tolerance', type=float, default=.003)
    p.add_argument('--z-limit', type=float, default=12.)
    p.add_argument('--quadrature-order', type=int, default=256)
    p.add_argument('--prior', choices=['exact', 'gaussian'], default='exact')
    p.add_argument('--bins', type=int, nargs='+', default=[32,64,128])
    p.add_argument('--pseudocount', type=float, default=.5)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--checkpoint-ids', nargs='+', help='Explicit run:epoch pairs for validation')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    if 0. not in args.gammas or any(g < 0 or not np.isfinite(g) for g in args.gammas) or len(set(args.gammas)) != len(args.gammas):
        p.error('Gammas must be unique, finite, nonnegative, and include zero')
    if 0. not in args.lambdas or 1. not in args.lambdas or any(not 0 <= v <= 1 for v in args.lambdas) or len(set(args.lambdas)) != len(args.lambdas):
        p.error('Lambdas must be unique in [0,1], including 0 and 1')
    if len(args.seeds) < 2 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        p.error('Use at least two distinct nonnegative seeds')
    if args.particles < 100 or args.steps < 2 or args.table_nodes < 33 or args.quadrature_order < 8 or args.workers < 1:
        p.error('Require particles>=100, steps>=2, table nodes>=33, quadrature>=8, workers>=1')
    if args.table_tolerance <= 0 or args.z_limit <= 0 or args.pseudocount <= 0 or min(args.bins) < 4 or len(set(args.bins)) != len(args.bins):
        p.error('Use positive tolerance, extent and pseudocount, with distinct bin counts >=4')
    if args.limit is not None and args.limit < 1:
        p.error('Limit must be positive')
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    profiles_dir, out = args.profiles_dir.resolve(), args.output_dir.resolve()
    meta = json.loads((profiles_dir/'metadata.json').read_text())
    if not meta['complete']:
        raise ValueError('Complete the mode analysis first')
    with (profiles_dir/'checkpoint_summary.csv').open() as f:
        records = [dict(run=r['run'], epoch=int(r['epoch']), checkpoint=r['checkpoint']) for r in csv.DictReader(f)]
    available = len(records)
    if args.checkpoint_ids:
        wanted = set(args.checkpoint_ids)
        records = [r for r in records if f"{r['run']}:{r['epoch']}" in wanted]
        if len(records) != len(wanted):
            raise ValueError('Some requested checkpoint IDs were not found')
    if args.limit:
        records = records[:args.limit]
    config = dict(schema_version=1, root=str(args.root.resolve()), mixture=meta['mixture'],
                  profile_fingerprint=meta['fingerprint'], checkpoint_ids=[f"{r['run']}:{r['epoch']}" for r in records],
                  sigma_min=meta['sigma_min'], sigma_max=meta['sigma_max'], steps=args.steps,
                  sampler_integrator=args.integrator,
                  moment_integrator='euler' if args.integrator == 'euler' else 'adaptive',
                  particles=args.particles, seeds=args.seeds, gammas=args.gammas, lambdas=args.lambdas,
                  prior=args.prior, table_nodes=args.table_nodes, table_tolerance=args.table_tolerance,
                  z_limit=args.z_limit, quadrature_order=args.quadrature_order, bins=args.bins, pseudocount=args.pseudocount,
                  source_hashes={name:sha256(Path(__file__).parent/name) for name in
                                 ('residual_ablation.py','run_residual_ablation.py','checkpoint_score.py','score_error_modes.py')})
    fingerprint = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    (out/'checkpoints').mkdir(parents=True,exist_ok=True)
    manifest_path = out/'metadata.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Output contains a different experiment configuration; choose another output directory')
    manifest = dict(config, fingerprint=fingerprint, available_checkpoints=available, requested_checkpoints=len(records),
                    completed_checkpoints=0, complete=False, all_available_checkpoints=len(records)==available,
                    estimator='fixed exact-target quantile partition, pooled counts, Jeffreys pseudocount',
                    integrator=('Euler (gamma=0) / Euler-Maruyama (gamma>0) in standardized log-variance coordinates'
                                if args.integrator == 'euler' else
                                'additive-noise stochastic Heun in standardized log-variance coordinates'),
                    uncertainty='paired delete-one-seed jackknife Monte Carlo standard error; not training-run uncertainty',
                    target='diffused mixture at sigma_min; no final step to zero',
                    pairing='same initial states and Brownian increments across lambda/gamma/checkpoints',
                    limitations=['Binned KL is not continuous KL; inspect all bin counts and exact-score control.',
                                 'Monte Carlo standard errors do not quantify variability across training runs.',
                                 'Gaussian profile and Gaussian moment predictions are surrogate comparisons for mixture dynamics.'])
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    mixture = GaussianMixture1D(**config['mixture'])
    clock = sampling_grid(mixture,config['sigma_min'],config['sigma_max'],config['steps'])
    control_path = out/'exact_control.npz'
    if control_path.exists():
        with np.load(control_path) as old:
            if str(old['fingerprint']) != fingerprint:
                raise ValueError('Exact control provenance mismatch')
    else:
        field = AblationField(mixture,clock,np.zeros(len(clock['sigma'])),np.zeros(len(clock['sigma'])))
        initial = shared_initial_states(mixture,config['sigma_max'],config['particles'],config['seeds'],config['prior'])
        samples = sample_paired(field,config['gammas'],[None],initial,config['seeds'],
                                integrator=config['sampler_integrator'])
        np.savez_compressed(control_path, **summarize_samples(samples,mixture,config['sigma_min'],config['bins']), fingerprint=fingerprint)
    print(f'Exact-score control ready; running {len(records)} checkpoints with {args.workers} workers.',flush=True)
    start = time.perf_counter()
    if args.workers == 1:
        for index,record in enumerate(records):
            result=compute_record(record,config,fingerprint,out/'checkpoints'/f"{record['run']}_epoch{record['epoch']:02d}.npz")
            print(f'[{index+1}/{len(records)}] {result}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(compute_record,r,config,fingerprint,out/'checkpoints'/f"{r['run']}_epoch{r['epoch']:02d}.npz") for r in records]
            for index,future in enumerate(as_completed(futures)):
                print(f'[{index+1}/{len(records)}] {future.result()}',flush=True)
    summarize_results(records,config,out)
    manifest.update(completed_checkpoints=len(records),complete=True,elapsed_seconds=time.perf_counter()-start)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if not args.no_plots:
        from .plot_residual_ablation import create_plots
        create_plots(out)
    print(f'Completed paired residual ablation: {out}',flush=True)


if __name__ == '__main__':
    main()

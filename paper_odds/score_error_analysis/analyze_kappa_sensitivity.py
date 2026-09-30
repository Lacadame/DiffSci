"""Compare pointwise, conservatively smoothed, and cumulative kappa estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .compare_exponential_predictions import (
    PAPER, agreement_metrics, build_comparisons, exponential_log_ratio, write_csv,
)
from .kappa_estimators import (
    cumulative_basis, fit_cumulative, primitive_at, smooth_profile, smoothed_mode_response,
)
from .plot_checkpoint_modes import load_summary, measured_markers, phase_overlays, plt, Normalize, save_figure
from .score_error_modes import fit_exponential, integration_weights, profile_response


def mode_response(kappa, amplitude, gamma, horizon, mode):
    gamma = np.asarray(gamma)
    rates = gamma if mode == 'shape' else (1+gamma)/2
    factors = 1+gamma if mode == 'shape' else (1+gamma)/2
    return amplitude*factors*np.array([cumulative_basis(kappa+r, horizon) for r in rates])


def analyze(paper, out, bandwidths=(.01, .05, .2)):
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = load_summary(paper/'outputs/score_error_modes')
    with np.load(paper/'outputs/score_error_modes/clock.npz') as clock:
        d = clock['distance']; horizon = float(clock['horizon'])
    w = integration_weights(d)
    gamma = np.array([0., .2, 1., 5.])
    fit_rows, coordinates, sources = [], [], []
    for number, cp in enumerate(checkpoints, 1):
        path = paper/'outputs/score_error_modes/profiles'/f"{cp['run']}_epoch{int(cp['epoch']):02d}.npz"
        sources.append(path)
        with np.load(path) as data:
            profiles = {mode: data[key] for mode, key in [('shape', 'a_linear'), ('mean', 'u_linear')]}
        reference = profile_response(d, profiles['shape'], profiles['mean'], gamma)
        fits = {}
        for mode, suffix in [('shape', 'a'), ('mean', 'm')]:
            y = profiles[mode]
            reference_response = reference[f'{mode}_response']
            fits[mode] = dict(original=dict(kappa=cp[f'kappa_{suffix}'], amplitude=cp[f'epsilon_{suffix}'],
                relative_rmse=cp[f'{mode}_fit_relative_rmse'], at_bound=cp[f'{mode}_fit_at_bound']))
            smoothing = {}
            for bandwidth in bandwidths:
                method = f'smooth_{bandwidth:g}'
                smoothed = smooth_profile(d, y, bandwidth)
                fits[mode][method] = fit_exponential(smoothed['distance'], smoothed['profile'], weights=smoothed['weights'])
                changed = smoothed_mode_response(smoothed, gamma, mode=mode)-reference_response
                smoothing[method] = dict(
                    smoothing_response_relative_change=float(np.linalg.norm(changed)/max(np.linalg.norm(reference_response), 1e-30)),
                    smoothing_signed_area_drift=float(abs(smoothed['smoothed_area']-smoothed['original_area'])/max(w@abs(y), 1e-30)))
            fits[mode]['cumulative'] = fit_cumulative(d, y, mode=mode)
            for method, fit in fits[mode].items():
                response = mode_response(fit['kappa'], fit['amplitude'], gamma, horizon, mode)
                fit_rows.append(dict(run=cp['run'], epoch=int(cp['epoch']), mode=mode, method=method,
                    kappa=fit['kappa'], amplitude=fit['amplitude'], relative_rmse=fit['relative_rmse'],
                    objective='cumulative signed ODE mode response' if method == 'cumulative' else 'pointwise signed profile',
                    at_bound=fit['at_bound'],
                    loss_envelope_low=fit.get('loss_envelope_low', np.nan),
                    loss_envelope_high=fit.get('loss_envelope_high', np.nan),
                    upper_unresolved=fit.get('upper_unresolved', False),
                    cancellation_ratio=fit.get('cancellation_ratio', np.nan),
                    kernel_response_relative_error=float(np.linalg.norm(response-reference_response)/max(np.linalg.norm(reference_response), 1e-30)),
                    terminal_l2_fraction=float(primitive_at(d, y*y, .01)/(w@y**2)) if w@y**2 else 0.,
                    terminal_absolute_area_fraction=float(primitive_at(d, abs(y), .01)/(w@abs(y))) if w@abs(y) else 0.,
                    smoothing_response_relative_change=smoothing.get(method, {}).get('smoothing_response_relative_change', 0.),
                    smoothing_signed_area_drift=smoothing.get(method, {}).get('smoothing_signed_area_drift', 0.)))
        for method in fits['shape']:
            a, m = fits['shape'][method], fits['mean'][method]
            coordinates.append(dict(run=cp['run'], epoch=int(cp['epoch']), method=method,
                kappa_a=a['kappa'], kappa_m=m['kappa'], epsilon_a=a['amplitude'], epsilon_m=m['amplitude'],
                amplitude_ratio=abs(m['amplitude']/a['amplitude']) if a['amplitude'] else np.inf,
                shape_fit_relative_rmse=a['relative_rmse'], mean_fit_relative_rmse=m['relative_rmse'],
                shape_fit_at_bound=a['at_bound'], mean_fit_at_bound=m['at_bound'], horizon=horizon,
                upper_unresolved_a=a.get('upper_unresolved', False), upper_unresolved_m=m.get('upper_unresolved', False),
                in_original_axes=bool(0 <= a['kappa'] <= 2 and -.5 <= m['kappa'] <= 2),
                infinite_horizon_valid=bool(a['kappa'] > 0 and m['kappa'] > -.5)))
        if number % 24 == 0:
            print(f'Fitted sensitivity estimators for {number}/{len(checkpoints)} checkpoints', flush=True)
    write_csv(out/'mode_fits.csv', fit_rows)
    write_csv(out/'checkpoint_coordinates.csv', coordinates)
    summary = {}
    for method in fits['shape']:
        group = [r for r in coordinates if r['method'] == method]
        summary[method] = dict(checkpoints=len(group), inside_original_axes=sum(r['in_original_axes'] for r in group),
            infinite_horizon_valid=sum(r['infinite_horizon_valid'] for r in group))
        for mode in ('shape', 'mean'):
            rows = [r for r in fit_rows if r['mode'] == mode and r['method'] == method]
            values = np.array([r['kappa'] for r in rows])
            summary[method][mode] = dict(median_kappa=float(np.median(values)), min_kappa=float(min(values)),
                max_kappa=float(max(values)), kappa_above_100=int(sum(values > 100)),
                median_kernel_response_relative_error=float(np.median([r['kernel_response_relative_error'] for r in rows])),
                median_smoothing_response_relative_change=float(np.median([r['smoothing_response_relative_change'] for r in rows])),
                max_smoothing_signed_area_drift=max(r['smoothing_signed_area_drift'] for r in rows),
                upper_unresolved=sum(r['upper_unresolved'] for r in rows))
    return checkpoints, fit_rows, coordinates, summary, sources


def compare_measurements(paper, coordinates, out):
    measurements, _ = build_comparisons(paper)
    measurements = [r for r in measurements if r['bins'] in (0, 64)]
    lookup = {(r['run'], r['epoch'], r['method']): r for r in coordinates}
    methods = list(dict.fromkeys(r['method'] for r in coordinates))
    comparisons = []
    for measured in measurements:
        for method in methods:
            cp = lookup[measured['run'], measured['epoch'], method]
            comparisons.append(dict(dataset=measured['dataset'], run=measured['run'], epoch=measured['epoch'],
                gamma=measured['gamma'], baseline_gamma=measured['baseline_gamma'], bins=measured['bins'],
                kl_direction=measured['kl_direction'], method=method, above_control_floor=measured['above_control_floor'],
                measured_log10_ratio=measured['measured_log10_ratio'],
                profile_log10_ratio=measured['profile_log10_ratio'],
                predicted_log10_ratio=exponential_log_ratio(cp, measured['gamma'], measured['baseline_gamma'])))
    write_csv(out/'predictions_vs_measurements.csv', comparisons)
    groups = defaultdict(list)
    for r in comparisons:
        groups[r['dataset'], r['gamma'], r['kl_direction'], r['method']].append(r)
    statistics = []
    for (dataset, gamma, direction, method), group in sorted(groups.items()):
        subsets = {'all': group}
        if dataset != 'saved_sweep':
            subsets['above_control_floor'] = [r for r in group if r['above_control_floor']]
        for subset, selected in subsets.items():
            for run in ('all', 'default3', 'default4', 'default5'):
                chosen = [r for r in selected if run == 'all' or r['run'] == run]
                metrics = agreement_metrics([r['predicted_log10_ratio'] for r in chosen],
                    [r['measured_log10_ratio'] for r in chosen], [r['run'] for r in chosen])
                statistics.append(dict(dataset=dataset, gamma=gamma, kl_direction=direction, method=method,
                    run=run, subset=subset, baseline_gamma=group[0]['baseline_gamma'], bins=group[0]['bins'], **metrics))
    write_csv(out/'correlations.csv', statistics)
    return statistics


def make_figures(paper, out, checkpoints, fit_rows, coordinates, statistics):
    methods = list(dict.fromkeys(r['method'] for r in coordinates))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, mode in zip(axes, ('shape', 'mean')):
        for j, method in enumerate(methods):
            values = np.sort([r['kappa'] for r in fit_rows if r['mode'] == mode and r['method'] == method])
            ax.plot(values, np.arange(1, len(values)+1)/len(values), label=method, lw=2)
        ax.set_xscale('symlog', linthresh=1)
        ax.set(xlabel=f'{mode.capitalize()} kappa (symmetric-log scale)', ylabel='Fraction of checkpoints', title=f'{mode.capitalize()} localization')
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('Kappa sensitivity: original squared-error fit, area-conserving smoothing, cumulative response fit')
    save_figure(fig, out, 'kappa_distributions')

    clock = np.load(paper/'outputs/score_error_modes/clock.npz'); d = clock['distance']
    # Inspect both a large shape estimate and the largest mean estimate.
    examples = [(max(checkpoints, key=lambda r: r['kappa_a']), 'shape', 'a_linear', 'a'),
                (max(checkpoints, key=lambda r: r['kappa_m']), 'mean', 'u_linear', 'm')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    for i, (cp, mode, key, suffix) in enumerate(examples):
        with np.load(paper/'outputs/score_error_modes/profiles'/f"{cp['run']}_epoch{int(cp['epoch']):02d}.npz") as z:
            y = z[key]
        smooth = smooth_profile(d, y, .05)
        beta = 0. if mode == 'shape' else .5
        primitive = primitive_at(d, np.exp(-beta*d)*y, d)
        axes[i, 0].plot(d, y, c='black', label='original signed profile')
        axes[i, 0].plot(smooth['distance'], smooth['profile'], label='smoothed, bandwidth 0.05')
        axes[i, 1].plot(d, primitive, c='black', lw=2, label='actual cumulative ODE mode response')
        for method in ('original', 'smooth_0.05', 'cumulative'):
            row = next(r for r in coordinates if r['run'] == cp['run'] and r['epoch'] == cp['epoch'] and r['method'] == method)
            amplitude, k = row[f'epsilon_{suffix}'], row[f'kappa_{suffix}']
            axes[i, 1].plot(d, amplitude*cumulative_basis(k+beta, d), '--', label=f'{method}: k={k:.3g}')
            axes[i, 2].plot(d, amplitude*np.exp(-k*d), label=f'{method}: k={k:.3g}')
        axes[i, 2].plot(d, y, c='black', alpha=.6, label='original profile')
        for j in (0, 2):
            axes[i, j].set_xscale('symlog', linthresh=.001)
            axes[i, j].set_yscale('symlog', linthresh=.01)
        for ax in axes[i]:
            ax.set_xlabel('Log-variance distance d (0 = terminal time)')
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        axes[i, 0].set_title(f"{cp['run']}, epoch {int(cp['epoch'])}: {mode}; original k={cp[f'kappa_{suffix}']:.0f}")
        axes[i, 1].set_title('Cumulative signed response (linear axes)')
        axes[i, 2].set_title('Fitted profiles (symmetric-log axes)')
    fig.suptitle('Large kappa fits chase narrow terminal peaks; cumulative fits target integrated signed effects')
    save_figure(fig, out, 'large_kappa_profiles')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for i, direction in enumerate(('q_p', 'p_q')):
        for j, gamma in enumerate((.2, 1., 5.)):
            ax = axes[i, j]
            for dataset, label in [('controlled_full', 'full score'), ('controlled_affine', 'affine only')]:
                vals = [next(r['spearman'] for r in statistics if r['dataset'] == dataset and r['gamma'] == gamma
                        and r['kl_direction'] == direction and r['method'] == m and r['run'] == 'all' and r['subset'] == 'all') for m in methods]
                ax.plot(range(len(methods)), vals, 'o-', label=label)
            ax.set_xticks(range(len(methods)), methods, rotation=30, ha='right')
            ax.set(title=f'gamma={gamma:g}; {direction}', ylabel='Spearman with measured log10 KL ratio', ylim=(-.1, 1))
            ax.axhline(0, c='.7', lw=.6); ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.suptitle('Finite-horizon predictions from alternative kappa fits; measured ODE baseline, 64 bins\nNo measured KL values used in fitting; cumulative objective uses the signed ODE mode responses')
    save_figure(fig, out, 'estimator_correlations')

    # New coordinates get separate overlays; original phase artifacts remain intact.
    for method in ('smooth_0.05', 'cumulative'):
        if method not in methods:
            continue
        destination = out/method; destination.mkdir(exist_ok=True)
        rows = [r for r in coordinates if r['method'] == method]
        phase_overlays(rows, destination, paper)


def validate_resolution(paper, coordinates):
    """Double smoothing bins and refine the cumulative objective's time grid."""
    # Smoothing resolution is independent of model inference: double rebin density.
    cp_ids = sorted(set((r['run'], r['epoch']) for r in coordinates if r['epoch'] in (0, 24, 47)))
    d = np.load(paper/'outputs/score_error_modes/clock.npz')['distance']
    checks = []
    lookup = {(r['run'], r['epoch'], r['method']): r for r in coordinates}
    for run, epoch in cp_ids:
        with np.load(paper/'outputs/score_error_modes/profiles'/f'{run}_epoch{epoch:02d}.npz') as data:
            for mode, key, suffix in [('shape', 'a_linear', 'a'), ('mean', 'u_linear', 'm')]:
                y = data[key]
                for bandwidth in (.01, .05, .2):
                    method = f'smooth_{bandwidth:g}'
                    if (run, epoch, method) not in lookup:
                        continue
                    finer = smooth_profile(d, y, bandwidth, points_per_bandwidth=16)
                    fit = fit_exponential(finer['distance'], finer['profile'], weights=finer['weights'])
                    old = lookup[run, epoch, method]
                    change = abs(fit['kappa']-old[f'kappa_{suffix}'])/(1+abs(old[f'kappa_{suffix}']))
                    checks.append(dict(run=run, epoch=epoch, mode=mode, method=method, scaled_kappa_change=change))
                fine_d = np.sort(np.r_[d, (d[:-1]+d[1:])/2])
                fit = fit_cumulative(fine_d, np.interp(fine_d, d, y), mode=mode)
                old = lookup[run, epoch, 'cumulative']
                change = abs(fit['kappa']-old[f'kappa_{suffix}'])/(1+abs(old[f'kappa_{suffix}']))
                checks.append(dict(run=run, epoch=epoch, mode=mode, method='cumulative', scaled_kappa_change=change))
    return dict(check='Twice the smoothing-bin density; cumulative quadrature on a midpoint-refined piecewise-linear profile. No additional network evaluations.',
                passed=all(r['scaled_kappa_change'] < .02 for r in checks),
                max_scaled_kappa_change=max(r['scaled_kappa_change'] for r in checks), checks=checks)


def run(paper=PAPER, output_dir=None):
    paper = Path(paper); out = Path(output_dir) if output_dir else paper/'outputs/kappa_sensitivity'
    checkpoints, fits, coordinates, summary, sources = analyze(paper, out)
    statistics = compare_measurements(paper, coordinates, out)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    make_figures(paper, out, checkpoints, fits, coordinates, statistics)
    resolution = validate_resolution(paper, coordinates)
    (out/'resolution_validation.json').write_text(json.dumps(resolution, indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    sources += [paper/p for p in ('outputs/score_error_modes/clock.npz', 'outputs/score_error_modes/checkpoint_summary.csv',
        'outputs/score_error_modes/kl_comparison.csv', 'outputs/residual_ablation/ablation_results.csv',
        'score_error_analysis/kappa_estimators.py', 'score_error_analysis/analyze_kappa_sensitivity.py',
        'score_error_analysis/score_error_modes.py', 'score_error_analysis/compare_exponential_predictions.py',
        'score_error_analysis/plot_checkpoint_modes.py', 'plot_phase_diagrams.py', 'outputs/phase_diagrams/metadata.json')]
    meta = dict(complete=resolution['passed'], checkpoints=len(checkpoints), bandwidths=[.01, .05, .2],
        bounds=[-4., 1e6], smoothing='Area-conserving uniform d-bin averages; reflecting Gaussian, 8 bins per standard deviation',
        cumulative='Signed cumulative ODE response: kernel 1 for shape, exp(-d/2) for mean; uniform d loss weights',
        identifiability='Grid envelope with normalized MSE at most 0.01 above optimum; not a confidence interval',
        original_results_preserved=True, used_measured_kl_for_fitting=False,
        source_sha256={str(p.relative_to(paper)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        resolution_validation_passed=resolution['passed'])
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    print('Smoothing resolution:', resolution['passed'], resolution['max_scaled_kappa_change'], flush=True)
    if not resolution['passed']:
        raise RuntimeError('Inspect failed smoothing resolution checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path, default=PAPER)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.paper_dir, args.output_dir)

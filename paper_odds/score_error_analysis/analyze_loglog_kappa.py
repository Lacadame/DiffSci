"""Fit log-log slopes for all checkpoints and evaluate their KL predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze_kappa_sensitivity import compare_measurements
from .compare_exponential_predictions import PAPER, write_csv
from .loglog_kappa import fit_loglog
from .plot_checkpoint_modes import load_summary, phase_overlays, plt, save_figure


VARIANTS = {
    'loglog': ('log_variance', 1e-6),
    'loglog_cutoff_1e-8': ('log_variance', 1e-8),
    'loglog_cutoff_1e-4': ('log_variance', 1e-4),
    'loglog_unweighted': ('points', 1e-6),
}


def analyze(paper, out):
    checkpoints = load_summary(paper/'outputs/score_error_modes')
    with np.load(paper/'outputs/score_error_modes/clock.npz') as clock:
        d = clock['distance']
    sources, fits, coordinates = [], [], []
    for cp in checkpoints:
        path = paper/'outputs/score_error_modes/profiles'/f"{cp['run']}_epoch{int(cp['epoch']):02d}.npz"
        sources.append(path)
        with np.load(path) as z:
            profiles = {mode: z[key] for mode, key in [('shape', 'a_linear'), ('mean', 'u_linear')]}
        coordinates.append(dict(run=cp['run'], epoch=int(cp['epoch']), method='original',
            kappa_a=cp['kappa_a'], kappa_m=cp['kappa_m'], epsilon_a=cp['epsilon_a'], epsilon_m=cp['epsilon_m'],
            amplitude_ratio=cp['amplitude_ratio'], horizon=cp['horizon'],
            shape_fit_relative_rmse=cp['shape_fit_relative_rmse'], mean_fit_relative_rmse=cp['mean_fit_relative_rmse'],
            shape_fit_at_bound=cp['shape_fit_at_bound'], mean_fit_at_bound=cp['mean_fit_at_bound'],
            mixed_sign_a=bool(cp['shape_sign_changes']), mixed_sign_m=bool(cp['mean_sign_changes']),
            in_original_axes=bool(0 <= cp['kappa_a'] <= 2 and -.5 <= cp['kappa_m'] <= 2),
            infinite_horizon_valid=cp['infinite_horizon_valid']))
        for method, (weighting, cutoff) in VARIANTS.items():
            pair = {}
            for mode, y in profiles.items():
                fit = fit_loglog(d, y, weighting=weighting, relative_cutoff=cutoff)
                if not fit['valid']:
                    raise ValueError(f'Undefined log-log fit: {cp["run"]}, {cp["epoch"]}, {mode}, {method}')
                fits.append(dict(run=cp['run'], epoch=int(cp['epoch']), mode=mode, method=method, **fit))
                pair[mode] = fit
            a, m = pair['shape'], pair['mean']
            coordinates.append(dict(run=cp['run'], epoch=int(cp['epoch']), method=method,
                kappa_a=a['kappa'], kappa_m=m['kappa'], epsilon_a=a['amplitude'], epsilon_m=m['amplitude'],
                amplitude_ratio=m['amplitude']/a['amplitude'], horizon=cp['horizon'],
                shape_fit_relative_rmse=a['magnitude_relative_rmse'], mean_fit_relative_rmse=m['magnitude_relative_rmse'],
                shape_fit_at_bound=False, mean_fit_at_bound=False,
                mixed_sign_a=a['mixed_sign'], mixed_sign_m=m['mixed_sign'],
                in_original_axes=bool(0 <= a['kappa'] <= 2 and -.5 <= m['kappa'] <= 2),
                infinite_horizon_valid=bool(a['kappa'] > 0 and m['kappa'] > -.5)))
    write_csv(out/'mode_fits.csv', fits)
    write_csv(out/'checkpoint_coordinates.csv', coordinates)
    statistics = compare_measurements(paper, coordinates, out)
    summary = {}
    for method in VARIANTS:
        group = [r for r in coordinates if r['method'] == method]
        summary[method] = dict(checkpoints=len(group), inside_original_axes=sum(r['in_original_axes'] for r in group),
                               infinite_horizon_valid=sum(r['infinite_horizon_valid'] for r in group))
        for mode in ('shape', 'mean'):
            selected = [r for r in fits if r['mode'] == mode and r['method'] == method]
            values = [r['kappa'] for r in selected]
            summary[method][mode] = dict(min_kappa=min(values), median_kappa=float(np.median(values)), max_kappa=max(values),
                median_log_r2=float(np.median([r['log_r2'] for r in selected])), mixed_sign=sum(r['mixed_sign'] for r in selected),
                min_retained_clock_fraction=min(r['retained_clock_fraction'] for r in selected))
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return checkpoints, coordinates, fits, statistics, summary, sources


def figures(paper, out, checkpoints, coordinates, fits, statistics):
    with np.load(paper/'outputs/score_error_modes/clock.npz') as clock:
        d = clock['distance']
    examples = [max(checkpoints, key=lambda r: r['kappa_a']), max(checkpoints, key=lambda r: r['kappa_m']),
                next(r for r in checkpoints if r['run'] == 'default4' and r['epoch'] == 24)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for j, cp in enumerate(examples):
        with np.load(paper/'outputs/score_error_modes/profiles'/f"{cp['run']}_epoch{int(cp['epoch']):02d}.npz") as data:
            for i, (mode, key, suffix) in enumerate([('shape', 'a_linear', 'a'), ('mean', 'u_linear', 'm')]):
                ax, y = axes[i, j], data[key]
                ax.plot(np.exp(d), abs(y), c='.7', lw=.8)
                for mask, color, label in [(y > 0, 'C0', 'positive profile'), (y < 0, 'C1', 'negative profile')]:
                    ax.scatter(np.exp(d[mask]), abs(y[mask]), c=color, s=12, label=label)
                for method, style in [('loglog', '-'), ('loglog_unweighted', '--')]:
                    r = next(r for r in fits if r['run'] == cp['run'] and r['epoch'] == cp['epoch']
                             and r['mode'] == mode and r['method'] == method)
                    ax.plot(np.exp(d), r['amplitude']*np.exp(-r['kappa']*d), style,
                            label=f"{method}: k={r['kappa']:.3g}; log R²={r['log_r2']:.2f}")
                ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=.2)
                ax.set(xlabel='Variance ratio V / V_ref', ylabel=f'Absolute {mode} profile',
                       title=f"{cp['run']}, epoch {int(cp['epoch'])}: {mode}; original k={cp[f'kappa_{suffix}']:.3g}")
                ax.legend(fontsize=8)
    fig.suptitle('Global least-squares slopes on log–log axes; kappa = minus slope\n'
                 'Blue/orange points retain the original signs; straight lines fit magnitude envelopes.')
    save_figure(fig, out, 'loglog_profile_examples')

    methods = ['original', *VARIANTS]
    labels = ['Original', 'Log-log\ncut 1e-6', 'Log-log\ncut 1e-8', 'Log-log\ncut 1e-4', 'Log-log\nunweighted']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for i, direction in enumerate(('q_p', 'p_q')):
        for j, gamma in enumerate((.2, 1., 5.)):
            ax = axes[i, j]
            for dataset, label in [('controlled_full', 'full score'), ('controlled_affine', 'affine only')]:
                vals = [next(r['spearman'] for r in statistics if r['dataset'] == dataset and r['gamma'] == gamma
                    and r['kl_direction'] == direction and r['method'] == method and r['run'] == 'all' and r['subset'] == 'all') for method in methods]
                ax.plot(range(len(methods)), vals, 'o-', label=label)
            ax.set_xticks(range(len(methods)), labels, rotation=20, ha='right')
            ax.set(title=f'gamma={gamma:g}; {direction}', ylabel='Spearman with measured log10 KL ratio', ylim=(-.5, 1))
            ax.axhline(0, c='.6', lw=.8); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('Log-log fit predictions: magnitude envelopes lose temporal sign cancellation\n'
                 'Finite horizon; controlled sampler with measured ODE baseline; 64 bins; all 144 checkpoints')
    save_figure(fig, out, 'loglog_correlations')
    for method in ('loglog', 'loglog_unweighted'):
        destination = out/method; destination.mkdir(exist_ok=True)
        phase_overlays([r for r in coordinates if r['method'] == method], destination, paper)


def validate_clock_weights(paper):
    """Reproduce weighted OLS with independent weighted design-matrix solves."""
    from .score_error_modes import integration_weights
    d = np.load(paper/'outputs/score_error_modes/clock.npz')['distance']
    w = integration_weights(d)
    checks = []
    for path in sorted((paper/'outputs/score_error_modes/profiles').glob('*.npz')):
        with np.load(path) as data:
            for key in ('a_linear', 'u_linear'):
                y = data[key]; mask = abs(y) > 1e-6*max(abs(y))
                X = np.column_stack([np.ones(mask.sum()), d[mask]])
                independent = np.linalg.lstsq(np.sqrt(w[mask, None])*X, np.sqrt(w[mask])*np.log(abs(y[mask])), rcond=None)[0]
                fit = fit_loglog(d, y)
                error = float(np.max(abs(independent-[fit['log_amplitude'], fit['slope']])))
                checks.append(dict(checkpoint=path.stem, mode=key, coefficient_error=error))
    return dict(passed=all(r['coefficient_error'] < 1e-10 for r in checks), checks=len(checks),
                max_coefficient_error=max(r['coefficient_error'] for r in checks))


def run(paper=PAPER, output_dir=None):
    paper = Path(paper); out = Path(output_dir) if output_dir else paper/'outputs/loglog_kappa'
    out.mkdir(parents=True, exist_ok=True)
    checkpoints, coordinates, fits, statistics, summary, sources = analyze(paper, out)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    figures(paper, out, checkpoints, coordinates, fits, statistics)
    validation = validate_clock_weights(paper)
    (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    sources += [paper/p for p in ('outputs/score_error_modes/clock.npz', 'outputs/score_error_modes/checkpoint_summary.csv',
        'outputs/score_error_modes/kl_comparison.csv', 'outputs/residual_ablation/ablation_results.csv',
        'score_error_analysis/loglog_kappa.py', 'score_error_analysis/analyze_loglog_kappa.py',
        'score_error_analysis/analyze_kappa_sensitivity.py', 'score_error_analysis/score_error_modes.py',
        'score_error_analysis/compare_exponential_predictions.py', 'score_error_analysis/plot_checkpoint_modes.py',
        'plot_phase_diagrams.py', 'outputs/phase_diagrams/metadata.json')]
    metadata = dict(complete=validation['passed'], checkpoints=len(checkpoints),
        variants={k: dict(weighting=w, relative_cutoff=c) for k, (w, c) in VARIANTS.items()},
        fit='log(abs(profile)) = log(amplitude) - kappa*log(V/V_ref); amplitude is positive',
        signs='Both signs retained by fitting magnitude; sign changes recorded. Signed kernel cancellation is lost.',
        prediction='Finite-horizon envelope prediction; each checkpoint own amplitude ratio; no KL measurements used in fitting.',
        phase_flags='Crosses use relative magnitude-profile RMSE > 0.25; inspect mixed-sign flags separately.',
        source_sha256={str(p.relative_to(paper)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    for r in statistics:
        if r['dataset'] == 'controlled_full' and r['run'] == 'all' and r['subset'] == 'all' and r['method'] == 'loglog':
            print(r['gamma'], r['kl_direction'], 'Spearman', r['spearman'], 'median gap', r['median_absolute_log10_gap'], flush=True)
    if not validation['passed']:
        raise RuntimeError('Independent weighted-OLS verification failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path, default=PAPER)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.paper_dir, args.output_dir)

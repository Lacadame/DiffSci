"""Compare fitted-exponential and signed-profile KL ratios to saved measurements.

No checkpoint inference or sampling is repeated. Run as a module from repo root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .plot_checkpoint_modes import (
    load_summary, measured_markers, phase_measurements, phase_overlays,
    plt, Normalize, save_figure,
)
from .score_error_modes import exponential_response


PAPER = Path(__file__).resolve().parents[1]
MODELS = ('profile', 'exponential_finite', 'exponential_infinite')
LABELS = ('Signed profiles', 'Exponential, finite horizon', 'Exponential, infinite horizon')
DATASETS = ('saved_sweep', 'controlled_full', 'controlled_affine')


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exponential_log_ratio(checkpoint, gamma, baseline, *, infinite=False):
    """Evaluate each checkpoint's own amplitudes, never its display-panel ratio."""
    values = exponential_response(
        checkpoint['kappa_a'], checkpoint['kappa_m'],
        checkpoint['epsilon_a'], checkpoint['epsilon_m'],
        [baseline, gamma], np.inf if infinite else checkpoint['horizon'],
    )
    if not np.isfinite(values).all() or np.any(values <= 0):
        return np.nan
    return float(np.log10(values[1])-np.log10(values[0]))


def correlation(x, y, method):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(method(x, y).statistic)


def agreement_metrics(prediction, measured, runs):
    """Descriptive association, calibration, and within-run centered association."""
    x, y, runs = np.asarray(prediction, float), np.asarray(measured, float), np.asarray(runs)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, runs = x[valid], y[valid], runs[valid]
    centered_x, centered_y = x.copy(), y.copy()
    for run in np.unique(runs):
        mask = runs == run
        centered_x[mask] -= x[mask].mean()
        centered_y[mask] -= y[mask].mean()
    # Spearman is identical on ratios and log ratios. Report both Pearson scales.
    with np.errstate(over='ignore', under='ignore'):
        ratios_x, ratios_y = 10.**x, 10.**y
    finite_ratios = np.isfinite(ratios_x) & np.isfinite(ratios_y) & (ratios_x > 0) & (ratios_y > 0)
    return dict(
        n=int(valid.sum()), excluded_nonfinite=int((~valid).sum()),
        spearman=correlation(x, y, spearmanr), pearson_log10=correlation(x, y, pearsonr),
        pearson_ratio=correlation(ratios_x[finite_ratios], ratios_y[finite_ratios], pearsonr),
        n_finite_ratios=int(finite_ratios.sum()),
        pearson_log10_within_run_centered=correlation(centered_x, centered_y, pearsonr),
        sign_agreement=float(np.mean(np.sign(x) == np.sign(y))) if len(x) else np.nan,
        median_absolute_log10_gap=float(np.median(abs(x-y))) if len(x) else np.nan,
        rmse_log10=float(np.sqrt(np.mean((x-y)**2))) if len(x) else np.nan,
    )


def build_comparisons(paper_dir, targets=(.2, 1., 5.)):
    modes, ablation = paper_dir/'outputs/score_error_modes', paper_dir/'outputs/residual_ablation'
    checkpoints = load_summary(modes)
    lookup = {(r['run'], int(r['epoch'])): r for r in checkpoints}
    if len(lookup) != len(checkpoints):
        raise ValueError('Duplicate checkpoint IDs')
    mode_meta = json.loads((modes/'metadata.json').read_text())
    ablation_meta = json.loads((ablation/'metadata.json').read_text())
    if not mode_meta['complete'] or not ablation_meta['complete']:
        raise ValueError('Complete the profile analysis and ablation first')
    if mode_meta['fingerprint'] != ablation_meta['profile_fingerprint']:
        raise ValueError('Profile analysis and ablation have different provenance')
    if ablation_meta['prior'] != 'exact':
        raise ValueError('This comparison assumes the controlled exact sampling prior')
    for name in ('sigma_min', 'sigma_max', 'mixture'):
        if mode_meta[name] != ablation_meta[name]:
            raise ValueError(f'Incompatible profile and ablation {name}')
    rows = []

    def append(record, dataset, gamma, target, baseline, bins, direction, measured, profile, floor):
        key = record['run'], int(record['epoch'])
        cp = lookup[key]
        rows.append(dict(
            dataset=dataset, run=key[0], epoch=key[1], target_gamma=target, gamma=gamma,
            baseline_gamma=baseline, bins=bins, kl_direction=direction,
            measured_log10_ratio=measured, profile_log10_ratio=profile,
            exponential_finite_log10_ratio=exponential_log_ratio(cp, gamma, baseline),
            exponential_infinite_log10_ratio=exponential_log_ratio(cp, gamma, baseline, infinite=True),
            above_control_floor=floor,
            max_fit_relative_rmse=max(cp['shape_fit_relative_rmse'], cp['mean_fit_relative_rmse']),
            poor_or_boundary_fit=(max(cp['shape_fit_relative_rmse'], cp['mean_fit_relative_rmse']) > .25
                                  or cp['shape_fit_at_bound'] or cp['mean_fit_at_bound']),
        ))

    groups = defaultdict(list)
    for r in read_csv(modes/'kl_comparison.csv'):
        groups[r['run'], int(r['epoch'])].append(r)
    if set(groups) != set(lookup):
        raise ValueError('Saved sweep is missing checkpoint measurements')
    for group in groups.values():
        for target in targets:
            r = min(group, key=lambda r: abs(float(r['gamma'])-target))
            gamma, baseline = float(r['gamma']), float(r['empirical_baseline_gamma'])
            if gamma == baseline:
                raise ValueError('Do not correlate the identically zero baseline ratio')
            for direction in ('q_p', 'p_q'):
                append(r, 'saved_sweep', gamma, target, baseline, 0, direction,
                       float(r[f'empirical_log10_ratio_{direction}']),
                       float(r['profile_log10_ratio_recorded_baseline']), '')
                # Independently reconstructed predictions must reproduce the saved ones.
                for model in MODELS[1:]:
                    np.testing.assert_allclose(rows[-1][f'{model}_log10_ratio'],
                        float(r[f'{model}_log10_ratio_recorded_baseline']), atol=1e-12, equal_nan=True)

    for r in read_csv(ablation/'ablation_results.csv'):
        gamma = float(r['gamma'])
        if gamma not in targets:
            continue
        lam = float(r['residual_lambda'])
        if lam not in (0., 1.):
            continue
        for direction in ('q_p', 'p_q'):
            append(r, 'controlled_full' if lam == 1 else 'controlled_affine', gamma, gamma, 0.,
                   int(r['bins']), direction, float(r[f'log10_ratio_ode_{direction}']),
                   float(r[f'gaussian_profile_log10_ratio_ode_{direction}']),
                   r[f'above_control_floor_{direction}'] == 'True')
    # Fail on incomplete joins instead of silently correlating different checkpoint sets.
    coverage = defaultdict(list)
    for r in rows:
        coverage[r['dataset'], r['bins'], r['target_gamma'], r['kl_direction']].append((r['run'], r['epoch']))
    for dataset in DATASETS:
        for bins in ([0] if dataset == 'saved_sweep' else ablation_meta['bins']):
            for target in targets:
                for direction in ('q_p', 'p_q'):
                    ids = coverage[dataset, bins, target, direction]
                    if len(ids) != len(lookup) or set(ids) != set(lookup):
                        raise ValueError(f'Incomplete or duplicate comparison: {dataset}, {bins}, {target}, {direction}')
    return rows, checkpoints


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['dataset'], r['bins'], r['gamma'], r['target_gamma'], r['kl_direction']].append(r)
    results = []
    for (dataset, bins, gamma, target, direction), group in sorted(groups.items()):
        subsets = dict(all=group, common_valid=[r for r in group if all(
            np.isfinite(r[f'{model}_log10_ratio']) for model in MODELS)])
        if dataset != 'saved_sweep':
            subsets['above_control_floor'] = [r for r in group if r['above_control_floor']]
        for subset, selected in subsets.items():
            for run in ['all', *sorted(set(r['run'] for r in group))]:
                selected_run = [r for r in selected if run == 'all' or r['run'] == run]
                for model in MODELS:
                    metrics = agreement_metrics([r[f'{model}_log10_ratio'] for r in selected_run],
                                                [r['measured_log10_ratio'] for r in selected_run],
                                                [r['run'] for r in selected_run])
                    results.append(dict(dataset=dataset, bins=bins, gamma=gamma, target_gamma=target,
                                        baseline_gamma=group[0]['baseline_gamma'], kl_direction=direction,
                                        subset=subset, run=run, model=model, selected_count=len(selected_run), **metrics))
    return results


def comparison_figures(rows, out, primary_bins):
    for dataset in DATASETS:
        for direction in ('q_p', 'p_q'):
            selected = [r for r in rows if r['dataset'] == dataset and r['kl_direction'] == direction
                        and r['bins'] == (0 if dataset == 'saved_sweep' else primary_bins)]
            targets = sorted(set(r['target_gamma'] for r in selected))
            fig, axes = plt.subplots(3, len(targets), figsize=(13, 11), squeeze=False, layout='constrained')
            for j, target in enumerate(targets):
                group = [r for r in selected if r['target_gamma'] == target]
                for i, (model, label) in enumerate(zip(MODELS, LABELS)):
                    ax = axes[i, j]
                    x = np.array([r[f'{model}_log10_ratio'] for r in group])
                    y = np.array([r['measured_log10_ratio'] for r in group])
                    valid = np.isfinite(x) & np.isfinite(y)
                    for k, run in enumerate(sorted(set(r['run'] for r in group))):
                        mask = np.array([r['run'] == run for r in group]) & valid
                        ax.scatter(x[mask], y[mask], s=26, c=f'C{k}', alpha=.8, label=run)
                    if dataset != 'saved_sweep':
                        floor = np.array([not r['above_control_floor'] for r in group]) & valid
                        ax.scatter(x[floor], y[floor], s=6, c='white', zorder=4)
                    if valid.any():
                        low, high = min(x[valid].min(), y[valid].min()), max(x[valid].max(), y[valid].max())
                        pad = max(.03, .06*(high-low))
                        ax.plot([low-pad, high+pad], [low-pad, high+pad], '--', c='.5', lw=.8)
                        ax.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad), aspect='equal')
                    metrics = agreement_metrics(x, y, [r['run'] for r in group])
                    ax.axhline(0, c='.7', lw=.6); ax.axvline(0, c='.7', lw=.6)
                    ax.set_title(f"{label}; gamma={group[0]['gamma']:.4g}\n"
                                 f"Spearman={metrics['spearman']:+.3f}, Pearson={metrics['pearson_log10']:+.3f}, n={metrics['n']}", fontsize=10)
                    ax.set(xlabel='Predicted log10 KL ratio', ylabel='Measured log10 KL ratio')
                    if i == j == 0:
                        ax.legend(fontsize=8)
            baseline = selected[0]['baseline_gamma']
            description = {'saved_sweep': 'Original saved sweep', 'controlled_full': 'Controlled sampler: full learned score',
                           'controlled_affine': 'Controlled sampler: affine error only'}[dataset]
            kl = 'KL(q || p)' if direction == 'q_p' else 'KL(p || q)'
            note = '' if dataset == 'saved_sweep' else f'; {primary_bins} bins; white centers: near control floor'
            fig.suptitle(f'{description}; {kl}; baseline gamma={baseline:g}{note}\n'
                         'Signed least-squares exponential fits; all fit qualities included; three correlated training trajectories.', fontsize=11)
            save_figure(fig, out, f'{dataset}_{direction}')


def measured_coordinate_figures(checkpoints, paper_dir, out, primary_bins):
    measurements = phase_measurements(paper_dir, primary_bins)
    ka, km = (np.array([r[k] for r in checkpoints]) for k in ('kappa_a', 'kappa_m'))
    poor = np.array([max(r['shape_fit_relative_rmse'], r['mean_fit_relative_rmse']) > .25
                     or r['shape_fit_at_bound'] or r['mean_fit_at_bound'] for r in checkpoints])
    norm = Normalize(-2, 2)
    for direction in ('q_p', 'p_q'):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
        for ax, gamma in zip(axes, (.2, 1., 5.)):
            values = [measurements[r['run'], int(r['epoch']), gamma][direction] for r in checkpoints]
            measured_markers(ax, ka, km, values, poor, norm)
            ax.set_xscale('symlog', linthresh=1); ax.set_yscale('symlog', linthresh=1)
            ax.axvline(1, ls='--', c='.6', lw=.8); ax.axhline(0, ls='--', c='.6', lw=.8)
            ax.set(xlabel=r'Shape localization $\kappa_a$', ylabel=r'Mean localization $\kappa_m$', title=f'gamma={gamma:g}')
            ax.grid(alpha=.15)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        fig.colorbar(scalar, ax=axes, label='Measured log10(KL_gamma / KL_ODE)', fraction=.02, extend='both')
        kl = 'KL(q || p)' if direction == 'q_p' else 'KL(p || q)'
        fig.suptitle(f'All {len(checkpoints)} checkpoints: measured {kl}, full score, {primary_bins} bins\n'
                     'Symmetric-log axes; crosses: poor/boundary fits; no theory background outside the original panels.', fontsize=12)
        save_figure(fig, out, f'phase_coordinates_measured_{direction}')


def run(paper_dir=PAPER, output_dir=None, primary_bins=64):
    paper_dir = Path(paper_dir)
    out = Path(output_dir) if output_dir else paper_dir/'outputs/exponential_kl_comparison'
    out.mkdir(parents=True, exist_ok=True)
    rows, checkpoints = build_comparisons(paper_dir)
    if primary_bins not in {r['bins'] for r in rows if r['dataset'] == 'controlled_full'}:
        raise ValueError(f'No controlled measurements with {primary_bins} bins')
    statistics = summarize(rows)
    write_csv(out/'predictions_vs_measurements.csv', rows)
    write_csv(out/'correlations.csv', statistics)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    comparison_figures(rows, out, primary_bins)
    measured_coordinate_figures(checkpoints, paper_dir, out, primary_bins)
    for direction in ('q_p', 'p_q'):
        phase_overlays(checkpoints, paper_dir/'outputs/score_error_modes', paper_dir,
                       direction=direction, primary_bins=primary_bins)
    inputs = ['outputs/score_error_modes/checkpoint_summary.csv', 'outputs/score_error_modes/kl_comparison.csv',
              'outputs/score_error_modes/metadata.json', 'outputs/residual_ablation/ablation_results.csv',
              'outputs/residual_ablation/metadata.json', 'outputs/phase_diagrams/metadata.json', 'plot_phase_diagrams.py',
              'score_error_analysis/compare_exponential_predictions.py',
              'score_error_analysis/plot_checkpoint_modes.py', 'score_error_analysis/score_error_modes.py']
    metadata = dict(complete=True, checkpoints=len(checkpoints), comparison_rows=len(rows),
                    statistic_rows=len(statistics), primary_bins=primary_bins, targets=[.2, 1., 5.],
                    inputs_sha256={p: hashlib.sha256((paper_dir/p).read_bytes()).hexdigest() for p in inputs},
                    baselines={'saved_sweep': 'actual recorded gamma=0.01', 'controlled': 'measured gamma=0 ODE'},
                    correlation='Spearman on ratios/log ratios; Pearson on both scales; per gamma and KL direction',
                    subsets=['all', 'common_valid across the three predictions', 'above_control_floor for controlled measurements'],
                    limitation='144 checkpoints from three correlated training runs; descriptive statistics, no independent-checkpoint p-values',
                    phase_colors='Full learned score, primary bin count, measured ODE baseline; same RdBu_r [-2,2] scale as theory backgrounds')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {len(rows)} comparisons and {len(statistics)} correlation summaries in {out}', flush=True)
    for r in statistics:
        if r['dataset'] == 'controlled_full' and r['bins'] == primary_bins and r['run'] == 'all' and r['subset'] == 'all':
            print(f"gamma={r['gamma']:g} {r['kl_direction']} {r['model']}: "
                  f"Spearman={r['spearman']:+.4f}, Pearson(log10)={r['pearson_log10']:+.4f}, "
                  f"median gap={r['median_absolute_log10_gap']:.4f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path, default=PAPER)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--primary-bins', type=int, default=64)
    args = parser.parse_args()
    run(args.paper_dir, args.output_dir, args.primary_bins)

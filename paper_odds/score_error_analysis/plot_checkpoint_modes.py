"""Plots of checkpoint coefficients, phase coordinates, and residual diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'diffsci-mpl'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.stats import spearmanr

from .score_error_modes import exponential_response


def load_summary(output_dir):
    """Read the per-checkpoint table without requiring pandas."""
    with (Path(output_dir)/'checkpoint_summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    text_keys = {'run', 'checkpoint'}
    for row in rows:
        for key, value in row.items():
            if key in text_keys:
                continue
            row[key] = value == 'True' if value in ('True', 'False') else float(value)
    return rows


def save_figure(fig, out, name):
    fig.savefig(out/f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def phase_coordinates(rows, out):
    ka, km, rho, residual = [np.array([r[k] for r in rows]) for k in
                            ('kappa_a', 'kappa_m', 'amplitude_ratio', 'nonlinear_energy_fraction')]
    finite = np.isfinite(ka) & np.isfinite(km)
    poor = np.array([r['shape_fit_relative_rmse'] > .25 or r['mean_fit_relative_rmse'] > .25
                     or r['shape_fit_at_bound'] or r['mean_fit_at_bound'] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    plot_values = [np.log10(np.maximum(rho, 1e-12)), residual,
                   np.maximum([r['shape_fit_relative_rmse'] for r in rows],
                              [r['mean_fit_relative_rmse'] for r in rows])]
    labels = [r'$\log_{10}|\epsilon_m/\epsilon_a|$', 'Nonlinear fraction of integrated score-error energy',
              'Larger of the two relative fit RMSEs']
    for ax, values, label in zip(axes, plot_values, labels):
        artist = ax.scatter(ka[finite], km[finite], c=np.asarray(values)[finite], cmap='viridis', s=35,
                            edgecolors='0.25', linewidths=.3, alpha=.85)
        ax.scatter(ka[finite & poor], km[finite & poor], marker='x', color='black', s=12, linewidths=.4)
        ax.axvline(1, ls='--', c='.5', lw=1)
        ax.axhline(0, ls='--', c='.5', lw=1)
        ax.set(xlabel=r'Shape localization $\kappa_a$', ylabel=r'Mean localization $\kappa_m$')
        ax.set_xscale('symlog', linthresh=1)
        ax.set_yscale('symlog', linthresh=1)
        ax.set_xlim(min(-.05, 1.2*ka[finite].min()), max(2., 1.2*ka[finite].max()))
        ax.set_ylim(min(-.5, 1.2*km[finite].min()), max(2., 1.2*km[finite].max()))
        fig.colorbar(artist, ax=ax, label=label, orientation='horizontal', pad=.16, fraction=.07)
    fig.suptitle(f'{len(rows)} checkpoints: Gaussian surrogate coordinates\n'
                 f'Crosses: relative fit RMSE > 0.25 or exponent at search bound ({poor.sum()}/{len(rows)}); symmetric-log axes', fontsize=13)
    save_figure(fig, out, 'checkpoint_phase_coordinates')


def phase_measurements(paper_dir, primary_bins=64):
    """Full-score measurements at exact gamma, with the measured ODE baseline."""
    path = Path(paper_dir)/'outputs/residual_ablation/ablation_results.csv'
    if not path.exists():
        return None
    result = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if int(row['bins']) != primary_bins or float(row['residual_lambda']) != 1:
                continue
            key = row['run'], int(row['epoch']), float(row['gamma'])
            if key in result:
                raise ValueError(f'Duplicate phase measurement: {key}')
            result[key] = {d: float(row[f'log10_ratio_ode_{d}']) for d in ('q_p', 'p_q')}
    if not result:
        raise ValueError(f'No full-score measurements with {primary_bins} bins in {path}')
    return result


def measured_markers(ax, x, y, values, poor, norm):
    """Outlined markers remain visible even when their color matches the panel."""
    x, y, values, poor = map(np.asarray, (x, y, values, poor))
    for marker, subset in [('o', ~poor), ('x', poor)]:
        if not subset.any():
            continue
        for color, width in [('black', 4.4), ('white', 3.4)]:
            ax.scatter(x[subset], y[subset], marker=marker, color=color, s=70,
                       linewidths=width, zorder=6)
        ax.scatter(x[subset], y[subset], marker=marker, c=values[subset],
                   cmap='RdBu_r', norm=norm, s=70, linewidths=2.6, zorder=7)


def phase_overlays(rows, out, paper_dir, *, direction='q_p', primary_bins=64):
    """Reuse the actual phase plotting formulas; bin rho only for panel display."""
    spec = importlib.util.spec_from_file_location('existing_phase_diagrams', paper_dir/'plot_phase_diagrams.py')
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)
    meta = json.loads((paper_dir/'outputs/phase_diagrams/metadata.json').read_text())
    gammas, ratios = np.array(meta['gammas']), np.array(meta['ratios'])
    if np.any(np.diff(ratios) <= 0) or np.any(ratios <= 0):
        raise ValueError('Existing ratio panels must be increasing and positive')
    ka_grid = np.linspace(*meta['kappa_a_range'], 241)
    km_grid = np.linspace(*meta['kappa_m_range'], 241)
    ka, km, rho = [np.array([r[k] for r in rows]) for k in ('kappa_a', 'kappa_m', 'amplitude_ratio')]
    bins = np.searchsorted(np.sqrt(ratios[:-1]*ratios[1:]), rho)
    in_axes = np.isfinite(ka) & np.isfinite(km) & (ka >= ka_grid[0]) & (ka <= ka_grid[-1]) & (km >= km_grid[0]) & (km <= km_grid[-1])
    poor = np.array([max(r['shape_fit_relative_rmse'], r['mean_fit_relative_rmse']) > .25
                     or r['shape_fit_at_bound'] or r['mean_fit_at_bound'] for r in rows])
    norm = Normalize(-2, 2)
    measurements = phase_measurements(paper_dir, primary_bins)
    fig, axes = plt.subplots(len(gammas), len(ratios), figsize=(19, 10), squeeze=False, layout='constrained')
    for i, gamma in enumerate(gammas):
        measured = None if measurements is None else np.array([
            measurements[(r['run'], int(r['epoch']), float(gamma))][direction] for r in rows
        ])
        if measured is not None and not np.isfinite(measured).all():
            raise ValueError('Phase marker colors require finite measured log ratios')
        for j, ratio in enumerate(ratios):
            ax = axes[i, j]
            values = phase.log_kl_ratio(ka_grid[None, :], km_grid[:, None], gamma, ratio)
            mesh, _ = phase.draw_panel(ax, ka_grid, km_grid, values, gamma, ratio, norm, compact=True)
            mask = (bins == j) & in_axes
            if measured is None:
                for marker, subset in [('o', mask & ~poor), ('x', mask & poor)]:
                    ax.scatter(ka[subset], km[subset], marker=marker, color='black', s=17,
                               linewidths=.7, zorder=6)
            else:
                measured_markers(ax, ka[mask], km[mask], measured[mask], poor[mask], norm)
            ax.text(.98, .02, f'{mask.sum()} shown', transform=ax.transAxes, ha='right', fontsize=8,
                    bbox={'facecolor':'white', 'alpha':.8, 'edgecolor':'none'})
    label = phase.colorbar_label() if measurements is None else r'$\log_{10}(\mathrm{KL}_{\gamma}/\mathrm{KL}_{0})$: theory background / measured markers'
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label=label, fraction=.018, pad=.01, extend='both')
    direction_label = 'KL(q || p)' if direction == 'q_p' else 'KL(p || q)'
    color_note = ('Crosses denote poor/boundary fits.' if measurements is None else
                  f'Marker colors: measured {direction_label}, full score, {primary_bins} bins, true ODE baseline. Crosses: poor/boundary fits.')
    fig.suptitle(f'Checkpoints on the existing infinite-horizon diagrams ({in_axes.sum()}/{len(rows)} within axes)\n'
                 'Each checkpoint assigned to its nearest ratio panel in log space; background uses panel ratio.\n'
                 + color_note, fontsize=12)
    save_figure(fig, out, 'checkpoint_phase_overlays' + ('' if direction == 'q_p' else '_p_q'))
    return int((~in_axes).sum())


def example_profiles(rows, out):
    indices = np.random.default_rng(42).choice(len(rows), min(5, len(rows)), replace=False)
    clock = np.load(out/'clock.npz')
    d, sigma = clock['distance'], clock['sigma']
    fig, axes = plt.subplots(4, len(indices), figsize=(4*len(indices), 12), squeeze=False, layout='constrained')
    for j, index in enumerate(indices):
        r = rows[index]
        path = out/'profiles'/f"{r['run']}_epoch{int(r['epoch']):02d}.npz"
        with np.load(path) as p:
            axes[0, j].plot(sigma, p['b'], label=r'$b(t)$')
            axes[1, j].plot(sigma, p['C'], label=r'$C(t)$', color='tab:orange')
            for key, amplitude, exponent, color, label in (
                ('a_linear', r['epsilon_a'], r['kappa_a'], 'tab:orange', r'$a_{lin}=VC$'),
                ('u_linear', r['epsilon_m'], r['kappa_m'], 'tab:blue', r'$u_{lin}=Vb/\sqrt{V_{ref}}$')):
                axes[2, j].plot(d, p[key], color=color, label=label)
                axes[2, j].plot(d, amplitude*np.exp(-exponent*d), '--', color=color)
            for key, label, color in [('mean_energy', 'mean', 'tab:blue'), ('affine_energy', 'linear', 'tab:orange'),
                                       ('residual_energy', 'nonlinear', 'tab:green')]:
                axes[3, j].plot(d, p[key]/np.maximum(p['total_energy'], 1e-30), label=label, color=color)
        axes[0, j].set_title(f"{r['run']}, epoch {int(r['epoch'])}")
        for i in (0, 1):
            axes[i, j].set_xscale('log')
            axes[i, j].set_yscale('symlog', linthresh=.01)
            axes[i, j].set_xlabel(r'EDM noise $t=\sigma$')
        axes[2, j].set_yscale('symlog', linthresh=.01)
        axes[2, j].set_xlabel(r'Log-variance distance $d=\Lambda-\ell$')
        axes[3, j].set(xlabel=r'Log-variance distance $d$', ylim=(-.02, 1.02))
        for ax in axes[:, j]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    for i, ylabel in enumerate(('Signed intercept', 'Signed slope', 'Normalized profile; dashed = fit', 'Fraction of score-error energy')):
        axes[i, 0].set_ylabel(ylabel)
    save_figure(fig, out, 'example_mode_profiles')


def empirical_comparison(rows, out):
    with (out/'kl_comparison.csv').open() as f:
        comparisons = list(csv.DictReader(f))
    residual = {(r['run'], int(r['epoch'])): r['nonlinear_energy_fraction'] for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    residual_fig, residual_axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    summary = {}
    selected_gammas = []
    by_checkpoint = {}
    for r in comparisons:
        key = r['run'], int(r['epoch'])
        by_checkpoint.setdefault(key, []).append(r)
    for col, target_gamma in enumerate((.2, 1., 5.)):
        chosen = [min(group, key=lambda r: abs(float(r['gamma'])-target_gamma)) for group in by_checkpoint.values()]
        actual_gamma = sorted(set(float(r['gamma']) for r in chosen))
        selected_gammas.append(actual_gamma)
        for i, direction in enumerate(('q_p', 'p_q')):
            ax = axes[i, col]
            empirical = np.array([float(r[f'empirical_log10_ratio_{direction}']) for r in chosen])
            prediction = np.array([float(r['profile_log10_ratio_recorded_baseline']) for r in chosen])
            color = [residual[(r['run'], int(r['epoch']))] for r in chosen]
            good = np.isfinite(empirical) & np.isfinite(prediction)
            scatter = ax.scatter(prediction[good], empirical[good], c=np.asarray(color)[good], vmin=0, vmax=1,
                                 s=24, cmap='viridis', edgecolors='.2', linewidths=.2)
            ax.axhline(0, c='.5', ls='--', lw=.8)
            ax.axvline(0, c='.5', ls='--', lw=.8)
            bounds = [min(prediction[good].min(), empirical[good].min()), max(prediction[good].max(), empirical[good].max())]
            ax.plot(bounds, bounds, ':', c='.5', lw=.8)
            agreement = float(np.mean((prediction[good] < 0) == (empirical[good] < 0)))
            corr = float(spearmanr(prediction[good], empirical[good]).statistic) if good.sum() > 2 and np.ptp(prediction[good]) > 0 and np.ptp(empirical[good]) > 0 else np.nan
            mismatch = np.abs(prediction-empirical)
            residual_corr = float(spearmanr(np.asarray(color)[good], mismatch[good]).statistic) if good.sum() > 2 and np.ptp(mismatch[good]) > 0 else np.nan
            summary[f'{target_gamma:g}_{direction}'] = dict(actual_gammas=actual_gamma, count=int(good.sum()),
                                                          sign_agreement=agreement, spearman=corr,
                                                          residual_vs_absolute_disagreement_spearman=residual_corr)
            title_gamma = f'{actual_gamma[0]:.3g}' if len(actual_gamma) == 1 else f'nearest {target_gamma:g}'
            ax.set_title(rf'$\gamma={title_gamma}$; sign agreement {agreement:.0%}')
            ax.set_xlabel(r'Gaussian profile prediction: $\log_{10}(H_\gamma/H_{baseline})$')
            ax.set_ylabel(('KL(q || p)' if i == 0 else 'KL(p || q)') + ': observed log10 ratio')
            residual_ax = residual_axes[i, col]
            for run in sorted(set(r['run'] for r in chosen)):
                mask = np.array([r['run'] == run for r in chosen]) & good
                residual_ax.scatter(np.asarray(color)[mask], mismatch[mask], s=22, alpha=.75, label=run)
            residual_ax.set(xlabel='Nonlinear fraction of integrated score-error energy',
                            ylabel='Absolute predicted vs observed log10-ratio difference',
                            title=f'gamma={title_gamma}; {direction}; Spearman={residual_corr:+.2f}')
            residual_ax.legend(fontsize=8)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Nonlinear fraction of integrated score-error energy', fraction=.022)
    baseline = float(comparisons[0]['empirical_baseline_gamma'])
    fig.suptitle(f'Measured signed profiles vs saved sampling results; baseline gamma = {baseline:.3g}\n'
                 'Gaussian surrogate comparison; disagreement also includes nonexponential/large errors, prior and numerical effects.', fontsize=12)
    save_figure(fig, out, 'empirical_vs_gaussian_profiles')
    residual_fig.suptitle('Nonlinear residual vs Gaussian prediction disagreement\n'
                         'Descriptive association, with training runs distinguished; other approximation errors also contribute.', fontsize=12)
    save_figure(residual_fig, out, 'nonlinearity_vs_disagreement')
    return summary


def create_plots(output_dir, paper_dir):
    out, paper_dir = Path(output_dir), Path(paper_dir)
    rows = load_summary(out)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    phase_coordinates(rows, out)
    outside = phase_overlays(rows, out, paper_dir)
    if (paper_dir/'outputs/residual_ablation/ablation_results.csv').exists():
        phase_overlays(rows, out, paper_dir, direction='p_q')
    example_profiles(rows, out)
    agreement = empirical_comparison(rows, out)
    summary = dict(checkpoints=len(rows), outside_existing_axes=outside, baseline_comparisons=agreement)
    for key in ('nonlinear_energy_fraction', 'mean_energy_fraction', 'affine_energy_fraction',
                'shape_fit_relative_rmse', 'mean_fit_relative_rmse', 'max_abs_a_linear', 'max_abs_u_linear',
                'energy_identity_relative_error', 'quadrature_affine_relative_rms_max', 'quadrature_total_relative_error_max'):
        values = np.array([r[key] for r in rows])
        summary[key] = dict(min=float(np.min(values)), median=float(np.median(values)), max=float(np.max(values)))
    summary['poor_exponential_fits'] = sum(max(r['shape_fit_relative_rmse'], r['mean_fit_relative_rmse']) > .25 for r in rows)
    summary['valid_infinite_horizon_coordinates'] = sum(r['infinite_horizon_valid'] for r in rows)
    (out/'diagnostics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Saved phase overlays, coefficient profiles, and comparison figures in {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir', type=Path)
    p.add_argument('--paper-dir', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    create_plots(args.output_dir, args.paper_dir)

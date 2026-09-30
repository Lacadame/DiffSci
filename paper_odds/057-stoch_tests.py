# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-ExecuteTime,-trusted
#     notebook_metadata_filter: all,-widgets,-signature
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.9
# ---

# %% [markdown]
# # Stochastic sampling tests — revised draft (057)
#
# Companion to **The odds for diffusion stochastic sampling (2).pdf**, dated 28 September 2026.
# This repeats the spatial-mode, sampler-agreement and residual-ablation tests of 056 and implements
# Sections 9.2–9.6 of the new draft:
#
# - Fit **$a=VC$ and $u=Vb/\sqrt{V_{\rm ref}}$ directly**, retaining points with $a\geq1$.
# - Refit 1–3-piece power-law RMS envelopes, distinguish ensemble mean from centered SD, and propagate the fitted envelopes through the finite-horizon kernels.
# - Estimate **fixed-$\gamma$ win fractions against the measured ODE**, for both residual arms and both KL directions; report each training run and bin/floor sensitivity.
# - Add sign-confusion matrices, the bounded KL change, and cancellation / $L^1$ diagnostics.
# - Calibrate an OU Gaussian process in $\log\sigma$, evaluate noncentered shape-only orthant odds and both-mode Gaussian quadratic-form odds, and check leave-one-run-out sensitivity.
#
# The original mixture and all trained models are retained. **No retraining or new GPU sampling is required**:
# all empirical experiments use the existing caches. Any future cache regeneration should use **device 6**.
# The 144 checkpoints are repeated epochs from **three training trajectories**, not 144 independent replicates.
# The default envelope/GP cohort is the same 84 checkpoints with epoch ≥ 20 used in the draft;
# all-checkpoint empirical results are also shown. No gamma is selected by minimizing measured KL.
#
# Run all cells with NumPy, SciPy, pandas, Matplotlib and IPython. Figures and CSV tables are also written
# to `outputs/stoch_tests_057/`, with a manifest recording settings and source hashes. The original notebook and caches are unchanged.

# %%
from pathlib import Path
from io import BytesIO
from html import escape
import json
import sys
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from IPython.display import display, HTML, Image

n_steps = 500
random_seed, primary_bins, evaluation_gamma = 42, 64, 1.0
minimum_epoch = 20  # Inclusive cutoff for the three-panel profile overview.
above_floor_only = False  # Applies to the final agreement table.

relative_dir = Path('paper_odds')
root = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / relative_dir / 'outputs/score_error_modes').is_dir())
paper = root / relative_dir

def read_table(path, columns=None):
    return np.atleast_1d(np.genfromtxt(
        path, delimiter=',', names=True, usecols=columns, dtype=None,
        encoding='utf-8')).view(np.recarray)

figure_number = 0
output_dir = paper / 'outputs/stoch_tests_057'
output_dir.mkdir(parents=True, exist_ok=True)

def show_figure(fig, name=None):
    global figure_number
    figure_number += 1
    fig.savefig(output_dir / f'{name or f"figure_{figure_number:02d}"}.png', dpi=160, bbox_inches='tight')
    with BytesIO() as buffer:
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
    plt.close(fig)

def show_table(rows):
    if not rows:
        print('No rows for this selection.')
        return
    def format_value(value):
        return f'{value:.3g}' if isinstance(value, (float, np.floating)) else str(value)
    header = ''.join(f'<th>{escape(key.replace("_", " "))}</th>' for key in rows[0])
    body = ''.join('<tr>' + ''.join(f'<td>{escape(format_value(v))}</td>'
                                  for v in row.values()) + '</tr>' for row in rows)
    display(HTML('<div style="overflow:auto"><table style="border-spacing:12px 4px">'
                 f'<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'))

def agreement(predicted, measured):
    good = np.isfinite(predicted) & np.isfinite(measured)
    x, y = np.asarray(predicted)[good], np.asarray(measured)[good]
    if not len(x):
        return dict(n=0, log_RMSE=np.nan, ratio_RMSE=np.nan, log_bias=np.nan,
                    skill=np.nan, Pearson=np.nan, Spearman=np.nan)
    mse, baseline_mse = np.mean((x-y)**2), np.mean(y**2)
    varying = len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
    return dict(n=len(x), log_RMSE=np.sqrt(mse),
                ratio_RMSE=np.sqrt(np.mean((10.0**x - 10.0**y)**2)),
                log_bias=np.mean(x-y),
                skill=1-mse/baseline_mse if baseline_mse > 0 else np.nan,
                Pearson=np.corrcoef(x, y)[0, 1] if varying else np.nan,
                Spearman=spearmanr(x, y).statistic if varying else np.nan)

summary = read_table(paper / 'outputs/score_error_modes/checkpoint_summary.csv',
                     ['run', 'epoch', 'nonlinear_energy_fraction'])
sweeps = read_table(paper / 'outputs/score_error_modes/kl_comparison.csv', [
    'run', 'epoch', 'gamma', 'empirical_baseline_gamma',
    'empirical_kl_q_p', 'empirical_kl_p_q',
    'empirical_log10_ratio_q_p', 'empirical_log10_ratio_p_q',
    'profile_log10_ratio_recorded_baseline'])
with np.load(paper / 'outputs/score_error_modes/clock.npz') as saved:
    sigma, distance = saved['sigma'].copy(), saved['distance'].copy()
profile_keys = ('b', 'C', 'a_linear', 'u_linear', 'mean_energy',
                'affine_energy', 'residual_energy', 'total_energy')
profiles = {}
for row in summary:
    with np.load(paper / 'outputs/score_error_modes/profiles' / f'{row.run}_epoch{row.epoch:02d}.npz') as saved:
        profiles[(row.run, int(row.epoch))] = {key: saved[key].copy() for key in profile_keys}
ablation_dir = paper / f'outputs/residual_ablation_euler{n_steps}'
ablation = read_table(ablation_dir / 'ablation_results.csv')
paired = read_table(ablation_dir / 'paired_effects.csv')
settings = json.loads((ablation_dir / 'metadata.json').read_text())
assert settings['steps'] == n_steps and settings['sampler_integrator'] == 'euler'
assert settings['moment_integrator'] == 'euler'
assert settings['complete'] and len(profiles) == settings['completed_checkpoints'] == 144
assert primary_bins in settings['bins'] and evaluation_gamma in settings['gammas'][1:]
runs = sorted(set(summary.run))
run_colors = dict(zip(runs, ['tab:blue', 'tab:orange', 'tab:green']))
directions = {'q_p': r'$D_{KL}(q\Vert p)$', 'p_q': r'$D_{KL}(p\Vert q)$'}
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.2})
print(f'{len(profiles)} checkpoints; {len(sigma)} profile noise levels; n = {n_steps} sampling steps.')
print(f'Controlled experiment: {settings["particles"]:,} particles × '
      f'{len(settings["seeds"])} seeds per arm; {primary_bins} KL bins.')
print('Cached integrators: Euler (ODE), Euler–Maruyama (SDE), Euler Gaussian moments; refined moments checked below.')

if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from paper_odds.score_error_analysis.score_error_modes import integration_weights
from paper_odds.score_error_analysis.rms_power_laws import fit_piecewise_power_laws, evaluate_power_law
from paper_odds.score_error_analysis.draft057 import (
    responses, response_weights, bounded_change, bounded_from_log10, two_amplitude_odds,
    shape_gaussian_odds, gaussian_joint_odds, fit_log_noise_gp, gp_response_distribution, affine_moments,
)
profile_metadata = json.loads((paper / 'outputs/score_error_modes/metadata.json').read_text())
with np.load(paper / 'outputs/score_error_modes/clock.npz') as saved:
    forward_variance = saved['variance'].copy()
vref = profile_metadata['reference_variance']
sigma0 = np.sqrt(profile_metadata['data_variance'])
assert np.allclose(distance, np.log(forward_variance / vref))
assert settings['prior'] == 'exact' and settings['profile_fingerprint'] == profile_metadata['fingerprint']
assert settings['mixture'] == profile_metadata['mixture']
for profile in profiles.values():
    profile['a'] = forward_variance * profile['C']
    profile['u'] = forward_variance * profile['b'] / np.sqrt(vref)
    np.testing.assert_allclose(profile['a'], profile['a_linear'])
    np.testing.assert_allclose(profile['u'], profile['u_linear'])
    assert np.isfinite(profile['a']).all() and np.isfinite(profile['u']).all()
ids = list(profiles)
ordered_ids = sorted([cp for cp in ids if cp[1] >= minimum_epoch], key=lambda cp: (cp[1], cp[0]))
if len(ordered_ids) < 2:
    raise ValueError('Choose an epoch cutoff retaining at least two checkpoints.')
selected_set = set(ordered_ids)
analysis_gammas = np.asarray(settings['gammas'], float)
comparison_gammas = analysis_gammas[analysis_gammas > 0]
selected_a = np.stack([profiles[cp]['a'] for cp in ordered_ids])
selected_u = np.stack([profiles[cp]['u'] for cp in ordered_ids])
log_weights = integration_weights(np.log(sigma))
log_weights /= log_weights.sum()
clock_weights = integration_weights(distance)
piecewise_fit_space = 'log'  # Primary draft objective; 'magnitude' is also computed.
envelope_segments = 3       # Fixed descriptive family, not selected using empirical KL.
fit_restarts = 4
qmc_power, qmc_repeats = 14, 4  # 65,536 four-dimensional Gaussian draws per odds estimate.

# Attach the measured gamma=0 KL by checkpoint, bin count and residual arm.
empirical = pd.DataFrame(ablation)
keys = ['run', 'epoch', 'bins', 'residual_lambda']
assert not empirical.duplicated(keys + ['gamma']).any()
baseline = empirical.loc[empirical.gamma == 0, keys + ['kl_q_p', 'kl_p_q']].rename(
    columns={'kl_q_p': 'ode_kl_q_p', 'kl_p_q': 'ode_kl_p_q'})
empirical = empirical.merge(baseline, on=keys, how='left', validate='many_to_one')
assert empirical[['ode_kl_q_p', 'ode_kl_p_q']].notna().all().all()
for direction in directions:
    np.testing.assert_allclose(np.log10(empirical[f'kl_{direction}']/empirical[f'ode_kl_{direction}']),
                               empirical[f'log10_ratio_ode_{direction}'], atol=2e-14)
    empirical[f'bounded_{direction}'] = bounded_change(empirical[f'kl_{direction}'], empirical[f'ode_kl_{direction}'])

def export_table(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(output_dir / f'{name}.csv', index=False)
    return frame

print(f'Envelope/GP cohort: {len(ordered_ids)} checkpoints, epoch >= {minimum_epoch}; no a<1 filter.')
print('Mixture:', profile_metadata['mixture'])
print(f'Vref={vref:.6f}; data SD={sigma0:.6f}; shifted horizon={distance[-1]:.6f}.')

# %% [markdown]
# ## Fixed-$\gamma$ empirical odds
#
# The event is $D_{KL}^{(\lambda)}(\gamma)<D_{KL}^{(\lambda)}(0)$ at a **predeclared gamma**.
# Every arm uses its own measured ODE baseline. Negative
# $B=(h_\gamma-h_0)/(h_\gamma+h_0)$ means improvement; $B\in[-1,1]$ and $B=\tanh[\log(h_\gamma/h_0)/2]$.
# An exact double zero is undefined. The strict win rate and the fraction with improvement factor > 1.01
# are reported separately; equality is not a win.
#
# These cached experiments use the exact mixture prior, paired initial states and Brownian increments,
# 4 × 4,096 particles, 500 Euler / Euler–Maruyama steps in the standardized log-variance clock,
# and stop at $\sigma_{\min}=0.002$. The target is the mixture at **that same positive noise**.
# KL uses fixed exact-target quantile bins, pooled seed counts and a 0.5 pseudocount per bin.
# “Above floor” requires both numerator and ODE KL to exceed three times their respective exact-score controls;
# it is a diagnostic subset, not a significance test. Reported per-run ranges are descriptive;
# we do not attach binomial confidence intervals treating epochs as independent.

# %%
odds_rows, run_odds_rows = [], []
for cohort, cutoff in [('All epochs', 0), (f'Epoch ≥ {minimum_epoch}', minimum_epoch)]:
    subset = empirical[(empirical.epoch >= cutoff) & (empirical.gamma > 0)]
    for (gamma, bins, lam), group in subset.groupby(['gamma', 'bins', 'residual_lambda'], sort=True):
        for direction in directions:
            h, h0 = group[f'kl_{direction}'].to_numpy(), group[f'ode_kl_{direction}'].to_numpy()
            good = np.isfinite(h) & np.isfinite(h0) & (h+h0 > 0)
            above = group[f'above_control_floor_{direction}'].to_numpy(bool) & good
            win = h < h0
            run_rates = []
            for run in runs:
                keep = good & (group.run.to_numpy() == run)
                rate = np.mean(win[keep]) if keep.any() else np.nan
                run_rates.append(rate)
                run_odds_rows.append(dict(cohort=cohort, run=run, gamma=gamma, bins=bins,
                    arm='Full' if lam else 'Affine', direction=direction, n=int(keep.sum()), win_fraction=rate))
            odds_rows.append(dict(cohort=cohort, gamma=gamma, bins=bins,
                arm='Full' if lam else 'Affine', direction=direction, n=int(good.sum()),
                wins=int(win[good].sum()), ties=int(np.sum(h[good] == h0[good])),
                win_fraction=float(np.mean(win[good])), improvement_over_1pct=float(np.mean(h[good]*1.01 < h0[good])),
                median_bounded=float(np.median(bounded_change(h[good], h0[good]))),
                n_above_floor=int(above.sum()), win_fraction_above_floor=float(np.mean(win[above])) if above.any() else np.nan,
                run_min=float(np.nanmin(run_rates)), run_max=float(np.nanmax(run_rates))))
odds_table = export_table('empirical_odds', odds_rows)
run_odds_table = export_table('empirical_odds_by_run', run_odds_rows)
display(odds_table[(odds_table.bins == primary_bins) & odds_table.gamma.isin([.2, 1., 5.])].round(4))

fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
for col, (direction, label) in enumerate(directions.items()):
    rows = empirical[(empirical.bins == primary_bins) & np.isclose(empirical.gamma, evaluation_gamma)]
    for lam, arm, color in [(1, 'Full', 'tab:blue'), (0, 'Affine', 'tab:orange')]:
        arm_rows = rows[rows.residual_lambda == lam]
        values = arm_rows[f'bounded_{direction}']
        axes[0, col].hist(values, bins=np.linspace(-1, 1, 26), alpha=.5, color=color,
                          label=f'{arm}: {np.mean(values < 0):.1%} wins')
        axes[1, col].scatter(arm_rows[f'ode_kl_{direction}'], values, s=14, alpha=.4, color=color, label=arm)
    axes[0, col].axvline(0, color='black', ls='--')
    axes[0, col].set(title=f'{label}, fixed γ={evaluation_gamma:g}; all epochs', xlabel='Bounded KL change', ylabel='Checkpoints')
    axes[1, col].axhline(0, color='black', ls='--')
    axes[1, col].set(xscale='log', xlabel='Measured ODE KL', ylabel='Bounded KL change', ylim=(-1, 1))
    axes[0, col].legend()
show_figure(fig, 'fixed_gamma_changes')

fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
for ax, direction in zip(axes, directions):
    table = run_odds_table[(run_odds_table.bins == primary_bins) &
                          (run_odds_table.cohort == f'Epoch ≥ {minimum_epoch}') &
                          (run_odds_table.direction == direction)]
    for run in runs:
        for arm, style in [('Full', '-o'), ('Affine', '--s')]:
            rows = table[(table.run == run) & (table.arm == arm)].sort_values('gamma')
            ax.semilogx(rows.gamma, rows.win_fraction, style, color=run_colors[run], label=f'{run}: {arm}')
    ax.set(title=directions[direction], xlabel='Fixed γ', ylabel='Win fraction', ylim=(0, 1))
axes[0].legend(fontsize=8)
show_figure(fig, 'fixed_gamma_odds_by_run')

# %% [markdown]
# ## Spatial score-error modes and the new coordinates
#
# Project $s_\theta-s_{\rm exact}=b+C(x-\mu)+\varrho$ in $L^2(p_t)$ with 256 Gauss–Hermite nodes
# per mixture component. The residual is orthogonal to the affine functions; its energy is distinct
# from the nonlinearity of the exact mixture score. For the Gaussian surrogate,
#
# $$a=1-1/\alpha_\theta=VC,\qquad
# u=(\mu_\theta-\mu)/(\alpha_\theta\sqrt{V_{\rm ref}})=Vb/\sqrt{V_{\rm ref}}.$$
#
# These are **exact score coefficients**, even when large. Only the response/KL expansion is perturbative.
# No division by $1-a$, logarithm of $\alpha$, or $a<1$ filtering enters the fits below.
# We shift the reference endpoint to $V_{\rm ref}=V(0.002)$ consistently in the clock, normalization and KL;
# the relative SD change from the draft's $\sigma_0$ is about $1.64\times10^{-5}$.

# %%
ids = list(profiles)
random_ids = [ids[index] for index in
              np.random.default_rng(random_seed).choice(len(ids), 3, replace=False)]
checkpoint_colors = ['tab:blue', 'tab:orange', 'tab:green']

fig = plt.figure(figsize=(15, 8), layout='constrained')
grid = fig.add_gridspec(2, 3)
top_axes = [fig.add_subplot(grid[0, col]) for col in range(3)]
energy_axes = [fig.add_subplot(grid[1, col]) for col in range(3)]

for (run, epoch), color, energy_ax in zip(random_ids, checkpoint_colors, energy_axes):
    profile = profiles[(run, epoch)]
    label = f'{run}, epoch {epoch}'
    for ax, key in zip(top_axes, ['total_energy', 'b', 'C']):
        ax.plot(sigma, profile[key], color=color, label=label)
    for key, mode, style in [('mean_energy', 'Mean', '-'),
                             ('affine_energy', 'Affine', '--'),
                             ('residual_energy', 'Residual', ':')]:
        energy_ax.plot(distance, profile[key] / np.maximum(profile['total_energy'], 1e-300),
                       color=color, linestyle=style, linewidth=1.8, label=mode)
    energy_ax.set(title=f'Mode energies: {label}', xlabel='d = log(V / Vref)',
                  ylabel='Energy fraction', ylim=(0, 1))
    energy_ax.title.set_color(color)
    energy_ax.legend(fontsize=8)

top_axes[0].set(yscale='log', title='Score error: three random checkpoints',
                ylabel=r'$\mathbb{E}_{p_t}[\epsilon_\theta^2]$')
for ax, key, title in zip(top_axes[1:], ['b', 'C'],
                          ['Mean error b(t)', 'Affine coefficient C(t)']):
    peak = max(np.max(np.abs(profiles[checkpoint][key])) for checkpoint in random_ids)
    ax.set_yscale('symlog', linthresh=max(peak * 0.01, 1e-12))
    ax.axhline(0, color='gray', lw=0.7)
    ax.set(title=title, ylabel='Signed coefficient')
for ax in top_axes:
    ax.set_xscale('log')
    ax.set_xlabel(r'$t=\sigma$')
    ax.legend(fontsize=8)

show_figure(fig)

overview_ids = [checkpoint for checkpoint in profiles if checkpoint[1] >= minimum_epoch]
if not overview_ids:
    raise ValueError(f'No checkpoints at epoch >= {minimum_epoch}; lower minimum_epoch.')
overview_summary = summary[summary.epoch >= minimum_epoch]

overview_fig, overview_axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
profile_alpha = 0.3
epoch_values = np.array([epoch for _, epoch in profiles])
epoch_norm = plt.Normalize(vmin=epoch_values.min(), vmax=epoch_values.max())
epoch_cmap = plt.get_cmap('viridis')
ordered_ids = sorted(overview_ids, key=lambda checkpoint: (checkpoint[1], checkpoint[0]))

for ax, key, title in zip(overview_axes[:2], ['u', 'a'],
                          ['Mean coordinate u(t)', 'Shape coordinate a(t)']):
    for checkpoint in ordered_ids:
        epoch = checkpoint[1]
        ax.plot(sigma, profiles[checkpoint][key], color=epoch_cmap(epoch_norm(epoch)),
                alpha=profile_alpha, linewidth=0.9)
    ax.axhline(0, color='gray', lw=0.7)
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1e-2)
    ax.set(title=title, xlabel=r'$t=\sigma$', ylabel=f'Signed {key}(t)')

epoch_colorbar = overview_fig.colorbar(
    plt.cm.ScalarMappable(norm=epoch_norm, cmap=epoch_cmap), ax=overview_axes[:2],
    label='Training epoch', shrink=0.9, pad=0.02)
epoch_colorbar.set_ticks(np.unique(np.linspace(epoch_values.min(), epoch_values.max(), 5).round().astype(int)))

histogram_ax = overview_axes[2]
histogram_ax.hist(overview_summary.nonlinear_energy_fraction, bins=20, edgecolor='white')
histogram_ax.set(title='Integrated residual fraction',
                 xlabel='Fraction', ylabel='Checkpoints', xlim=(0, 1))
overview_fig.suptitle(f'Epoch ≥ {minimum_epoch}: {len(overview_ids)} checkpoints')
show_figure(overview_fig)


print(f'In the fitting cohort, a >= 1 at {(selected_a >= 1).sum()} / {selected_a.size} points, '
      f'in {(selected_a >= 1).any(axis=1).sum()} / {len(ordered_ids)} checkpoints. All are retained.')

# %% [markdown]
# ## Refit the power-law envelopes in $(a,u)$
#
# For the fixed epoch-selected cohort, compute $m_f=\langle f\rangle$,
# $R_f=\sqrt{\langle f^2\rangle}$ and $A_f=\sqrt{\langle(f-m_f)^2\rangle}$, so $R_f^2=m_f^2+A_f^2$.
# RMS is not the GP amplitude when the mean is nonzero. Fit both RMS and **centered SD** directly.
# Each checkpoint receives equal weight; equal epoch counts also weight the three runs equally.
#
# The continuous family is $\log\widehat R=\log B-p_1x-\sum_j(p_{j+1}-p_j)(x-\tau_j)_+$,
# $x=\log(\sigma/\sigma_{\min})$, $p_j\geq0$, with 1, 2 or 3 segments.
# Trapezoidal weights in $\log\sigma$ prevent the hybrid grid from overweighting low noise.
# RMS fits use both log-magnitude and magnitude objectives, as in 056; SD fits use the selected objective.
# The three-piece family is a declared descriptive choice. Better in-sample fit alone is not model-selection evidence.
# The symmetric curves are magnitude guides, not confidence bands or fitted signed means.

# %%
envelopes, envelope_fits, fit_rows = {}, {}, []
for name, values in [('a', selected_a), ('u', selected_u)]:
    mean, rms, sd = values.mean(axis=0), np.sqrt(np.mean(values**2, axis=0)), values.std(axis=0)
    np.testing.assert_allclose(rms*rms, mean*mean+sd*sd)
    envelopes[name] = dict(mean=mean, RMS=rms, SD=sd,
                          mean_square_fraction=float(log_weights @ (mean*mean)/(log_weights @ (rms*rms))))
    for statistic in ['RMS', 'SD']:
        observed = envelopes[name][statistic]
        log_fits = fit_piecewise_power_laws(sigma, observed, log_weights, max_segments=3,
                                           fit_space='log', seed=random_seed, restarts=fit_restarts)
        objectives = {'log': log_fits}
        if statistic == 'RMS' or piecewise_fit_space == 'magnitude':
            objectives['magnitude'] = fit_piecewise_power_laws(sigma, observed, log_weights, max_segments=3,
                fit_space='magnitude', seed=random_seed, restarts=fit_restarts, initial_fits=log_fits)
        for objective, fits in objectives.items():
            envelope_fits[(name, statistic, objective)] = fits
            for fit in fits:
                fit_rows.append(dict(coordinate=name, statistic=statistic, objective=objective,
                    K=fit['n_exponents'], amplitude=fit['amplitude'],
                    exponents=', '.join(f'{p:.6g}' for p in fit['exponents']),
                    breaks_sigma=', '.join(f'{t:.6g}' for t in fit['breakpoints']),
                    breaks_over_data_sd=', '.join(f'{t/sigma0:.6g}' for t in fit['breakpoints']),
                    log10_RMSE=fit['log10_RMSE'], relative_RMSE=fit['relative_RMSE'], R2=fit['R2'],
                    mean_square_fraction=envelopes[name]['mean_square_fraction'],
                    search_converged=fit['search_converged'], rate_bound_hit=fit['rate_bound_hit']))
fit_table = export_table('envelope_fits', fit_rows)
display(fit_table[fit_table.objective == piecewise_fit_space].round(5))
if (~fit_table.search_converged | fit_table.rate_bound_hit).any():
    print('Inspect optimizer diagnostics:', fit_table.loc[~fit_table.search_converged | fit_table.rate_bound_hit].to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(17, 8), layout='constrained')
fit_colors = ['#208f8d', '#d55e00', '#542788']
for row, (name, values) in enumerate([('a', selected_a), ('u', selected_u)]):
    for cp, values_i in zip(ordered_ids, values):
        axes[row, 0].plot(sigma, values_i, color=epoch_cmap(epoch_norm(cp[1])), alpha=.22, lw=.8)
    env = envelopes[name]
    fitted = envelope_fits[(name, 'RMS', piecewise_fit_space)][envelope_segments-1]
    axes[row, 0].plot(sigma, env['mean'], 'k-', lw=1.4, label='Ensemble mean')
    for sign in [-1, 1]:
        axes[row, 0].plot(sigma, sign*env['RMS'], 'k--', lw=1, label='± observed RMS' if sign == 1 else None)
        axes[row, 0].plot(sigma, sign*fitted['prediction'], color='crimson', ls=':', lw=2,
                          label=f'± fitted RMS, K={envelope_segments}' if sign == 1 else None)
    axes[row, 0].set_yscale('symlog', linthresh=1e-3)
    axes[row, 0].set(title=f'Signed {name}; all selected checkpoints', ylabel=name)
    for col, statistic in enumerate(['RMS', 'SD'], start=1):
        ax = axes[row, col]
        ax.plot(sigma, env[statistic], 'k-', label=f'Observed {statistic}')
        for color, fit in zip(fit_colors, envelope_fits[(name, statistic, piecewise_fit_space)]):
            ax.plot(sigma, fit['prediction'], color=color, label=f'K={fit["n_exponents"]}; log RMSE={fit["log10_RMSE"]:.3f}')
            ax.scatter(fit['breakpoints'], evaluate_power_law(fit['breakpoints'], fit), s=15, color=color)
        ax.set(yscale='log', title=f'{name}: {statistic}', ylabel=statistic)
    for ax in axes[row]:
        ax.set_xscale('log'); ax.set_xlabel('σ')
        ax.axvline(sigma0, color='gray', ls=':', lw=.9)
        ax.legend(fontsize=7)
fig.colorbar(plt.cm.ScalarMappable(norm=epoch_norm, cmap=epoch_cmap), ax=axes[:, 0], label='Epoch', shrink=.8)
show_figure(fig, 'linear_coordinate_envelopes')

profile_export = pd.DataFrame({'sigma': sigma, 'distance': distance})
for name in envelopes:
    for stat in ['mean', 'RMS', 'SD']:
        profile_export[f'{name}_{stat}'] = envelopes[name][stat]
    profile_export[f'{name}_RMS_fit'] = envelope_fits[(name, 'RMS', piecewise_fit_space)][envelope_segments-1]['prediction']
    profile_export[f'{name}_SD_fit'] = envelope_fits[(name, 'SD', piecewise_fit_space)][envelope_segments-1]['prediction']
export_table('envelope_profiles', profile_export)

norm_rows = []
low_noise = sigma < sigma0
for name in ['a', 'u']:
    for statistic in ['RMS', 'SD']:
        for kind, curve in [('Observed', envelopes[name][statistic]),
                            ('Fitted', envelope_fits[(name, statistic, piecewise_fit_space)][envelope_segments-1]['prediction'])]:
            norm = clock_weights @ curve
            norm_rows.append(dict(coordinate=name, statistic=statistic, kind=kind, L1_clock=norm,
                fraction_below_data_scale=(clock_weights[low_noise] @ curve[low_noise])/norm))
norm_table = export_table('envelope_clock_norms', norm_rows)
display(norm_table.round(5))


# %% [markdown]
# ## Finite-horizon predictions from the fitted envelopes
#
# With $d=\log(V/V_{\rm ref})$, the exact-prior response kernels are
#
# $$L_\gamma[a]=(1+\gamma)\int_0^\Lambda e^{-\gamma d}a(d)\,dd,\qquad
# M_\gamma[u]=\frac{1+\gamma}{2}\int_0^\Lambda e^{-(1+\gamma)d/2}u(d)\,dd,$$
# $$\widehat h_\gamma=L_\gamma^2/4+M_\gamma^2/2.$$
#
# First reproduce the draft's **centered independent Gaussian-amplitude benchmark**:
# $a=Z_a\widehat R_a$, $u=Z_u\widehat R_u$, $Z_a,Z_u\sim N(0,1)$ independently.
# The expected-KL ratio is $(L_\gamma^2/4+M_\gamma^2/2)/(L_0^2/4+M_0^2/2)$.
# The win probability follows Theorem 6.2; when $\Delta L<0<\Delta M$ it is
# $\frac2\pi\arctan\sqrt{-\Delta L/(2\Delta M)}$, where $\Delta L=L_\gamma^2-L_0^2$
# and $\Delta M=M_\gamma^2-M_0^2$. Both opposite-sign cases are handled.
# **Random signs of fixed magnitude give a deterministic KL comparison**, not this arctangent law.
# A ratio of expected KLs is not an expected ratio or a win probability.
#
# We integrate the actual finite interval $[0.002,80]$, with the endpoint-shifted clock;
# no infinite-horizon or pure-power threshold is substituted for a piecewise profile.
# The plotted optimal gamma is a descriptive optimum of the theoretical curve only.
# The response table also locates positive shape/ODE crossings within the declared search interval γ ≤ 30; it makes no claim about unresolved crossings outside that interval.

# %%
gamma_curve = np.unique(np.r_[0., np.geomspace(.001, 30, 301), analysis_gammas])
envelope_prediction_rows = []
from scipy.optimize import brentq
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
for k in [1, 2, 3]:
    afit = envelope_fits[('a', 'RMS', piecewise_fit_space)][k-1]
    ufit = envelope_fits[('u', 'RMS', piecewise_fit_space)][k-1]
    predicted = responses(distance, afit['prediction'], ufit['prediction'], gamma_curve)
    L, M, h = predicted['L'], predicted['M'], predicted['kl']
    odds = two_amplitude_odds(L*L-L[0]**2, M*M-M[0]**2)
    color = fit_colors[k-1]
    axes[0].plot(gamma_curve, L/L[0], color=color, label=f'Shape, K={k}')
    axes[0].plot(gamma_curve, M/M[0], color=color, ls='--', label=f'Mean, K={k}')
    axes[1].plot(gamma_curve, np.log10(h/h[0]), color=color, label=f'K={k}')
    axes[2].plot(gamma_curve[1:], odds[1:], color=color, label=f'K={k}')
    shape_opt = gamma_curve[np.argmin(L)]
    # Locate every positive crossing resolved in the declared [0.001, 30] search.
    shape_difference = L/L[0]-1
    crossing_indices = np.flatnonzero(shape_difference[1:-1]*shape_difference[2:] < 0)+1
    crossings = [brentq(lambda g: responses(distance, afit['prediction'], ufit['prediction'], [g])['L'][0]/L[0]-1,
                        gamma_curve[i], gamma_curve[i+1]) for i in crossing_indices]
    last_crossing = crossings[-1] if crossings else np.nan
    for gamma in comparison_gammas:
        idx = np.flatnonzero(gamma_curve == gamma)[0]
        envelope_prediction_rows.append(dict(K=k, gamma=gamma, shape_ratio=L[idx]/L[0], mean_ratio=M[idx]/M[0],
            log10_expected_KL_ratio=np.log10(h[idx]/h[0]), two_amplitude_odds=odds[idx],
            shape_optimal_gamma_on_grid=shape_opt, last_positive_shape_crossing_below_30=last_crossing, shape_L1=float(clock_weights @ afit['prediction'])))
for ax in axes:
    ax.set_xscale('symlog', linthresh=.01); ax.set_xlabel('γ'); ax.legend(fontsize=8)
axes[0].axhline(1, color='gray', ls=':'); axes[0].set(title='Response ratios', ylabel='Response / ODE response')
axes[1].axhline(0, color='gray', ls=':'); axes[1].set(title='Ratio of expected KLs', ylabel='log10 ratio')
axes[2].set(title='Independent Gaussian-amplitude odds', ylabel='Probability of improvement', ylim=(0, 1))
show_figure(fig, 'envelope_predictions')
envelope_prediction_table = export_table('envelope_predictions', envelope_prediction_rows)
display(envelope_prediction_table.round(5))

# Physical crossover changes slightly at a positive stopping level.
crossover_rows = []
for gamma in comparison_gammas:
    da = np.log1p(gamma)/gamma
    crossover_rows.append(dict(gamma=gamma,
        shape_crossover_sigma=np.sqrt(vref*np.exp(da)-profile_metadata['data_variance']),
        mean_crossover_sigma=np.sqrt(vref*np.exp(2*da)-profile_metadata['data_variance'])))
show_table(crossover_rows)

# %% [markdown]
# ## Signed profiles, cancellations, and full Gaussian moments
#
# Envelope predictions discard each checkpoint's sign pattern. Direct predictions retain it inside the
# integral. Here we recompute them from the **501 cached ablation grid nodes**, the same grid used
# for the cached 500-step Euler moment predictions, and verify agreement with the cache.
# The denser hybrid grid above remains the envelope/GP fitting grid.
#
# For the Gaussian surrogate we also integrate the exact moment equations
# $r'=-(1+\gamma)[(1-a)r-u]/2$ and $v'=[-\gamma+(1+\gamma)a]v+\gamma$, from $r=0,v=1$.
# The new check freezes linearly interpolated coefficients at refined interval midpoints and advances
# those constant-coefficient equations analytically. Comparing refinements 4 and 8 separates the
# numerical integration error from the small-error approximation; the original Euler curves are retained.
# This solves a **Gaussian surrogate**; the affine-error arm still includes the nonlinear exact mixture score.
#
# The cancellation diagnostic is $|L_0|/\int|a|\,dd$ (one means no cancellation).
# A small shape denominator need not destabilize the *total* KL ratio if the mean term is non-negligible.
# The $L^1$ norms and actual profile/moment discrepancy quantify the perturbative regime; no pointwise
# smallness or positive auxiliary Gaussian precision is imposed.

# %%
cache_a, cache_u = [], []
cache_profile_kl, cache_euler_kl = [], []
for cp in ids:
    with np.load(ablation_dir / 'checkpoints' / f'{cp[0]}_epoch{cp[1]:02d}.npz') as saved:
        s = saved['sigma'][::-1].copy()
        var = profile_metadata['data_variance'] + s*s
        d = np.log(var/vref)
        if cache_a:
            np.testing.assert_allclose(d, cached_distance)
        cached_distance = d
        cache_a.append(var*saved['C'][::-1])
        cache_u.append(var*saved['b'][::-1]/np.sqrt(vref))
        cache_profile_kl.append(saved['gaussian_profile_kl'].copy())
        cache_euler_kl.append(saved['gaussian_moment_kl'].copy())
cache_a, cache_u = np.stack(cache_a), np.stack(cache_u)
profile_predictions = responses(cached_distance, cache_a, cache_u, analysis_gammas)
np.testing.assert_allclose(profile_predictions['kl'], cache_profile_kl, rtol=1e-10, atol=1e-15)
kl_refined4, _, _ = affine_moments(cached_distance, cache_a, cache_u, analysis_gammas, refinement=4)
kl_refined8, refined_r, refined_v = affine_moments(cached_distance, cache_a, cache_u, analysis_gammas, refinement=8)
cache_euler_kl = np.stack(cache_euler_kl)
refined_log = np.log10(kl_refined8/kl_refined8[:, :1, :])
refinement_error = np.abs(refined_log-np.log10(kl_refined4/kl_refined4[:, :1, :]))
print(f'Max log-ratio change from refinement 4 to 8: {refinement_error.max():.3g} dex.')
cp_index = {cp: i for i, cp in enumerate(ids)}
gamma_index = {g: i for i, g in enumerate(analysis_gammas)}
for direction_index, direction in enumerate(directions):
    empirical[f'refined_log10_ratio_{direction}'] = [refined_log[cp_index[(row.run, row.epoch)], gamma_index[row.gamma], direction_index]
                                                     for row in empirical.itertuples()]

wc = integration_weights(cached_distance)
L1a, L1u = np.abs(cache_a) @ wc, np.abs(cache_u) @ wc
L, M = profile_predictions['L'], profile_predictions['M']
cancellation_rows = []
for i, (run, epoch) in enumerate(ids):
    cancellation_rows.append(dict(run=run, epoch=epoch, L1_a=L1a[i], L1_u=L1u[i],
        shape_cancellation=abs(L[i, 0])/L1a[i], mean_cancellation=abs(M[i, 0])/
            ((1/2*np.exp(-cached_distance/2)*np.abs(cache_u[i])) @ wc),
        predicted_ode_kl=profile_predictions['kl'][i, 0],
        ode_shape_KL_fraction=L[i, 0]**2/(4*profile_predictions['kl'][i, 0])))
cancellation_table = export_table('cancellation_diagnostics', cancellation_rows)
display(cancellation_table.groupby('run')[['L1_a', 'L1_u', 'shape_cancellation', 'predicted_ode_kl']].agg(['median', 'min', 'max']).round(5))

linearization_rows = []
for j, gamma in enumerate(analysis_gammas[1:], 1):
    plog = np.log10(profile_predictions['kl'][:, j]/profile_predictions['kl'][:, 0])
    for k, direction in enumerate(directions):
        gap = refined_log[:, j, k]-plog
        euler_gap = np.log10(cache_euler_kl[:, j, k]/cache_euler_kl[:, 0, k])-refined_log[:, j, k]
        linearization_rows.append(dict(gamma=gamma, direction=direction,
            median_abs_moment_profile_gap=float(np.median(np.abs(gap))),
            p95_abs_moment_profile_gap=float(np.quantile(np.abs(gap), .95)),
            max_abs_moment_profile_gap=float(np.max(np.abs(gap))),
            median_abs_euler_refined_gap=float(np.median(np.abs(euler_gap))),
            max_abs_euler_refined_gap=float(np.max(np.abs(euler_gap))),
            max_refinement_gap=float(refinement_error[:, j, k].max())))
linearization_table = export_table('moment_linearization_check', linearization_rows)
display(linearization_table.round(5))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
idx = gamma_index[evaluation_gamma]
plog = np.log10(profile_predictions['kl'][:, idx]/profile_predictions['kl'][:, 0])
for run in runs:
    keep = cancellation_table.run.to_numpy() == run
    axes[0].scatter(cancellation_table.shape_cancellation[keep], np.abs(refined_log[keep, idx, 0]-plog[keep]),
                    s=18, alpha=.5, color=run_colors[run], label=run)
    axes[1].scatter(cancellation_table.L1_a[keep], cancellation_table.L1_u[keep], s=18, alpha=.5, color=run_colors[run])
axes[0].set(xlabel='Shape cancellation ratio', ylabel='Refined moment / profile log-ratio gap (dex)', title=f'γ={evaluation_gamma:g}')
axes[0].legend(fontsize=8)
axes[1].set(xlabel='∫ |a| dd', ylabel='∫ |u| dd', xscale='log', yscale='log')
show_figure(fig, 'linearization_and_cancellation')

# %% [markdown]
# ## Original sweeps: historical comparison
#
# Keep the gamma curves and prediction/measurement scatter from 056 for context. Their recorded baseline
# is $\gamma_0\simeq0.01$, **not an ODE**. The original sweeps use a different endpoint convention
# (a last step to zero while their saved reference is diffused at 0.002).
# Nearest recorded gamma is stated in each scatter plot. These measurements are **not** used for the
# fixed-gamma odds or for tuning the envelope/GP model. The former sweep-minimum statistic is removed.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
for ax, (checkpoint_run, checkpoint_epoch), checkpoint_color in zip(axes, random_ids, checkpoint_colors):
    selected_sweep = np.sort(sweeps[(sweeps.run == checkpoint_run) &
                                   (sweeps.epoch == checkpoint_epoch)], order='gamma')
    for (direction, label), style in zip(directions.items(), ['-', '-.']):
        ax.semilogx(selected_sweep.gamma, selected_sweep[f'empirical_log10_ratio_{direction}'],
                    color=checkpoint_color, ls=style, label=label)
    ax.semilogx(selected_sweep.gamma, selected_sweep.profile_log10_ratio_recorded_baseline,
                color='black', ls='--', label='Gaussian profile response')
    ax.axhline(0, color='gray', lw=0.7)
    ax.set(title=f'{checkpoint_run}, epoch {checkpoint_epoch}', xlabel=r'$\gamma$',
           ylabel='log10(KL / recorded-baseline KL)')
    ax.title.set_color(checkpoint_color)
    ax.legend(fontsize=8)
show_figure(fig)

correlation_gammas = [0.2, 1.0, 5.0]
recorded_gammas = np.unique(sweeps.gamma)
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout='constrained')
for col, target_gamma in enumerate(correlation_gammas):
    recorded_gamma = recorded_gammas[np.argmin(np.abs(np.log(recorded_gammas / target_gamma)))]
    comparison = sweeps[np.isclose(sweeps.gamma, recorded_gamma)]
    for row, (direction, label) in enumerate(directions.items()):
        ax = axes[row, col]
        x = comparison.profile_log10_ratio_recorded_baseline
        y = comparison[f'empirical_log10_ratio_{direction}']
        good = np.isfinite(x) & np.isfinite(y)
        for run in runs:
            keep = good & (comparison.run == run)
            ax.scatter(x[keep], y[keep], s=20, alpha=0.3, color=run_colors[run], label=run)
        for (checkpoint_run, checkpoint_epoch), checkpoint_color in zip(random_ids, checkpoint_colors):
            highlight = good & (comparison.run == checkpoint_run) & (comparison.epoch == checkpoint_epoch)
            ax.scatter(x[highlight], y[highlight], s=160, marker='*', color=checkpoint_color,
                       edgecolor='black', linewidth=0.7, zorder=4,
                       label=f'{checkpoint_run}, epoch {checkpoint_epoch}')
        limits = [min(x[good].min(), y[good].min()), max(x[good].max(), y[good].max())]
        ax.plot(limits, limits, 'k--', lw=1)
        metrics = agreement(x, y)
        ax.set(xlabel='Predicted log10 ratio', ylabel='Measured log10 ratio',
               title=f'{label}, γ = {recorded_gamma:.4g}\n'
                     f'RMSE = {metrics["log_RMSE"]:.3f}; Spearman = {metrics["Spearman"]:.3f}; n = {metrics["n"]}')
axes[0, 0].legend(fontsize=8)
show_figure(fig)

# %% [markdown]
# ## Controlled removal of the nonlinear residual
#
# As in 056, compare $s_\lambda=s_{\rm exact}+b+C(x-\mu)+\lambda\varrho$ for $\lambda=1$ (full)
# and $\lambda=0$ (affine error only). The same three randomly selected checkpoints appear throughout.
# Dashed curves use signed Gaussian response profiles; dotted curves retain the cached 500-step Euler moments.
# The following absolute-gap scatter asks whether removing the residual improves **ratio-prediction agreement**,
# with each arm using its own ODE denominator. Points below the diagonal improve after removal.
# Pale points fail the control-floor diagnostic in at least one arm. This is a paired intervention,
# but it does not uniquely separate non-Gaussian dynamics, discretization and KL-estimation effects.

# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 10.5), layout='constrained')
for row_index, ((checkpoint_run, checkpoint_epoch), checkpoint_color) in enumerate(
        zip(random_ids, checkpoint_colors)):
    selected_ablation = ablation[(ablation.run == checkpoint_run) &
                                (ablation.epoch == checkpoint_epoch) & (ablation.bins == primary_bins)]
    for col, (direction, label) in enumerate(directions.items()):
        ax = axes[row_index, col]
        for residual_lambda, arm, style in [(1, 'Full learned score', 'o-'),
                                            (0, 'Affine error only', 's-.')]:
            rows = np.sort(selected_ablation[selected_ablation.residual_lambda == residual_lambda], order='gamma')
            ax.plot(rows.gamma, rows[f'log10_ratio_ode_{direction}'], style,
                    color=checkpoint_color, label=arm)
        ax.plot(rows.gamma, rows[f'gaussian_profile_log10_ratio_ode_{direction}'],
                'k--', label='Gaussian profile response')
        ax.plot(rows.gamma, rows[f'gaussian_moment_log10_ratio_ode_{direction}'],
                ':', color='tab:purple', label='Gaussian moments')
        ax.axhline(0, color='gray', lw=0.7)
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_xlim(0, rows.gamma.max() * 1.1)
        ax.set(title=f'{checkpoint_run}, epoch {checkpoint_epoch}: {label}',
               xlabel=r'$\gamma$', ylabel='log10(KL / own ODE KL)')
        ax.title.set_color(checkpoint_color)
        ax.legend(fontsize=8)
fig.suptitle(f'Controlled residual removal; {primary_bins} target-quantile bins')
show_figure(fig)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
paired_summary = []
for ax, (direction, label) in zip(axes, directions.items()):
    rows = paired[(paired.bins == primary_bins) & np.isclose(paired.gamma, evaluation_gamma) &
                  (paired.kl_direction == direction)]
    above = rows.both_arms_above_control_floor.astype(bool)
    for keep, color, name in [(above, 'tab:blue', 'Above control floor'), (~above, 'lightgray', 'Near control floor')]:
        ax.scatter(rows.full_profile_disagreement[keep], rows.affine_profile_disagreement[keep],
                   c=color, s=22, alpha=0.75, label=name)
    limit = max(rows.full_profile_disagreement.max(), rows.affine_profile_disagreement.max()) * 1.03
    ax.plot([0, limit], [0, limit], 'k--', lw=1)
    ax.set(xlabel='Full-score absolute prediction gap (dex)',
           ylabel='Affine-only absolute prediction gap (dex)', title=f'{label}, γ = {evaluation_gamma:g}')
    ax.legend(fontsize=8)
    energy = np.array([summary.nonlinear_energy_fraction[(summary.run == row.run) &
                                                       (summary.epoch == row.epoch)][0] for row in rows])
    paired_summary.append(dict(direction=direction, gamma=evaluation_gamma, n=len(rows),
        fraction_with_smaller_gap=float(np.mean(rows.disagreement_reduction > 0)),
        median_full_gap=float(np.median(rows.full_profile_disagreement)),
        median_affine_gap=float(np.median(rows.affine_profile_disagreement)),
        residual_fraction_vs_abs_effect_Spearman=float(spearmanr(energy, np.abs(rows.nonlinear_effect_log10_ratio)).statistic),
        above_control_floor=int(above.sum())))
show_figure(fig)
show_table(paired_summary)
export_table('residual_ablation_summary', paired_summary)

# %% [markdown]
# ## Agreement, sign confusion and bounded KL change
#
# Retain log-RMSE, ratio-RMSE, bias, skill relative to the ratio-one predictor, Pearson and Spearman.
# Also compare bounded changes $B$ and report **TP, FP, FN, TN**, where positive means “stochasticity helps.”
# The binary classifier uses a strict negative change; exact ties count as no improvement and are listed.
# Accuracy and balanced accuracy are distinct from correlation. Undefined log ratios are excluded and counted
# through `n`; the bounded statistic itself has no division by $h_0$.
#
# The main comparison includes every checkpoint, as in 056. Set `above_floor_only=True` to apply the
# predefined floor filter to this section. The CSV retains all four positive cached gammas. Confusion matrices
# and bounded-change scatter below use `evaluation_gamma` for both arms and both KL directions.

# %%
agreement_rows = []
for gamma in comparison_gammas:
    for direction in directions:
        for lam, arm in [(1, 'Full'), (0, 'Affine')]:
            rows = empirical[np.isclose(empirical.gamma, gamma) & (empirical.bins == primary_bins) &
                             (empirical.residual_lambda == lam)]
            if above_floor_only:
                rows = rows[rows[f'above_control_floor_{direction}']]
            measured = rows[f'log10_ratio_ode_{direction}'].to_numpy()
            for name, predicted in [
                ('Signed profile', rows[f'gaussian_profile_log10_ratio_ode_{direction}'].to_numpy()),
                ('Euler moments', rows[f'gaussian_moment_log10_ratio_ode_{direction}'].to_numpy()),
                ('Refined moments', rows[f'refined_log10_ratio_{direction}'].to_numpy()),
                ('Ratio = 1', np.zeros(len(rows)))]:
                valid = np.isfinite(predicted) & np.isfinite(measured)
                y, p = measured[valid], predicted[valid]
                truth, guess = y < 0, p < 0
                tp, fp = int(np.sum(truth & guess)), int(np.sum(~truth & guess))
                fn, tn = int(np.sum(truth & ~guess)), int(np.sum(~truth & ~guess))
                bounded_error = bounded_from_log10(p)-bounded_from_log10(y)
                balanced = .5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else np.nan
                agreement_rows.append(dict(gamma=gamma, direction=direction, arm=arm, prediction=name,
                    **agreement(predicted, measured), bounded_RMSE=float(np.sqrt(np.mean(bounded_error**2))) if len(y) else np.nan,
                    bounded_bias=float(np.mean(bounded_error)) if len(y) else np.nan,
                    TP=tp, FP=fp, FN=fn, TN=tn, accuracy=float(np.mean(truth == guess)) if len(y) else np.nan,
                    balanced_accuracy=balanced, measured_ties=int(np.sum(y == 0)), predicted_ties=int(np.sum(p == 0))))
agreement_table = export_table('agreement_metrics', agreement_rows)
display(agreement_table[(agreement_table.prediction == 'Signed profile') & agreement_table.gamma.isin([.2, 1., 5.])].round(4))

fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout='constrained')
for row_index, direction in enumerate(directions):
    for col_index, (lam, arm) in enumerate([(1, 'Full'), (0, 'Affine')]):
        data = empirical[np.isclose(empirical.gamma, evaluation_gamma) & (empirical.bins == primary_bins) &
                         (empirical.residual_lambda == lam)]
        if above_floor_only:
            data = data[data[f'above_control_floor_{direction}']]
        prediction = data[f'gaussian_profile_log10_ratio_ode_{direction}'].to_numpy()
        measured = data[f'log10_ratio_ode_{direction}'].to_numpy()
        ax = axes[row_index, col_index*2]
        for run in runs:
            keep = data.run.to_numpy() == run
            ax.scatter(bounded_from_log10(prediction[keep]), bounded_from_log10(measured[keep]),
                       s=17, alpha=.5, color=run_colors[run], label=run)
        ax.plot([-1, 1], [-1, 1], 'k--', lw=1)
        ax.axvline(0, color='gray', lw=.6); ax.axhline(0, color='gray', lw=.6)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel='Predicted bounded change', ylabel='Measured bounded change',
               title=f'{arm}: {directions[direction]}')
        ax = axes[row_index, col_index*2+1]
        item = agreement_table[(agreement_table.gamma == evaluation_gamma) & (agreement_table.direction == direction) &
                               (agreement_table.arm == arm) & (agreement_table.prediction == 'Signed profile')].iloc[0]
        matrix = np.array([[item.TP, item.FN], [item.FP, item.TN]], dtype=int)
        ax.imshow(matrix, cmap='Blues', vmin=0)
        for (y, x), value in np.ndenumerate(matrix):
            ax.text(x, y, str(value), ha='center', va='center', color='black')
        ax.set(xticks=[0, 1], xticklabels=['Win', 'No win'], yticks=[0, 1], yticklabels=['Win', 'No win'],
               xlabel='Predicted', ylabel='Measured', title=f'Accuracy {item.accuracy:.1%}')
fig.suptitle(f'Fixed γ={evaluation_gamma:g}; {primary_bins} bins; signed-profile prediction')
show_figure(fig, 'bounded_agreement_and_confusion')

# %% [markdown]
# ## Gaussian processes calibrated in log noise
#
# Use $a=m_a+A_aG_a(x)$ and $u=m_u+A_uG_u(x)$, $x=\log(\sigma/\sigma_0)$, with
# OU correlation $k(\Delta x)=\exp(-|\Delta x|/\ell)$. Estimate the mean and centered SD pointwise,
# interpolate profiles to 257 uniformly spaced log-noise positions, standardize pointwise, and average
# lag products across positions and checkpoints. Fit $\ell$ by pair-count-weighted least squares through
# one third of the observed log-noise span. ACF residuals, including negative correlations, remain visible;
# stationarity and Gaussianity are modeling assumptions, not consequences of the envelope fit.
#
# The main GP uses the fitted **SD** envelope and the empirical mean. A second calculation uses the measured
# SD as a sensitivity check. Shape-only odds use Theorem 7.1 with the nonzero mean retained. Their empirical
# comparison is the **projected shape-only Gaussian-response win fraction**, not an unperformed shape-only
# mixture sampling experiment. Both-mode odds use the four-dimensional Gaussian quadratic form in Section 7.4,
# with independent shape/mean processes as an explicit baseline. A Gaussian fit to the empirical four response
# coordinates also retains cross-mode covariance and tests that independence assumption.
#
# The reported numerical SE comes from four scrambled Sobol integrations; it is not sampling-seed or
# training-run uncertainty. All these probability models describe the leading-order Gaussian surrogate.

# %%
gp_models = {name: fit_log_noise_gp(sigma, values) for name, values in [('a', selected_a), ('u', selected_u)]}
gp_fit_rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
for name, color in [('a', 'tab:blue'), ('u', 'tab:orange')]:
    model = gp_models[name]
    axes[0].plot(model['lag'], model['acf'], color=color, label=f'{name}: observed')
    axes[0].plot(model['lag'], np.exp(-model['lag']/model['length']), '--', color=color,
                 label=f'{name}: OU ℓ={model["length"]:.3g}')
    gp_fit_rows.append(dict(coordinate=name, length_log_sigma=model['length'], acf_RMSE=model['acf_rmse'],
                           length_at_bound=model['length_at_bound'], mean_square_fraction=envelopes[name]['mean_square_fraction']))
    axes[1].semilogx(sigma, envelopes[name]['mean']**2/envelopes[name]['RMS']**2, color=color, label=name)
za, zu = gp_models['a']['standardized'], gp_models['u']['standardized']
axes[2].plot(gp_models['a']['log_grid'], np.mean(za*zu, axis=0), color='purple')
axes[0].axhline(0, color='gray', lw=.8)
axes[0].set(xlabel='Lag in log σ', ylabel='Correlation', title='Pooled normalized autocorrelation')
axes[0].legend(fontsize=8)
axes[1].set(xlabel='σ', ylabel='Mean² / RMS²', title='Systematic component', ylim=(0, 1)); axes[1].legend()
axes[2].axhline(0, color='gray', lw=.8)
axes[2].set(xlabel='log σ', ylabel='Same-time correlation', title='Shape / mean dependence', ylim=(-1, 1))
show_figure(fig, 'gp_calibration_diagnostics')
gp_fit_table = export_table('gp_fit', gp_fit_rows)
display(gp_fit_table.round(5))

def projected_gp(models, amplitudes):
    return {name: gp_response_distribution(distance, sigma, models[name]['mean'], amplitudes[name],
                 models[name]['length'], analysis_gammas, mode=name) for name in ['a', 'u']}

def independent_joint_parameters(projected, j):
    indices = [0, j]
    ma, ca = projected['a']; mu, cu = projected['u']
    mean = np.r_[ma[indices], mu[indices]]
    cov = np.zeros((4, 4))
    cov[:2, :2] = ca[np.ix_(indices, indices)]
    cov[2:, 2:] = cu[np.ix_(indices, indices)]
    return mean, cov

selected_response = responses(distance, selected_a, selected_u, analysis_gammas)
gp_odds_rows = []
for amplitude_kind in ['Fitted SD', 'Measured SD']:
    amplitudes = {name: envelope_fits[(name, 'SD', piecewise_fit_space)][envelope_segments-1]['prediction']
                  if amplitude_kind == 'Fitted SD' else gp_models[name]['sd'] for name in gp_models}
    projected = projected_gp(gp_models, amplitudes)
    for j, gamma in enumerate(analysis_gammas[1:], 1):
        mean, cov = independent_joint_parameters(projected, j)
        shape_probability = shape_gaussian_odds(mean[:2], cov[:2, :2])
        joint = gaussian_joint_odds(mean, cov, seed=random_seed, power=qmc_power, repeats=qmc_repeats)
        expected_ode = (mean[0]**2+cov[0, 0])/4+(mean[2]**2+cov[2, 2])/2
        expected_sde = (mean[1]**2+cov[1, 1])/4+(mean[3]**2+cov[3, 3])/2
        gp_odds_rows.append(dict(model=f'Independent OU GP: {amplitude_kind}', gamma=gamma,
            shape_odds=shape_probability, both_mode_odds=joint['probability'], numerical_SE=joint['numerical_se'],
            log10_expected_KL_ratio=np.log10(expected_sde/expected_ode)))
for j, gamma in enumerate(analysis_gammas[1:], 1):
    X = np.column_stack([selected_response['L'][:, 0], selected_response['L'][:, j],
                         selected_response['M'][:, 0], selected_response['M'][:, j]])
    mean, cov = X.mean(axis=0), np.cov(X, rowvar=False, ddof=0)
    joint = gaussian_joint_odds(mean, cov, seed=random_seed, power=qmc_power, repeats=qmc_repeats)
    gp_odds_rows.append(dict(model='Empirical 4D Gaussian (cross covariance retained)', gamma=gamma,
        shape_odds=shape_gaussian_odds(mean[:2], cov[:2, :2]), both_mode_odds=joint['probability'],
        numerical_SE=joint['numerical_se'], log10_expected_KL_ratio=np.log10(
            np.mean(selected_response['kl'][:, j])/np.mean(selected_response['kl'][:, 0]))))
    gp_odds_rows.append(dict(model='Observed signed Gaussian responses', gamma=gamma,
        shape_odds=float(np.mean(X[:, 1]**2 < X[:, 0]**2)),
        both_mode_odds=float(np.mean(selected_response['kl'][:, j] < selected_response['kl'][:, 0])),
        numerical_SE=np.nan, log10_expected_KL_ratio=np.log10(
            np.mean(selected_response['kl'][:, j])/np.mean(selected_response['kl'][:, 0]))))
gp_odds_table = export_table('gp_odds', gp_odds_rows)
display(gp_odds_table.round(5))

# Cohort-matched comparison: every point has the epoch >= minimum_epoch selection.
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
for name, rows in gp_odds_table.groupby('model', sort=False):
    axes[0].semilogx(rows.gamma, rows.shape_odds, 'o-', label=name)
    for ax in axes[1:]:
        ax.semilogx(rows.gamma, rows.both_mode_odds, 'o-', alpha=.8, label=name)
for ax, direction in zip(axes[1:], directions):
    rows = odds_table[(odds_table.cohort == f'Epoch ≥ {minimum_epoch}') & (odds_table.bins == primary_bins) &
                      (odds_table.direction == direction)]
    for arm, style in [('Full', 'k-s'), ('Affine', 'k--^')]:
        subset = rows[rows.arm == arm]
        ax.semilogx(subset.gamma, subset.win_fraction, style, lw=2, label=f'Measured {arm}')
    benchmark = envelope_prediction_table[envelope_prediction_table.K == envelope_segments]
    ax.semilogx(benchmark.gamma, benchmark.two_amplitude_odds, ':', color='brown', label='Two Gaussian amplitudes / RMS')
    ax.set(title=f'Both modes vs mixture: {directions[direction]}')
axes[0].set(title='Shape only: Gaussian surrogate')
for ax in axes:
    ax.set(xlabel='Fixed γ', ylabel='Win probability / fraction', ylim=(0, 1))
axes[0].legend(fontsize=6); axes[1].legend(fontsize=6)
show_figure(fig, 'gp_and_empirical_odds')

# Refine every log-noise interval, keeping the same fitted functions and interpolated mean.
# This checks integration resolution independently of fitting and QMC repeat uncertainty.
x = np.log(sigma)
fine_sigma = np.exp(np.sort(np.r_[x, (x[1:]+x[:-1])/2]))
fine_sigma[[0, -1]] = sigma[[0, -1]]
fine_distance = np.log((profile_metadata['data_variance']+fine_sigma**2)/vref)
fine_projected = {}
for name in ['a', 'u']:
    fine_mean = np.interp(np.log(fine_sigma), x, gp_models[name]['mean'])
    fine_sd = evaluate_power_law(fine_sigma, envelope_fits[(name, 'SD', piecewise_fit_space)][envelope_segments-1])
    fine_projected[name] = gp_response_distribution(fine_distance, fine_sigma, fine_mean, fine_sd,
                                                    gp_models[name]['length'], analysis_gammas, mode=name)
gp_resolution_rows = []
for j, gamma in enumerate(analysis_gammas[1:], 1):
    mean, cov = independent_joint_parameters(fine_projected, j)
    shape = shape_gaussian_odds(mean[:2], cov[:2, :2])
    joint = gaussian_joint_odds(mean, cov, seed=random_seed, power=qmc_power, repeats=qmc_repeats)
    coarse = gp_odds_table[(gp_odds_table.model == 'Independent OU GP: Fitted SD') & (gp_odds_table.gamma == gamma)].iloc[0]
    gp_resolution_rows.append(dict(gamma=gamma, coarse_nodes=len(sigma), fine_nodes=len(fine_sigma),
        shape_odds_change=shape-coarse.shape_odds, both_odds_change=joint['probability']-coarse.both_mode_odds,
        fine_numerical_SE=joint['numerical_se']))
gp_resolution_table = export_table('gp_quadrature_check', gp_resolution_rows)
display(gp_resolution_table.round(6))


# %% [markdown]
# ### Training-run holdout and correlation-length sensitivity
#
# For each held-out run, estimate mean, measured SD and OU length from the other two runs only,
# then compare predicted odds with the held-out signed-profile win fraction and both mixture-sampling arms.
# This is a diagnostic of transfer across the three available trajectories; three folds do not identify a
# population uncertainty interval. The measured-SD GP avoids refitting envelope breakpoints in each fold.
# The main GP remains an in-sample descriptive fit.
#
# The shape-only length sensitivity preserves the measured nonzero mean. We also show the centered version;
# the draft's sharp/white-noise limits assume zero mean and cannot simply be substituted into the noncentered model.
# The plotted lengths are finite and resolved on the integration grid, not a discrete approximation to an
# arbitrarily short correlation limit.

# %%
holdout_rows = []
cp_runs = np.array([cp[0] for cp in ordered_ids])
for run in runs:
    train, test = cp_runs != run, cp_runs == run
    if train.sum() < 2 or not test.any():
        continue
    models = {name: fit_log_noise_gp(sigma, values[train]) for name, values in [('a', selected_a), ('u', selected_u)]}
    projected = projected_gp(models, {name: model['sd'] for name, model in models.items()})
    for j, gamma in enumerate(analysis_gammas[1:], 1):
        mean, cov = independent_joint_parameters(projected, j)
        predicted = gaussian_joint_odds(mean, cov, seed=random_seed, power=qmc_power, repeats=qmc_repeats)
        record = dict(held_out_run=run, gamma=gamma, n_train=int(train.sum()), n_test=int(test.sum()),
            shape_length=models['a']['length'], mean_length=models['u']['length'],
            GP_shape_odds=shape_gaussian_odds(mean[:2], cov[:2, :2]), GP_both_odds=predicted['probability'],
            numerical_SE=predicted['numerical_se'],
            held_out_shape_fraction=float(np.mean(selected_response['L'][test, j]**2 < selected_response['L'][test, 0]**2)),
            held_out_both_fraction=float(np.mean(selected_response['kl'][test, j] < selected_response['kl'][test, 0])))
        for direction in directions:
            for arm in ['Full', 'Affine']:
                measured = run_odds_table[(run_odds_table.run == run) & (run_odds_table.gamma == gamma) &
                    (run_odds_table.cohort == f'Epoch ≥ {minimum_epoch}') & (run_odds_table.bins == primary_bins) &
                    (run_odds_table.direction == direction) & (run_odds_table.arm == arm)]
                record[f'measured_{arm}_{direction}'] = measured.win_fraction.iloc[0]
        holdout_rows.append(record)
holdout_table = export_table('gp_run_holdout', holdout_rows)
display(holdout_table.round(4))

fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
shape_amplitude = envelope_fits[('a', 'SD', piecewise_fit_space)][envelope_segments-1]['prediction']
length_grid = np.geomspace(max(.15, np.diff(np.log(sigma)).max()*2), 30, 45)
length_rows = []
for gamma, color in zip([.2, 1., 5.], ['tab:blue', 'tab:orange', 'tab:green']):
    centered, noncentered = [], []
    for length in length_grid:
        mean, cov = gp_response_distribution(distance, sigma, gp_models['a']['mean'], shape_amplitude, length, [0, gamma])
        centered.append(shape_gaussian_odds(np.zeros(2), cov))
        noncentered.append(shape_gaussian_odds(mean, cov))
        length_rows.append(dict(gamma=gamma, length=length, centered_odds=centered[-1], noncentered_odds=noncentered[-1]))
    ax.semilogx(length_grid, noncentered, '-', color=color, label=f'γ={gamma:g}, measured mean')
    ax.semilogx(length_grid, centered, '--', color=color, label=f'γ={gamma:g}, zero mean')
ax.axvline(gp_models['a']['length'], color='black', ls=':', label='Fitted shape length')
ax.set(xlabel='OU length in log σ', ylabel='Shape-only win probability', ylim=(0, 1))
ax.legend(fontsize=8, ncol=2)
show_figure(fig, 'gp_length_sensitivity')
export_table('gp_length_sensitivity', length_rows)

# %% [markdown]
# ## Bin sensitivity and reproducibility
#
# The cached 32-, 64- and 128-bin partitions give a direct estimator-sensitivity check without new samples.
# A change in win classification close to the exact-score floor should not be interpreted as strong evidence
# for or against stochasticity. Full CSV tables retain both the complete cohort and the above-floor subset.
# No mixture parameters, model weights or cached experiments are modified.

# %%
sensitivity = odds_table[(odds_table.cohort == f'Epoch ≥ {minimum_epoch}') & np.isclose(odds_table.gamma, evaluation_gamma)]
display(sensitivity[['bins', 'arm', 'direction', 'n', 'win_fraction', 'n_above_floor', 'win_fraction_above_floor', 'run_min', 'run_max']].round(4))
# Shared delete-one-seed sensitivity preserves pairing across all checkpoints.
# This is a Monte Carlo stability range, not a confidence interval over networks.
from paper_odds.score_error_analysis.residual_ablation import binned_kl
seed_win_counts = np.zeros((len(analysis_gammas), 2, len(settings['seeds']), 2))
for cp in ordered_ids:
    with np.load(ablation_dir / 'checkpoints' / f'{cp[0]}_epoch{cp[1]:02d}.npz') as saved:
        counts = saved[f'counts_{primary_bins}']
        pooled = counts.sum(axis=2, keepdims=True)
        delete_kl = binned_kl(pooled-counts, pseudocount=settings['pseudocount'])
        seed_win_counts += delete_kl < delete_kl[:1]
seed_rows = []
for j, gamma in enumerate(analysis_gammas[1:], 1):
    for arm_index, lam in enumerate(settings['lambdas']):
        for direction_index, direction in enumerate(directions):
            rates = seed_win_counts[j, arm_index, :, direction_index]/len(ordered_ids)
            seed_rows.append(dict(gamma=gamma, arm='Full' if lam else 'Affine', direction=direction,
                                  delete_one_seed_min=rates.min(), delete_one_seed_max=rates.max()))
seed_sensitivity_table = export_table('seed_deletion_sensitivity', seed_rows)
display(seed_sensitivity_table.round(4))

hash_paths = [paper/'056-stoch_tests-wo_fits.ipynb',
              paper/'The_odds_for_diffusion_stochastic_sampling (2).pdf',
              paper/'outputs/score_error_modes/metadata.json', paper/'outputs/score_error_modes/clock.npz',
              ablation_dir/'metadata.json', ablation_dir/'ablation_results.csv', ablation_dir/'paired_effects.csv',
              paper/'score_error_analysis/draft057.py', paper/'score_error_analysis/rms_power_laws.py',
              paper/'score_error_analysis/score_error_modes.py', paper/'score_error_analysis/residual_ablation.py']
hash_paths += sorted((paper/'outputs/score_error_modes/profiles').glob('*.npz'))
hash_paths += sorted((ablation_dir/'checkpoints').glob('*.npz'))
import scipy
notebook_sources = [cell['source'] for cell in json.loads((paper/'057-stoch_tests.ipynb').read_text())['cells']]
notebook_source_hash = hashlib.sha256(json.dumps(notebook_sources, ensure_ascii=False).encode()).hexdigest()
manifest = dict(notebook='057-stoch_tests.ipynb', draft_date='2026-09-28', minimum_epoch=minimum_epoch,
    selected_checkpoints=[f'{run}:{epoch}' for run, epoch in ordered_ids], primary_bins=primary_bins,
    evaluation_gamma=evaluation_gamma, comparison_gammas=comparison_gammas.tolist(), random_seed=random_seed,
    coordinates='a=V*C; u=V*b/sqrt(Vref); no a<1 mask', reference_variance=vref,
    stopping_sigma=float(sigma[0]), horizon=float(distance[-1]), envelope_objective=piecewise_fit_space,
    envelope_segments=envelope_segments, fit_restarts=fit_restarts, above_floor_only=above_floor_only,
    gp='OU in log(sigma), empirical mean, fitted centered SD; independent modes baseline',
    gp_grid_points=257, gp_max_lag_fraction=1/3, qmc_power=qmc_power, qmc_repeats=qmc_repeats,
    no_new_sampling=True, no_retraining=True, future_gpu_device=6,
    profile_fingerprint=profile_metadata['fingerprint'], ablation_fingerprint=settings['fingerprint'],
    numpy_version=np.__version__, scipy_version=scipy.__version__, notebook_source_sha256=notebook_source_hash,
    source_sha256={str(path.relative_to(paper)): hashlib.sha256(path.read_bytes()).hexdigest() for path in hash_paths})
(output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f'Wrote figures, tables and provenance to {output_dir}')

# %% [markdown]
# ## Notes for revising the draft
#
# The detailed numerical findings and suggested wording are in
# [057-draft-notes.md](057-draft-notes.md). In particular, replace provisional Tables 3–4 using the
# new linear-coordinate fits, retain the finite stopping level throughout, and distinguish the
# Gaussian surrogate's probability model from the empirical three-run checkpoint distribution.
# The randomization of mixture parameters and the associated retraining remain deferred.

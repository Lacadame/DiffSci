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
#     display_name: ddpm_env
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
#     version: 3.12.2
# ---

# %% [markdown]
# # Score error and stochastic sampling
#
# A compact analysis of **144 checkpoints from three training runs** on the one-dimensional Gaussian mixture. It covers sampling quality, the spatial decomposition of score error, direct Gaussian predictions, and a controlled intervention on the nonlinear residual.
#
# Run all cells using NumPy, SciPy, Matplotlib and IPython. The notebook reads the existing `outputs/score_error_modes/` and `outputs/residual_ablation_euler500/` results beside it; checkpoint inference and sampling are already cached. Change the selections in the next cell to explore another checkpoint, stochasticity, or KL partition. Here $p$ is the target distribution and $q$ is the sampler distribution.
#
# All sampled trajectories use **500 steps**, with Euler for the ODE and Euler–Maruyama for the SDE. The Gaussian moment equations also use 500-step Euler. Cache settings are checked below; regeneration instructions are in [the analysis README](score_error_analysis/README.md#notebook-056-euler-with-500-steps).

# %%
from pathlib import Path
from io import BytesIO
from html import escape
import json
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

def show_figure(fig):
    with BytesIO() as buffer:
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
    plt.close(fig)

def show_table(rows):
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
print('Integrators: Euler (ODE), Euler–Maruyama (SDE); Euler for Gaussian moments.')

# %% [markdown]
# ## Sampling quality in the original sweeps
#
# The original saved time grids contain 501 nodes, confirming **500 integration steps**. The sweep script uses Euler–Maruyama for stochastic sampling and Euler for the ODE, on the EDM noise schedule ending at zero. The saved stochastic KL values use a reference diffused at $\sigma=0.002$.
#
# The smallest recorded stochasticity is **$\gamma_0\simeq0.01$**, not an ODE measurement. For consistency with the original notebook, the improvement statistic is
#
# $$I=\frac{D_{KL}(\gamma_0)}{\min_{j\geq3}D_{KL}(\gamma_j)},$$
#
# with gammas sorted and indexed from zero. Thus the candidate minimum excludes the first three recorded values. $I>1$ indicates improvement over the recorded baseline. This minimum is selected on the same sweep; it is descriptive, not an independent estimate of a chosen sampler's performance.

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for col, (direction, label) in enumerate(directions.items()):
    values, minima = [], []
    for run, epoch in profiles:
        rows = np.sort(sweeps[(sweeps.run == run) & (sweeps.epoch == epoch)], order='gamma')
        kl = rows[f'empirical_kl_{direction}']
        values.append(kl[0] / np.min(kl[3:]))
        minima.append(np.min(kl[3:]))
    values, minima = np.asarray(values), np.asarray(minima)
    axes[0, col].hist(values, bins=20, edgecolor='white')
    axes[0, col].axvline(1, color='black', ls='--')
    axes[0, col].set(xlabel='Improvement I', ylabel='Checkpoints',
                     title=f'{label}: {np.mean(values > 1.01):.1%} have I > 1.01')
    axes[1, col].scatter(minima, values, s=20, alpha=0.65)
    axes[1, col].set(xscale='log', xlabel='Minimum candidate KL', ylabel='Improvement I')
show_figure(fig)

# %% [markdown]
# ## Score-error profiles and spatial modes
#
# For $X\sim p_t$ and $t=\sigma>0$, use spatial least squares to decompose
#
# $$\epsilon_\theta(x,t)=s_\theta(x,t)-s_{\rm exact}(x,t)
# =b(t)+C(t)(x-\mu)+r(x,t).$$
#
# The mixture has $\mu=-0.01$ and $V(t)=0.1219+t^2$. Its mean, affine and residual energies add to $\mathbb E_{p_t}[\epsilon_\theta^2]$. The residual is orthogonal to affine functions under $p_t$; it is distinct from the nonlinearity of the exact mixture score.
#
# For the Gaussian response, use $d=\log(V/V_{\rm ref})$, $a=VC$, and $u=Vb/\sqrt{V_{\rm ref}}$, where $V_{\rm ref}=V(0.002)$. These are signed profiles. The top row compares three random checkpoints with consistent colors. The second row shows their mode-energy fractions separately, keeping the checkpoint colors and using line styles to distinguish mean, affine, and residual energy. Change `random_seed` to select another group. Immediately after the six-panel figure, a three-panel overview shows the mean-error and affine-coefficient profiles for checkpoints with `epoch >= minimum_epoch` (default 20), followed by the histogram for that same subset. Translucent profile lines use a shared viridis color scale for training epoch across all three runs (kept fixed when changing the cutoff); symmetric-log vertical axes retain both signs. The histogram summarizes $\int E_{\rm residual}\,dd/\int E_{\rm total}\,dd$ over that same checkpoint subset. Profiles use 256 Gauss–Hermite nodes per mixture component.
#
# Immediately below the overview, three panels show the auxiliary Gaussian score parameters
#
# $$\mu_\theta(t)=\mu+\frac{V(t)b(t)}{1-V(t)C(t)},\qquad
# \alpha_\theta(t)=\frac{1}{1-V(t)C(t)}.$$
#
# They use the same epoch cutoff, viridis colors, and opacity. Dashed reference lines mark $\mu=-0.01$ and $\alpha=1$. These are parameters of the auxiliary Gaussian score, not terminal sampler moments. Only points with $1-VC>0$ admit a positive Gaussian variance parameter: other points are masked, leaving gaps in the curves. Large values near $1-VC=0$ are retained. The mean panel uses a symmetric-log vertical axis, and the positive variance-scale parameter uses a logarithmic axis.
#
# The third panel plots the signed natural logarithm $a(t)=\log\alpha_\theta(t)$: $a>0$ corresponds to $\alpha_\theta>1$ and $a<0$ to $0<\alpha_\theta<1$. It uses a symmetric-log vertical axis, with a linear region $|a|\leq10^{-3}$ (controlled by `log_alpha_linthresh`), equal limits above and below zero, and the same epoch colors and validity gaps. This displays the signs and the small high-noise values together. The existing $\alpha_\theta$ panel already has a log axis; this extra panel applies symmetric-log scaling to the signed $\log\alpha_\theta$ values themselves.

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

for ax, key, title in zip(overview_axes[:2], ['b', 'C'],
                          ['Mean error b(t)', 'Affine coefficient C(t)']):
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

# Exact Gaussian score parameters corresponding to the projected affine error.
profile_metadata = json.loads((paper / 'outputs/score_error_modes/metadata.json').read_text())
reference_mean = profile_metadata['reference_mean']
with np.load(paper / 'outputs/score_error_modes/clock.npz') as saved:
    forward_variance = saved['variance'].copy()

log_alpha_linthresh = 1e-3  # Linear neighborhood of zero on the signed log-alpha axis.
gaussian_parameter_fig, parameter_axes = plt.subplots(1, 3, figsize=(18, 4.5), layout='constrained')
gaussian_parameter_profiles = {}
invalid_points = 0
invalid_checkpoints = 0
for checkpoint in ordered_ids:
    profile = profiles[checkpoint]
    denominator = 1-forward_variance*profile['C']
    valid = np.isfinite(denominator) & (denominator > 0) & np.isfinite(profile['b'])
    alpha_theta = np.full_like(sigma, np.nan)
    mu_theta = np.full_like(sigma, np.nan)
    alpha_theta[valid] = 1/denominator[valid]
    mu_theta[valid] = (reference_mean + forward_variance[valid]*profile['b'][valid]
                      / denominator[valid])
    gaussian_parameter_profiles[checkpoint] = dict(mu_theta=mu_theta, alpha_theta=alpha_theta, valid=valid)
    invalid_points += int(np.count_nonzero(~valid))
    invalid_checkpoints += int(np.any(~valid))
    color = epoch_cmap(epoch_norm(checkpoint[1]))
    parameter_axes[0].plot(sigma, mu_theta, color=color, alpha=profile_alpha, linewidth=0.9)
    parameter_axes[1].plot(sigma, alpha_theta, color=color, alpha=profile_alpha, linewidth=0.9)
    parameter_axes[2].plot(sigma, np.log(alpha_theta), color=color, alpha=profile_alpha, linewidth=0.9)

parameter_axes[0].set(title=r'Gaussian mean parameter $\mu_\theta(t)$', ylabel=r'$\mu_\theta(t)$')
parameter_axes[0].set_yscale('symlog', linthresh=1e-2)
parameter_axes[0].axhline(reference_mean, color='black', ls='--', lw=0.9,
                         label=rf'Reference $\mu={reference_mean:.2f}$')
parameter_axes[1].set(title=r'Gaussian variance-scale parameter $\alpha_\theta(t)$',
                      ylabel=r'$\alpha_\theta(t)$', yscale='log')
parameter_axes[1].axhline(1, color='black', ls='--', lw=0.9, label=r'Reference $\alpha=1$')
# Equal limits make positive and negative log-alpha magnitudes visually comparable.
log_alpha_limit = max(log_alpha_linthresh, 1.1*max(
    np.nanmax(np.abs(np.log(parameters['alpha_theta'])))
    for parameters in gaussian_parameter_profiles.values()))
parameter_axes[2].set(title=r'Log variance scale $\log\alpha_\theta(t)$',
                      ylabel=r'$\log\alpha_\theta(t)$', ylim=(-log_alpha_limit, log_alpha_limit))
parameter_axes[2].set_yscale('symlog', linthresh=log_alpha_linthresh)
parameter_axes[2].axhline(0, color='black', ls='--', lw=0.9,
                         label=r'Reference $\log\alpha=0$ ($\alpha=1$)')
for ax in parameter_axes:
    ax.set_xscale('log')
    ax.set_xlabel(r'$t=\sigma$')
    ax.legend(fontsize=8)
parameter_colorbar = gaussian_parameter_fig.colorbar(
    plt.cm.ScalarMappable(norm=epoch_norm, cmap=epoch_cmap), ax=parameter_axes,
    label='Training epoch', shrink=0.9, pad=0.02)
parameter_colorbar.set_ticks(epoch_colorbar.get_ticks())
gaussian_parameter_fig.suptitle(
    f'Epoch ≥ {minimum_epoch}: {len(ordered_ids)} checkpoints; positive Gaussian precision only')
show_figure(gaussian_parameter_fig)
print(f'Gaussian parameter mapping: {invalid_points}/{len(ordered_ids)*len(sigma)} grid points '
      f'masked across {invalid_checkpoints}/{len(ordered_ids)} checkpoints (1 − VC ≤ 0 or nonfinite).')

# %% [markdown]
# ## Piecewise power-law magnitude models
#
# Use the **exact draft coordinates** from the preceding parameter plots,
#
# $$u_j(t)=\frac{\mu_{\theta,j}(t)-\mu}{\sqrt{V_{\rm ref}}},\qquad
# a_j(t)=\log\alpha_{\theta,j}(t),\qquad t=\sigma.$$
#
# The magnitudes below are $R_u(t)=\sqrt{N_t^{-1}\sum_j u_j(t)^2}$ and
# $R_a(t)=\sqrt{N_t^{-1}\sum_j a_j(t)^2}$, averaging checkpoints **before fitting**.
# Each selected checkpoint has equal weight; `minimum_epoch` selects the same cohort as above.
# Here the variance-coordinate magnitude is RMS($\log\alpha_\theta$), not RMS($\alpha_\theta$),
# and it is not the variance across checkpoints.
#
# Fit **one, two, and three exponents** to each RMS curve. For the variance coordinate,
# one pooled $R_a(t)$ supplies the **symmetric** signed fits $\pm\widehat R_a(t)$ in
# $\log\alpha_\theta$. Both signs therefore share the same exponents and breakpoints;
# positive and negative checkpoints are not fitted separately.
#
# With $x=\log(t/t_{\min})$ and ordered log-time breakpoints $\tau_j$, the continuous family is
#
# $$\log\widehat R(t)=\log B-p_1x
#   -\sum_{j=1}^{K-1}(p_{j+1}-p_j)(x-\tau_j)_+,\qquad K\in\{1,2,3\},\ p_j\geq0.$$
#
# Thus $\widehat R\propto t^{-p_j}$ on interval $j$, with $p_j=0$ allowing a plateau.
# Each family has $2K$ parameters: the amplitude, $K$ powers, and $K-1$ fitted breakpoints.
# The full saved range is $t=\sigma\in[0.002,80]$. No minimum segment width or smoothing
# is imposed, so a magnitude-space fit can still devote segments to narrow spikes.
#
# **Validity and weighting.** Points with $1-VC\leq0$ admit no positive Gaussian variance
# parameter. The first row uses every valid checkpoint at each time; the second uses
# the fixed cohort valid throughout. Large values near the mapping singularity are retained.
# Gray shading marks incomplete coverage in the first row. Trapezoidal weights in $\log t$,
# normalized to sum to one, give equal weight to equal log-noise intervals on the irregular grid.
#
# **Objective and numerical search.** `piecewise_fit_space = 'log'` selects least squares
# on $\log R$ for proportional agreement; `'magnitude'` selects least squares on $R$.
# Both are computed and compared. For log fits, nonnegative least squares finds the optimal
# powers for each breakpoint choice, followed by a coarse mesh and four seeded searches
# over breakpoints. For magnitude fits, the optimal amplitude is analytic and four searches
# optimize powers and breakpoints, also starting from the log fit. Each larger family
# includes the preceding fit as a candidate, so its objective cannot worsen. These are
# the best numerical fits found; selection minimizes the chosen in-sample objective and
# does not establish a statistical preference for three segments.
#
# **The three columns.** The first two show RMS($u$) and RMS($\log\alpha_\theta$), with
# the one-, two-, and three-power fits. The third shows the signed checkpoint profiles
# $a_j(t)=\log\alpha_{\theta,j}(t)$, observed $\pm R_a(t)$, and fitted
# $\pm\widehat R_a(t)$ for all three families. It displays the same variance-coordinate
# fits symmetrically about zero; it is not a fit to $\log|\log\alpha_\theta|$ profiles.
# The third column uses the same epoch colors and symmetric-log vertical scale as the
# parameter overview, with equal positive and negative limits. Dots mark the selected
# model's breakpoints. The symmetric guides are not confidence intervals or pointwise bounds.
#
# The tables report every family under the selected objective, then compare the best
# fits under both objectives. Relative RMSE is $\sqrt{\sum w(\widehat R-R)^2/\sum wR^2}$;
# log10 RMSE measures proportional error in decades. There are two sets of fitted coordinates,
# even though the figure has three columns. Implementation:
# [rms_power_laws.py](score_error_analysis/rms_power_laws.py).
#
# A useful stochastic model is
#
# $$u_j(t)=m_u(t)+A_u(t)Z_{u,j}(h(t)),\qquad
# a_j(t)=m_a(t)+A_a(t)Z_{a,j}(h(t)),\qquad
# \alpha_{\theta,j}(t)=e^{a_j(t)},$$
#
# with zero-mean, unit-marginal-variance GPs $Z$. A deterministic envelope gives the valid
# covariance $A(t)A(s)k(h(t),h(s))$; see [Rasmussen & Williams, Chapter 4, §4.2.4](https://gaussianprocess.org/gpml/chapters/RW4.pdf).
# Modeling $a=\log\alpha$ also guarantees positive variance scale. The time coordinate
# $h(t)$ and covariance must still be estimated or posited: RMS alone does not determine
# sign persistence or cancellations. Furthermore $R_f(t)^2=m_f(t)^2+A_f(t)^2$;
# fitting RMS estimates the stochastic amplitude only when the ensemble mean vanishes.
# `mean_square_fraction` in the table is $\sum w m_f^2/\sum wR_f^2$ and checks that assumption.
# The checkpoints come from three training runs with repeated epochs, so they are not independent GP realizations.
#
# This construction is useful for theory: for a linear Gaussian response
# $Y_\gamma=\int K_\gamma(d)f(d)\,\mathrm d d$,
#
# $$\mathbb E[Y_\gamma^2]=\left(\int K_\gamma(d)m_f(d)\,\mathrm d d\right)^2
# +\iint K_\gamma(d)K_\gamma(e)A_f(d)A_f(e)k(h(d),h(e))\,\mathrm d d\,\mathrm d e.$$
#
# Applying this to the two first-order kernels below gives
# $\mathbb E[\widehat h_\gamma]=\mathbb E[Y_{a,\gamma}^2]/4+\mathbb E[Y_{u,\gamma}^2]/2$.
# It remains a small-error Gaussian-surrogate calculation; a ratio of expected KLs is
# not the expected log KL ratio. These envelope fits do not change the direct predictions below.

# %%
import sys
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from paper_odds.score_error_analysis.rms_power_laws import (
    fit_piecewise_power_laws, evaluate_power_law,
)

# Rows are the same selected checkpoints as in the parameter overview.
rms_u = np.stack([(gaussian_parameter_profiles[cp]['mu_theta']-reference_mean)
                  / np.sqrt(profile_metadata['reference_variance']) for cp in ordered_ids])
rms_a = np.stack([np.log(gaussian_parameter_profiles[cp]['alpha_theta']) for cp in ordered_ids])
rms_valid = np.isfinite(rms_u) & np.isfinite(rms_a)
rms_u = np.where(rms_valid, rms_u, np.nan)
rms_a = np.where(rms_valid, rms_a, np.nan)
rms_counts = rms_valid.sum(axis=0)
rms_complete = rms_valid.all(axis=1)
if not np.all(rms_counts):
    raise ValueError('Some noise levels have no valid Gaussian mappings for this epoch cutoff.')

# Quadrature in log sigma avoids overweighting the denser part of the saved grid.
rms_log_time = np.log(sigma)
rms_weights = np.empty_like(sigma)
rms_weights[0] = (rms_log_time[1]-rms_log_time[0])/2
rms_weights[-1] = (rms_log_time[-1]-rms_log_time[-2])/2
rms_weights[1:-1] = (rms_log_time[2:]-rms_log_time[:-2])/2
rms_weights /= rms_weights.sum()

rms_cohorts = [('Valid at each t', np.ones(len(ordered_ids), dtype=bool), 'tab:orange')]
if rms_complete.any():
    rms_cohorts.append(('Valid throughout', rms_complete, 'tab:blue'))
else:
    print('No checkpoint is valid throughout the range; fixed-cohort sensitivity unavailable.')

rms_profiles = {}
for coordinate_name, values in [('u', rms_u), ('log alpha', rms_a)]:
    for cohort, members, _ in rms_cohorts:
        selected = values[members]
        rms_curve = np.sqrt(np.nanmean(selected**2, axis=0))
        mean_curve = np.nanmean(selected, axis=0)
        rms_profiles[(coordinate_name, cohort)] = dict(
            rms=rms_curve, mean=mean_curve, counts=np.isfinite(selected).sum(axis=0),
            mean_square_fraction=np.sum(rms_weights*mean_curve**2)/np.sum(rms_weights*rms_curve**2))

piecewise_fit_space = 'log'  # 'log' for proportional agreement; 'magnitude' for absolute RMS agreement.
piecewise_max_exponents = 3
if piecewise_fit_space not in ('log', 'magnitude'):
    raise ValueError("piecewise_fit_space must be 'log' or 'magnitude'.")
piecewise_fits, piecewise_best = {}, {}
for coordinate_name in ['u', 'log alpha']:
    for cohort, _, _ in rms_cohorts:
        observed_rms = rms_profiles[(coordinate_name, cohort)]['rms']
        log_fits = fit_piecewise_power_laws(
            sigma, observed_rms, rms_weights, max_segments=piecewise_max_exponents,
            fit_space='log', seed=random_seed)
        magnitude_fits = fit_piecewise_power_laws(
            sigma, observed_rms, rms_weights, max_segments=piecewise_max_exponents,
            fit_space='magnitude', seed=random_seed, initial_fits=log_fits)
        for objective, fits in [('log', log_fits), ('magnitude', magnitude_fits)]:
            piecewise_fits[(objective, coordinate_name, cohort)] = fits
            piecewise_best[(objective, coordinate_name, cohort)] = min(
                fits, key=lambda f: (f['objective'], f['n_exponents']))
            for fit in fits:
                if not fit['search_converged'] or fit['rate_bound_hit']:
                    print(f'Check optimizer: {objective}, {coordinate_name}, {cohort}, '
                          f'{fit["n_exponents"]} exponents; converged={fit["search_converged"]}, '
                          f'rate bound reached={fit["rate_bound_hit"]}.')

piecewise_colors = {1: '#208f8d', 2: '#d55e00', 3: '#542788'}
piecewise_styles = {1: ':', 2: '--', 3: '-.'}
piecewise_fig, piecewise_axes = plt.subplots(
    len(rms_cohorts), 3, figsize=(19, 4.4*len(rms_cohorts)), squeeze=False, layout='constrained')
piecewise_rows = []
for row_index, (cohort, members, _) in enumerate(rms_cohorts):
    for col, coordinate_name in enumerate(['u', 'log alpha']):
        ax = piecewise_axes[row_index, col]
        observed = rms_profiles[(coordinate_name, cohort)]
        ax.plot(sigma, observed['rms'], color='black', lw=1.5, alpha=0.8, label='Observed RMS')
        fits = piecewise_fits[(piecewise_fit_space, coordinate_name, cohort)]
        best = piecewise_best[(piecewise_fit_space, coordinate_name, cohort)]
        for fit in fits:
            n = fit['n_exponents']
            fit_label = (f'{n} exponent'+('s' if n > 1 else '')+
                         f'; log10 RMSE={fit["log10_RMSE"]:.3f}')
            ax.plot(sigma, fit['prediction'], color=piecewise_colors[n],
                    ls=piecewise_styles[n], lw=1.9, label=fit_label)
            piecewise_rows.append(dict(
                coordinate=coordinate_name, cohort=cohort, exponents=n,
                B_at_tmin=fit['amplitude'], p_in_time_order=', '.join(f'{p:.4g}' for p in fit['exponents']),
                breaks_t=', '.join(f'{t:.5g}' for t in fit['breakpoints']) or '—',
                relative_RMSE=fit['relative_RMSE'], log10_RMSE=fit['log10_RMSE'], R2=fit['R2'],
                mean_square_fraction=observed['mean_square_fraction']))
        if len(best['breakpoints']):
            ax.scatter(best['breakpoints'], evaluate_power_law(best['breakpoints'], best),
                       color=piecewise_colors[best['n_exponents']], s=30, zorder=5)
        if cohort == 'Valid at each t' and np.any(rms_counts < len(ordered_ids)):
            ax.fill_between(sigma, 0, 1, where=rms_counts < len(ordered_ids),
                            transform=ax.get_xaxis_transform(), color='gray', alpha=0.12)
        n_min, n_max = observed['counts'].min(), observed['counts'].max()
        count_label = str(n_min) if n_min == n_max else f'{n_min}–{n_max}'
        title = r'RMS of mean coordinate $u$' if col == 0 else r'RMS of $\log\alpha_\theta$'
        ax.set(title=f'{title}\n{cohort}: n={count_label}',
               xscale='log', yscale='log', xlabel=r'$t=\sigma$', ylabel='Ensemble RMS',
               xlim=(sigma[0], sigma[-1]),
               ylim=(observed['rms'].min()/2, max(observed['rms'].max(),
                     max(f['prediction'].max() for f in fits))*1.5))
        ax.legend(fontsize=8, loc='upper right')

    # Signed view of the same variance-coordinate RMS and fitted families.
    signed_ax = piecewise_axes[row_index, 2]
    selected_ids = [checkpoint for checkpoint, keep in zip(ordered_ids, members) if keep]
    for checkpoint, values in zip(selected_ids, rms_a[members]):
        signed_ax.plot(sigma, values, color=epoch_cmap(epoch_norm(checkpoint[1])),
                       alpha=0.18, lw=0.7, zorder=1)
    observed_a = rms_profiles[('log alpha', cohort)]
    shape_fits = piecewise_fits[(piecewise_fit_space, 'log alpha', cohort)]
    best_a = piecewise_best[(piecewise_fit_space, 'log alpha', cohort)]
    for sign in [1, -1]:
        signed_ax.plot(sigma, sign*observed_a['rms'], color='black', lw=1.4, alpha=0.8,
                       label='Observed ±RMS' if sign == 1 else None)
        for fit in shape_fits:
            n = fit['n_exponents']
            signed_ax.plot(sigma, sign*fit['prediction'], color=piecewise_colors[n],
                           ls=piecewise_styles[n], lw=1.9,
                           label=(f'± fit: {n} exponent'+('s' if n > 1 else '') if sign == 1 else None))
        if len(best_a['breakpoints']):
            signed_ax.scatter(best_a['breakpoints'], sign*evaluate_power_law(best_a['breakpoints'], best_a),
                              color=piecewise_colors[best_a['n_exponents']], s=30, zorder=5)
    signed_ax.axhline(0, color='gray', lw=0.8)
    if cohort == 'Valid at each t' and np.any(rms_counts < len(ordered_ids)):
        signed_ax.fill_between(sigma, 0, 1, where=rms_counts < len(ordered_ids),
                               transform=signed_ax.get_xaxis_transform(), color='gray', alpha=0.12)
    signed_limit = 1.1*max(np.nanmax(np.abs(rms_a[members])),
                           max(f['prediction'].max() for f in shape_fits), log_alpha_linthresh)
    n_min, n_max = observed_a['counts'].min(), observed_a['counts'].max()
    count_label = str(n_min) if n_min == n_max else f'{n_min}–{n_max}'
    signed_ax.set(title=rf'Symmetric fit to $\log\alpha_\theta$'+'\n'+f'{cohort}: n={count_label}',
                  xscale='log', xlabel=r'$t=\sigma$', ylabel=r'$\log\alpha_\theta(t)$',
                  xlim=(sigma[0], sigma[-1]), ylim=(-signed_limit, signed_limit))
    signed_ax.set_yscale('symlog', linthresh=log_alpha_linthresh)
    signed_ax.legend(fontsize=8, loc='upper right')
piecewise_profile_colorbar = piecewise_fig.colorbar(
    plt.cm.ScalarMappable(norm=epoch_norm, cmap=epoch_cmap), ax=piecewise_axes[:, 2],
    label='Checkpoint epoch', shrink=0.9, pad=0.02)
piecewise_profile_colorbar.set_ticks(epoch_colorbar.get_ticks())
piecewise_fig.suptitle(
    f'Epoch ≥ {minimum_epoch}: continuous power laws; least squares in '
    + ('log magnitude' if piecewise_fit_space == 'log' else 'magnitude'))
show_figure(piecewise_fig)
show_table(piecewise_rows)
display(HTML('<p><b>Best fit with at most three exponents, under each objective:</b></p>'))
piecewise_objective_rows = []
for coordinate_name in ['u', 'log alpha']:
    for cohort, _, _ in rms_cohorts:
        for objective in ['log', 'magnitude']:
            fit = piecewise_best[(objective, coordinate_name, cohort)]
            piecewise_objective_rows.append(dict(
                coordinate=coordinate_name, cohort=cohort, objective=objective,
                exponents=fit['n_exponents'], relative_RMSE=fit['relative_RMSE'],
                log10_RMSE=fit['log10_RMSE'], R2=fit['R2']))
show_table(piecewise_objective_rows)
print('Every fit uses t = sigma from', sigma[0], 'to', sigma[-1],
      '; fitted breakpoints and exponents are listed in increasing t order.')

print(f'Exact mapping coverage: {rms_counts.min()}–{rms_counts.max()}/{len(ordered_ids)} '
      f'checkpoints per t; {rms_complete.sum()} valid throughout.')

# %% [markdown]
# ### Gaussian parameter profiles with fitted RMS magnitudes
#
# Repeat the earlier $\mu_\theta,\alpha_\theta,\log\alpha_\theta$ plot with the same checkpoint curves,
# epoch colors, opacity, and validity gaps. Convert the selected fitted magnitudes
# back to the plotted coordinates as
#
# $$\mu\pm\sqrt{V_{\rm ref}}\,\widehat R_u(t),\qquad
#   \exp[\pm\widehat R_a(t)],\qquad \pm\widehat R_a(t).$$
#
# Black dashed guides use the conditional cohort; magenta dotted guides use the cohort
# valid throughout. These are **RMS guides about the exact-reference parameters**, not
# fitted signed means, confidence intervals, pointwise bounds, or GP credible bands.
# In particular $e^{\pm R_a}$ is a multiplicative guide in $\alpha$ because the fitted
# coordinate is $\log\alpha$. The plotted mean parameter uses a symmetric-log axis;
# the variance scale uses a log axis. The third panel shows signed $\log\alpha_\theta$ on the same symmetric-log scale as the preceding parameter overview, with equal positive and negative limits. Its $\pm\widehat R_a$ guides still come from the pooled RMS; they impose symmetry on the guides and do not establish symmetry of the checkpoint distribution. The original profiles and all sampler predictions
# remain unchanged.

# %%
piecewise_parameter_fig, piecewise_parameter_axes = plt.subplots(
    1, 3, figsize=(19, 4.8), layout='constrained')
for checkpoint in ordered_ids:
    parameters = gaussian_parameter_profiles[checkpoint]
    color = epoch_cmap(epoch_norm(checkpoint[1]))
    piecewise_parameter_axes[0].plot(sigma, parameters['mu_theta'], color=color,
                                     alpha=profile_alpha, lw=0.9)
    piecewise_parameter_axes[1].plot(sigma, parameters['alpha_theta'], color=color,
                                     alpha=profile_alpha, lw=0.9)
    piecewise_parameter_axes[2].plot(sigma, np.log(parameters['alpha_theta']), color=color,
                                     alpha=profile_alpha, lw=0.9)
for cohort, _, _ in rms_cohorts:
    color, style = ('black', '--') if cohort == 'Valid at each t' else ('#d81b60', ':')
    mean_fit = piecewise_best[(piecewise_fit_space, 'u', cohort)]
    shape_fit = piecewise_best[(piecewise_fit_space, 'log alpha', cohort)]
    mean_magnitude = np.sqrt(profile_metadata['reference_variance'])*mean_fit['prediction']
    shape_magnitude = shape_fit['prediction']
    for sign in [1, -1]:
        piecewise_parameter_axes[0].plot(
            sigma, reference_mean+sign*mean_magnitude, color=color, ls=style, lw=2.2,
            label=(f'± fitted RMS: {cohort} ({mean_fit["n_exponents"]} powers)' if sign == 1 else None))
        piecewise_parameter_axes[1].plot(
            sigma, np.exp(sign*shape_magnitude), color=color, ls=style, lw=2.2,
            label=(f'exp(± fitted RMS): {cohort} ({shape_fit["n_exponents"]} powers)' if sign == 1 else None))
        piecewise_parameter_axes[2].plot(
            sigma, sign*shape_magnitude, color=color, ls=style, lw=2.2,
            label=(f'± fitted RMS: {cohort} ({shape_fit["n_exponents"]} powers)' if sign == 1 else None))
piecewise_parameter_axes[0].set(title=r'Gaussian mean parameter $\mu_\theta(t)$', ylabel=r'$\mu_\theta(t)$')
piecewise_parameter_axes[0].set_yscale('symlog', linthresh=1e-2)
piecewise_parameter_axes[0].axhline(reference_mean, color='gray', ls='--', lw=0.9,
                                   label=rf'Reference $\mu={reference_mean:.2f}$')
piecewise_parameter_axes[1].set(title=r'Gaussian variance scale $\alpha_\theta(t)$',
                               ylabel=r'$\alpha_\theta(t)$', yscale='log')
piecewise_parameter_axes[1].axhline(1, color='gray', ls='--', lw=0.9, label=r'Reference $\alpha=1$')
piecewise_log_alpha_limit = max(log_alpha_limit, 1.1*max(
    piecewise_best[(piecewise_fit_space, 'log alpha', cohort)]['prediction'].max()
    for cohort, _, _ in rms_cohorts))
piecewise_parameter_axes[2].set(title=r'Log variance scale $\log\alpha_\theta(t)$',
                               ylabel=r'$\log\alpha_\theta(t)$',
                               ylim=(-piecewise_log_alpha_limit, piecewise_log_alpha_limit))
piecewise_parameter_axes[2].set_yscale('symlog', linthresh=log_alpha_linthresh)
piecewise_parameter_axes[2].axhline(0, color='gray', ls='--', lw=0.9,
                                   label=r'Reference $\log\alpha=0$ ($\alpha=1$)')
for ax in piecewise_parameter_axes:
    ax.set_xscale('log')
    ax.set_xlabel(r'$t=\sigma$')
    ax.legend(fontsize=7.5, loc='upper right')
piecewise_epoch_colorbar = piecewise_parameter_fig.colorbar(
    plt.cm.ScalarMappable(norm=epoch_norm, cmap=epoch_cmap), ax=piecewise_parameter_axes,
    label='Training epoch', shrink=0.9, pad=0.02)
piecewise_epoch_colorbar.set_ticks(epoch_colorbar.get_ticks())
piecewise_parameter_fig.suptitle(
    f'Epoch ≥ {minimum_epoch}: {len(ordered_ids)} checkpoints with fitted RMS guides\n'
    + ('Log-magnitude' if piecewise_fit_space == 'log' else 'Magnitude')
    + ' least squares; best numerical fit with at most three exponents')
show_figure(piecewise_parameter_fig)

# %% [markdown]
# ## Direct Gaussian predictions
#
# Here **direct** means applying the draft's small-error response kernels to the measured score-error profiles. We work directly with $b(t)$ and $C(t)$, without first reconstructing $\mu_\theta(t)$ and $\alpha_\theta(t)$ and solving their full moment dynamics. The connection to those parameters is algebraic, as follows. Equation numbers below refer to [the draft](The_odds_favor_noise_in_generative_diffusion_sampling.pdf).
#
# **From the network to spatial coefficients.** At each positive noise level $t=\sigma$, evaluate $\epsilon_\theta=s_\theta-s_{\rm exact}$ under the actual diffused mixture $p_t$. The one-dimensional least-squares projection gives
#
# $$b(t)=\mathbb E_{p_t}[\epsilon_\theta(X,t)],\qquad
# C(t)=\frac{\mathbb E_{p_t}[(X-\mu)\epsilon_\theta(X,t)]}{V(t)},$$
#
# where $\mu=-0.01$ and $V(t)=0.1219+t^2$. Expectations use 256 Gauss–Hermite nodes per mixture component. Thus $b$ and $C$ come from the trained network's error under the mixture, not from measured terminal sampler means or variances.
#
# **Connection to $\mu_\theta$ and $\alpha_\theta$.** Replace the reference mixture by a Gaussian with the same mean and variance and add the projected error. This gives the auxiliary affine score
#
# $$s_{\rm G}(x,t)=-\frac{x-\mu}{V(t)}+b(t)+C(t)(x-\mu).
# $$
#
# Matching this to the draft's Eq. (5), $s_{\rm G}=-(x-\mu_\theta)/(\alpha_\theta V)$, gives the **exact algebraic correspondence for this auxiliary score**:
#
# $$\alpha_\theta(t)=\frac{1}{1-V(t)C(t)},\qquad
# \mu_\theta(t)=\mu+\frac{V(t)b(t)}{1-V(t)C(t)},$$
#
# provided $1-VC>0$. This is Eq. (7) read in reverse. These are Gaussian-surrogate score parameters; they are not the actual mean and variance of the neural sampler. If $1-VC\leq0$, no positive $\alpha_\theta$ represents this affine score.
#
# The direct prediction uses the **first-order score coordinates** stored as `a_linear` and `u_linear`:
#
# $$a_{\rm lin}(t)=V(t)C(t),\qquad
# u_{\rm lin}(t)=\frac{V(t)b(t)}{\sqrt{V_{\rm ref}}},\qquad
# V_{\rm ref}=V(0.002)=0.121904.$$
#
# In terms of the paper's parameters,
#
# $$\log\alpha_\theta=-\log(1-a_{\rm lin})\simeq a_{\rm lin},\qquad
# u_{\rm paper}:=\frac{\mu_\theta-\mu}{\sqrt{V_{\rm ref}}}
# =\frac{u_{\rm lin}}{1-a_{\rm lin}}\simeq u_{\rm lin}.$$
#
# The last approximations hold to first order in small score error. The direct calculation uses $a_{\rm lin}$ and $u_{\rm lin}$ themselves; it does not apply the nonlinear parameter conversion above. The reference standard deviation is $\sqrt{V_{\rm ref}}$ because this surrogate calculation ends at the last positive noise level, $t=0.002$.
#
# **Propagate the two signed profiles.** Write
#
# $$d(t)=\log\frac{V(t)}{V_{\rm ref}},\qquad
# \Lambda=\log\frac{V(80)}{V_{\rm ref}},\qquad \ell=\Lambda-d.$$
#
# Here $d=0$ is the low-noise endpoint and $d=\Lambda$ is the start of sampling. With a matched Gaussian prior, the initial standardized mean error is zero and the initial relative variance is one. The terminal first-order relative-variance error and standardized mean error are, from Eqs. (26), (70), and (82),
#
# $$\Delta v_\gamma^{(1)}=(1+\gamma)\int_0^\Lambda
#  e^{-\gamma d}\,a_{\rm lin}(d)\,\mathrm d d,$$
#
# $$r_\gamma^{(1)}=\frac{1+\gamma}{2}\int_0^\Lambda
#  e^{-(1+\gamma)d/2}\,u_{\rm lin}(d)\,\mathrm d d.$$
#
# `profile_response` evaluates these by trapezoidal quadrature on the saved noise/variance grid. For this section that grid has 288 points. The profiles retain their signs inside the integrals, so errors at different noise levels can cancel. Squaring is performed only after integration.
#
# **Turn the response into the plotted KL ratio.** Expanding the Gaussian KL to second order gives Eq. (83):
#
# $$\widehat h_\gamma=
# \frac{(\Delta v_\gamma^{(1)})^2}{4}+
# \frac{(r_\gamma^{(1)})^2}{2}.$$
#
# Both KL directions have this same leading-order expression, with no second-order mean–shape cross term. The prediction shown here is
#
# $$\widehat L_\gamma=
# \log_{10}\!\left(\frac{\widehat h_\gamma}{\widehat h_{\gamma_0}}\right),
# \qquad \gamma_0=\text{smallest recorded gamma}\simeq0.01.$$
#
# Measured ratios use that same recorded baseline, rather than an unobserved ODE KL. Negative values indicate a smaller KL than at the baseline.
#
# **Relation to the later “Gaussian moments” curve.** That curve retains the full Gaussian moment dynamics, rewritten directly in the same score coordinates:
#
# $$\frac{\mathrm d r_\gamma}{\mathrm d\ell}
# =-\frac{1+\gamma}{2}\big[(1-a_{\rm lin})r_\gamma-u_{\rm lin}\big],\qquad
# \frac{\mathrm d v_\gamma}{\mathrm d\ell}
# =\big[-\gamma+(1+\gamma)a_{\rm lin}\big]v_\gamma+\gamma.$$
#
# Here $r_\gamma=(\mathbb E[Y]-\mu)/\sqrt{V_{\rm ref}}$ and $v_\gamma=\operatorname{Var}(Y)/V(t)$. These are the draft's Eqs. (12)–(13) expressed using $b,C$. The later calculation advances them with 500 explicit Euler steps and evaluates the full Gaussian KL formulas, Eqs. (20)–(21), which can differ between directions. It retains the products involving $a_{\rm lin}$ and the evolving moments that the direct first-order response drops.
#
# Both calculations use a **Gaussian surrogate for a mixture**. The direct curve assumes small score error and a matched prior, and omits the nonlinear residual. It does not correct the original sweeps' prior mismatch, discretization, endpoint difference, or KL-estimation error. Moreover, removing the residual in the controlled experiment leaves the *exact mixture score* plus affine error; that score is still generally nonlinear, so even the affine-only sampler need not follow Gaussian dynamics.
#
# Implementation: `project_grid` in [fit_checkpoint_modes.py](score_error_analysis/fit_checkpoint_modes.py) computes the projection; `normalized_profiles` and `profile_response` in [score_error_modes.py](score_error_analysis/score_error_modes.py) perform the normalization and response quadrature. `gaussian_moment_kl` in [residual_ablation.py](score_error_analysis/residual_ablation.py) implements the separate moment calculation.
#
# The first figure shows the gamma sweeps of the same three random checkpoints, using their colors from above. The second figure compares predictions with measurements for all 144 checkpoints in a two-row, three-column grid: rows are the two KL directions, and columns target gamma = 0.2, 1, and 5. Each column uses the nearest saved gamma in log space, stated in its titles. The identity line represents quantitative agreement. All three selected checkpoints are highlighted by colored stars; RMSE and Spearman values describe all 144 checkpoints at that column's saved gamma.

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
# Compare
#
# $$s_\lambda=s_{\rm exact}+b+C(x-\mu)+\lambda r,$$
#
# with $\lambda=0$ (affine error only) and $\lambda=1$ (full learned score). The cached experiment covers every checkpoint at $\gamma\in\{0,0.01,0.2,1,5\}$, using an exact mixture prior, matched initial states and Brownian increments, four seeds with 4,096 particles each, and **500 Euler steps for $\gamma=0$ or Euler–Maruyama steps for $\gamma>0$** from $\sigma=80$ to $0.002$. These updates retain the standardized log-variance clock $\ell=\log(V_{\max}/V)$ and the positive endpoint of the controlled experiment.
#
# Each arm now has a **measured ODE baseline ($\gamma=0$)**. KL uses pooled counts in target-quantile bins, analytic target probabilities, and a 0.5 pseudocount. An exact-score control indicates the numerical/sample floor. Full Gaussian moment dynamics provide an additional prediction beyond the small-error response, still under the Gaussian surrogate. Their moment equations use explicit Euler on the same 500 clock intervals.
#
# The first figure has **three rows, one for each of the same random checkpoints used above**, and two columns for the KL directions. Each panel compares the full-score and affine-only log ratios with the Gaussian profile and moment predictions. Checkpoint colors are preserved; circles with solid lines indicate the full score and squares with dash-dot lines indicate affine error only.
#
# **The second figure compares prediction accuracy across all 144 checkpoints**, at the single `evaluation_gamma` (default 1), with one panel per KL direction. For each checkpoint, let
#
# $$L_\lambda=\log_{10}\!\left[\frac{D_{KL}^{(\lambda)}(\gamma)}{D_{KL}^{(\lambda)}(0)}\right],$$
#
# and let $L_G$ be its direct Gaussian profile prediction. Each dot has coordinates
#
# $$x=|L_1-L_G| \quad\text{(full score)},\qquad
# y=|L_0-L_G| \quad\text{(affine error only)}.$$
#
# Both axes measure absolute prediction error in dex. Below the diagonal, removing the residual improves agreement; above it, agreement worsens. Near the origin, both arms agree well with the Gaussian profile prediction. This compares accuracy of the **stochastic/ODE KL ratio**, not absolute sampling quality: each arm uses its own ODE denominator. The absolute values also discard whether the prediction over- or underestimates the ratio. Pale points have a numerator or ODE KL at most three times the corresponding exact-score control in at least one arm, where ratios are less well resolved. This floor diagnostic is not a significance test.

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

# %% [markdown]
# ## Agreement, alongside correlation
#
# For each controlled comparison, let $y=\log_{10}(D_{KL}(\gamma)/D_{KL}(0))$ and $\hat y$ be its prediction. **Log-RMSE** measures distance from equality in dex; **ratio-RMSE** applies the same calculation to $10^y$. Log-bias is the average $\hat y-y$. **Skill** is $1-\mathrm{MSE}(\hat y,y)/\mathrm{MSE}(0,y)$: positive values outperform the constant ratio-one prediction, while negative values do worse. Pearson and Spearman describe association; neither establishes agreement. Constant predictions have undefined correlations.
#
# The table retains both KL directions and both residual arms at $\gamma=0.2,1,5$. Metrics use the predictions as recorded, with no calibration. Set `above_floor_only=True` to restrict each comparison to checkpoints whose numerator **and** ODE KL exceed three times the corresponding exact-score control. This is a diagnostic threshold, not a significance test; `n` records the finite pairs actually evaluated.

# %%
agreement_rows = []
for gamma in [0.2, 1.0, 5.0]:
    for direction in directions:
        for residual_lambda, arm in [(1, 'Full'), (0, 'Affine')]:
            rows = ablation[np.isclose(ablation.gamma, gamma) & (ablation.bins == primary_bins) &
                            (ablation.residual_lambda == residual_lambda)]
            if above_floor_only:
                rows = rows[rows[f'above_control_floor_{direction}'].astype(bool)]
            measured = rows[f'log10_ratio_ode_{direction}']
            predictions = {
                'Gaussian profile response': rows[f'gaussian_profile_log10_ratio_ode_{direction}'],
                'Gaussian moments': rows[f'gaussian_moment_log10_ratio_ode_{direction}'],
                'Ratio = 1': np.zeros(len(rows)),
            }
            for name, predicted in predictions.items():
                agreement_rows.append(dict(gamma=gamma, direction=direction, arm=arm,
                    prediction=name, **agreement(predicted, measured)))
print(f'{primary_bins} bins; ' + ('above-control-floor subset' if above_floor_only else 'all checkpoints'))
show_table(agreement_rows)

# %% [markdown]
# The controlled comparison asks whether removing the residual improves the Gaussian prediction; integrated residual energy asks whether a single scalar summarizes that effect. A weak energy/effect correlation does not establish that the residual is unimportant. Remaining disagreement in the affine arm can reflect non-Gaussian dynamics and numerical or KL-estimation effects; this experiment does not uniquely separate them.
#
# Read agreement metrics together with absolute KL and the exact-score control, and check sensitivity with `primary_bins=32` or `128`. The Gaussian formulas are approximations for this mixture. The 144 checkpoints belong to **three related training trajectories**, and pooling four sampling seeds does not make them independent training replicates.

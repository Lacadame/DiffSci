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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

# %% [markdown]
# ## Histogram of improvement $I$ across checkpoints

# %%
# load all_entropies_saved
path_list = [
    '/home/ubuntu/repos/DiffSci/stochasticity_paper/stats/output_default3',
    '/home/ubuntu/repos/DiffSci/stochasticity_paper/stats/output_default4',
    '/home/ubuntu/repos/DiffSci/stochasticity_paper/stats/output_default5',
]
all_entropies_saved = []
for path in path_list:
    all_entropies_saved.append(torch.load(f'{path}/all_entropies.pt'))


def iter_saved_items(saved_list):
    """Yield (key, value) from each dict in saved_list, preserving keys."""
    for saved_dict in saved_list:
        for key, value in sorted(saved_dict.items()):
            yield key, value


def compute_improvements(all_entropies_saved):
    # all_entropies_saved: list[dict[epoch -> (gamma_values, sde_entropies, inv_sde_entropies)]]
    # I = H_0 / min H
    improvements, epochs = [], []
    for epoch, g_ent in iter_saved_items(all_entropies_saved):
        _, sde_entropies, inv_sde_entropies = g_ent
        improvement = []
        for ent in [sde_entropies, inv_sde_entropies]:
            improvement.append(ent[0] / min(ent[3:]))
            # improvement.append(ent[0] / ent[10])          # corresponds to \gamma \approx 1
        improvements.append(improvement)
        epochs.append(epoch)
    return epochs, np.array(improvements)


epochs, improvements = compute_improvements(all_entropies_saved)
print(f'n checkpoints: {len(epochs)}')
print(f'I shape: {improvements.shape}')

# %%
# all_entropies_saved

# %%
# histogram of I across all checkpoints
font_size = 18
n_bins = 20
eps = 0.01
kl_names = [r'$H(\~p|p)$', r'$H(p|\~p)$']

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
p_lt_1 = []
for j, kl_name in enumerate(kl_names):
    ax = axs[j]
    I_vals = np.asarray(improvements[:, j], dtype=float)
    p = np.mean(I_vals > 1 + eps)
    p_lt_1.append(p)
    ax.hist(I_vals, bins=n_bins, color='tab:blue', edgecolor='k', alpha=0.8)
    ax.set_xlabel(r'$I={H_0}/{\min H}$', fontsize=font_size)
    ax.set_ylabel('count', fontsize=font_size)
    ax.set_title(kl_name, fontsize=font_size + 4)
    ax.tick_params(labelsize=font_size - 4)
    ax.text(
        0.5, -0.22, rf'$P(I>{1 + eps})$={p:.3f}',
        transform=ax.transAxes, ha='center', va='top', fontsize=font_size,
    )

fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
plt.show()

for kl_name, p in zip(kl_names, p_lt_1):
    print(f'{kl_name}: P(I<1) = {p:.3f}')

# %%
# scatter: I vs Hmin
from matplotlib import colors as mcolors
from scipy.stats import pearsonr, spearmanr, linregress

font_size = 18
axis_pad = 0.05
stats_text_xy = (0.03, 0.97)
kl_names = [r'$H(\~p|p)$', r'$H(p|\~p)$']

hmins = []
for _, g_ent in iter_saved_items(all_entropies_saved):
    _, sde_entropies, inv_sde_entropies = g_ent
    hmins.append([min(sde_entropies), min(inv_sde_entropies)])
hmins = np.asarray(hmins, dtype=float)

epochs_array = np.asarray(epochs)
epoch_norm = mcolors.Normalize(vmin=epochs_array.min(), vmax=epochs_array.max())
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(epochs_array)))

fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
for j, kl_name in enumerate(kl_names):
    ax = axs[j]
    x = hmins[:, j]
    y = np.asarray(improvements[:, j], dtype=float)
    ax.scatter(x, y, c=colors, marker='o', s=35, zorder=3, alpha=0.8)

    fit = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = fit.slope * x_line + fit.intercept
    ax.plot(x_line, y_line, color='k', linestyle='-', alpha=0.8, zorder=2)

    pearson_r, _ = pearsonr(x, y)
    spearman_r, _ = spearmanr(x, y)
    ax.text(
        stats_text_xy[0], stats_text_xy[1],
        r'Pearson $r$' + f'={pearson_r:+.2f}\n'
        + r'Spearman $\rho$' + f'={spearman_r:+.2f}',
        transform=ax.transAxes, fontsize=font_size, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )
    ax.set_title(kl_name, fontsize=font_size + 4)
    ax.set_xlabel(r'$H_{\mathrm{min}}$', fontsize=font_size)
    ax.set_ylabel(r'$I$', fontsize=font_size)
    ax.tick_params(labelsize=font_size - 4)

    x_vals = list(x) + list(x_line)
    y_vals = list(y) + list(y_line)
    x_lo, x_hi = min(x_vals), max(x_vals)
    y_lo, y_hi = min(y_vals), max(y_vals)
    x_margin = axis_pad * (x_hi - x_lo) if x_hi > x_lo else axis_pad
    y_margin = axis_pad * (y_hi - y_lo) if y_hi > y_lo else axis_pad
    ax.set_xlim(x_lo - x_margin, x_hi + x_margin)
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

fig.tight_layout()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=epoch_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.02)
cbar.set_ticks([epochs_array.min(), epochs_array.max()])
cbar.set_ticklabels([str(epochs_array.min()), str(epochs_array.max())])
cbar.set_label('epoch', fontsize=font_size)

# %%
# scatter: I vs H_0
from matplotlib import colors as mcolors
from scipy.stats import pearsonr, spearmanr, linregress

font_size = 18
axis_pad = 0.05
stats_text_xy = (0.03, 0.97)
kl_names = [r'$H(\~p|p)$', r'$H(p|\~p)$']

h0s = []
for _, g_ent in iter_saved_items(all_entropies_saved):
    _, sde_entropies, inv_sde_entropies = g_ent
    h0s.append([sde_entropies[0], inv_sde_entropies[0]])
h0s = np.asarray(h0s, dtype=float)

epochs_array = np.asarray(epochs)
epoch_norm = mcolors.Normalize(vmin=epochs_array.min(), vmax=epochs_array.max())
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(epochs_array)))

fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
for j, kl_name in enumerate(kl_names):
    ax = axs[j]
    x = h0s[:, j]
    y = np.asarray(improvements[:, j], dtype=float)
    ax.scatter(x, y, c=colors, marker='o', s=35, zorder=3, alpha=0.8)

    fit = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = fit.slope * x_line + fit.intercept
    ax.plot(x_line, y_line, color='k', linestyle='-', alpha=0.8, zorder=2)

    pearson_r, _ = pearsonr(x, y)
    spearman_r, _ = spearmanr(x, y)
    ax.text(
        stats_text_xy[0], stats_text_xy[1],
        r'Pearson $r$' + f'={pearson_r:+.2f}\n'
        + r'Spearman $\rho$' + f'={spearman_r:+.2f}',
        transform=ax.transAxes, fontsize=font_size, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )
    ax.set_title(kl_name, fontsize=font_size + 4)
    ax.set_xlabel(r'$H_0$', fontsize=font_size)
    ax.set_ylabel(r'$I$', fontsize=font_size)
    ax.tick_params(labelsize=font_size - 4)

    x_vals = list(x) + list(x_line)
    y_vals = list(y) + list(y_line)
    x_lo, x_hi = min(x_vals), max(x_vals)
    y_lo, y_hi = min(y_vals), max(y_vals)
    x_margin = axis_pad * (x_hi - x_lo) if x_hi > x_lo else axis_pad
    y_margin = axis_pad * (y_hi - y_lo) if y_hi > y_lo else axis_pad
    ax.set_xlim(x_lo - x_margin, x_hi + x_margin)
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

fig.tight_layout()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=epoch_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.02)
cbar.set_ticks([epochs_array.min(), epochs_array.max()])
cbar.set_ticklabels([str(epochs_array.min()), str(epochs_array.max())])
cbar.set_label('epoch', fontsize=font_size)

# %%
# plot score mean-error time-profiles (all checkpoints)
# error_values are E_{p_t}[|s_θ - ∇log p|^2], i.e. mean squared L^2 score error
import diffsci.models
from matplotlib import colors as mcolors

path_list = [
    '/home/ubuntu/repos/DiffSci/stochasticity_paper/stats/output_default3',
    '/home/ubuntu/repos/DiffSci/stochasticity_paper/stats/output_default4',
    '/home/ubuntu/repos/DiffSci/stochasticity_paper/stats/output_default5',
]
all_errors_saved = [torch.load(f'{path}/all_errors.pt') for path in path_list]

process = 'edm'
nsteps = 500
initial_time = 80
scheduler = diffsci.models.EDMScheduler()
initial_time_ = torch.tensor(float(initial_time))
initial_step = int(scheduler.step_from_time(t=initial_time_, n=nsteps))
time = scheduler.create_steps(nsteps + 1)
initial_time = time[initial_step].item()
time2 = time[initial_step:]
times = time2[0] - time2
x = initial_time - times[1:]
sigma_x = scheduler.scheduler_fns.noise_fn(x)
theoretical_profile = sigma_x**(-1)

font_size = 18
alpha = 0.1
error_epochs = [epoch for epoch, _ in iter_saved_items(all_errors_saved)]
epochs_array = np.asarray(error_epochs)
epoch_norm = mcolors.Normalize(vmin=epochs_array.min(), vmax=epochs_array.max())
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(epochs_array)))
c_var = 1e4
c_mean = 1e1
c_fit = 5e2
e_var = 6
e_mean = 2
e_fit = 4

fig, ax = plt.subplots(figsize=(7, 5))
for i, (_, (error_values, _)) in enumerate(iter_saved_items(all_errors_saved)):
    ax.plot(x, np.asarray(error_values, dtype=float), color=colors[i], alpha=alpha)
ax.plot(x, theoretical_profile**e_var/c_var, color='purple', linestyle='--', alpha=1, zorder=2,label=f'Variance threshold ($\sigma(t)^{{{-e_var}}}$)')
ax.plot(x, theoretical_profile**e_mean/c_mean, color='red', linestyle='--', alpha=1, zorder=2, label=f'Mean threshold ($\sigma(t)^{{{-e_mean}}}$)')
ax.plot(x, theoretical_profile**e_fit/c_fit, color='orange', linestyle='--', alpha=1, zorder=2, label=f'Manual fit ($\sigma(t)^{{{-e_fit}}}$)')

ax.set_title(r'empirical time-profiles', fontsize=font_size + 2)
ax.set_xlabel(r'forward time $t$', fontsize=font_size)
ax.set_ylabel(r'$E_{p_t}[|\epsilon_t|^2]$', fontsize=font_size)
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(labelsize=font_size - 4)
ax.legend(fontsize=font_size - 4)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=epoch_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
cbar.set_ticks([epochs_array.min(), epochs_array.max()])
cbar.set_ticklabels([str(epochs_array.min()), str(epochs_array.max())])
cbar.set_label('epoch', fontsize=font_size)
fig.tight_layout()

# %%
# best log-log exponent of score error time-profiles: E(t) ~ c t^α
from scipy.stats import linregress

font_size = 18
alpha_lines = 0.2
axis_pad = 0.05

t = np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, dtype=float)
exponents, intercepts = [], []
for _, (error_values, _) in iter_saved_items(all_errors_saved):
    y = np.asarray(error_values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y) & (t > 0) & (y > 0)
    fit = linregress(np.log(t[mask]), np.log(y[mask]))
    exponents.append(fit.slope)
    intercepts.append(fit.intercept)
exponents = np.asarray(exponents, dtype=float)
intercepts = np.asarray(intercepts, dtype=float)
t_line = t[t > 0]

fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))

ax = axs[0]
for i, (a, b) in enumerate(zip(exponents, intercepts)):
    ax.plot(t_line, np.exp(b) * t_line ** a, color=colors[i], alpha=alpha_lines)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'diffusion time $t$', fontsize=font_size)
ax.set_ylabel(r'$c\, t^{\alpha}$', fontsize=font_size)
ax.set_title(r'best log-log fits', fontsize=font_size + 2)
ax.tick_params(labelsize=font_size - 4)

ax = axs[1]
ax.scatter(epochs_array, exponents, c=colors, marker='o', s=35, zorder=3, alpha=0.8)
ax.axhline(-2, color='red', linestyle='--', linewidth=1.5, label=r'mean ($\alpha=-2$)')
ax.axhline(-6, color='purple', linestyle='--', linewidth=1.5, label=r'variance ($\alpha=-6$)')
ax.set_xlabel('epoch', fontsize=font_size)
ax.set_ylabel(r'$\alpha$', fontsize=font_size)
ax.set_title(r'best-fitting exponent $\alpha$', fontsize=font_size + 2)
ax.tick_params(labelsize=font_size - 4)
ax.legend(fontsize=font_size - 4, loc='best')
y_lo, y_hi = exponents.min(), exponents.max()
y_margin = axis_pad * (y_hi - y_lo) if y_hi > y_lo else axis_pad
ax.set_ylim(min(y_lo, -6) - y_margin, y_hi + y_margin)

fig.tight_layout()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=epoch_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.02)
cbar.set_ticks([epochs_array.min(), epochs_array.max()])
cbar.set_ticklabels([str(epochs_array.min()), str(epochs_array.max())])
cbar.set_label('epoch', fontsize=font_size)

print(f'α mean={exponents.mean():+.3f}, median={np.median(exponents):+.3f}, '
      f'min={exponents.min():+.3f}, max={exponents.max():+.3f}')

# %%
# plot score mean-error time-profiles of 5 random checkpoints
# Reuse the error data, EDM time grid, and reference curves from above.
rng = np.random.default_rng(42)  # Change the seed for a different sample.
checkpoint_profiles = [
    (path.rsplit('/', 1)[-1], epoch, error_values)
    for path, saved_errors in zip(path_list, all_errors_saved)
    for epoch, (error_values, _) in sorted(saved_errors.items())
]
selected_indices = rng.choice(len(checkpoint_profiles), size=5, replace=False)

font_size = 18
fig, ax = plt.subplots(figsize=(9, 5))
for index in selected_indices:
    run_name, epoch, error_values = checkpoint_profiles[index]
    ax.plot(
        x, np.asarray(error_values, dtype=float), linewidth=1.5,
        label=f'{run_name}, epoch {epoch}',
    )

for exponent, scale, color, label in [
    (e_var, c_var, 'purple', 'Variance threshold'),
    (e_mean, c_mean, 'red', 'Mean threshold'),
    (e_fit, c_fit, 'orange', 'Manual fit'),
]:
    ax.plot(
        x, theoretical_profile**exponent / scale,
        color=color, linestyle='--',
        label=rf'{label} ($\sigma(t)^{{{-exponent}}}$)',
    )

ax.set_title('Score error: 5 random checkpoints', fontsize=font_size + 2)
ax.set_xlabel(r'forward time $t$', fontsize=font_size)
ax.set_ylabel(r'$E_{p_t}[|\epsilon_t|^2]$', fontsize=font_size)
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(labelsize=font_size - 4)
ax.legend(fontsize=font_size - 7, loc='best')
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Least-squares spatial score-error modes: all 144 checkpoints
#
# For each checkpoint and positive EDM noise level $t=\sigma$, we project the **signed** score error under the exact diffused mixture:
# $$\epsilon_\theta(x,t)=b(t)+C(t)(x-\mu_t)+r(x,t).$$
# The trained dataset is **1D**. Mixture-weighted Gaussian quadrature estimates the least-squares expectations; $r$ is orthogonal to constant and linear functions. The full coefficient curves and their residual energies are saved for every checkpoint.
#
# To compare with the draft's *small-error Gaussian* phase diagrams, use
# $$d=\Lambda-\ell=\log(V(t)/V_{\rm ref}),\quad a_{\rm lin}=V(t)C(t),\quad u_{\rm lin}=V(t)b(t)/\sqrt{V_{\rm ref}},$$
# where $V(t)=0.1219+t^2$ and $V_{\rm ref}=V(0.002)$. We fit $a_{\rm lin}\approx\epsilon_a e^{-\kappa_a d}$ and $u_{\rm lin}\approx\epsilon_m e^{-\kappa_m d}$ by signed least squares, weighted uniformly in $d$.
#
# These are **Gaussian surrogate coordinates**, so inspect the fit errors, amplitudes and nonlinear residual alongside the phase point. Exact Gaussian-parameter mappings and early/late fit sensitivities are also saved. The existing empirical KL sweeps start at **$\gamma=0.01$, not zero**; empirical comparisons below use that recorded baseline.
#
# Implementation and reproduction details: [score-error analysis README](score_error_analysis/README.md).
#

# %%
# Load the completed analysis independently of the earlier notebook setup.
from pathlib import Path
import json
import sys
import numpy as np
from html import escape
import matplotlib.pyplot as plt
from IPython.display import display, Image, HTML

repo_root = next(
    path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / 'paper_odds/score_error_analysis/score_error_modes.py').exists()
)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
mode_output = repo_root / 'paper_odds/outputs/score_error_modes'
# To reproduce from the repository root:
# python -m paper_odds.score_error_analysis.fit_checkpoint_modes
mode_metadata = json.loads((mode_output / 'metadata.json').read_text())
mode_summary = np.genfromtxt(
    mode_output / 'checkpoint_summary.csv', names=True, delimiter=',',
    dtype=None, encoding='utf-8',
).view(np.recarray)
with np.load(mode_output / 'clock.npz') as saved_clock:
    mode_clock = {name: saved_clock[name] for name in saved_clock.files}
print(f"Loaded {len(mode_summary)} checkpoints; complete={mode_metadata['complete']}")
mode_columns = [
    'run', 'epoch', 'kappa_a', 'kappa_m', 'amplitude_ratio',
    'shape_fit_relative_rmse', 'mean_fit_relative_rmse',
    'nonlinear_energy_fraction',
]
# Scrollable table without requiring an additional dataframe library.
table_header = '<tr>' + ''.join(f'<th>{escape(key)}</th>' for key in mode_columns) + '</tr>'
table_rows = ''.join(
    '<tr>' + ''.join(f'<td>{escape(str(row[key]))}</td>' for key in mode_columns) + '</tr>'
    for row in mode_summary
)
display(HTML('<div style="max-height:400px;overflow:auto"><table>'
             + table_header + table_rows + '</table></div>'))


# %%
# Inspect b(t), C(t), normalized profiles, and the nonlinear remainder.
run_to_inspect = 'default3'
epoch_to_inspect = 24
selected_row = mode_summary[
    (mode_summary['run'] == run_to_inspect)
    & (mode_summary['epoch'] == epoch_to_inspect)
][0]
with np.load(mode_output / 'profiles' / f'{run_to_inspect}_epoch{epoch_to_inspect:02d}.npz') as saved_profile:
    mode_profile = {name: saved_profile[name] for name in saved_profile.files}

sigma, d = mode_clock['sigma'], mode_clock['distance']
fig, axs = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
for ax, key in zip(axs[0], ['b', 'C']):
    ax.plot(sigma, mode_profile[key])
    ax.set(xscale='log', xlabel=r'EDM noise $t=\sigma$', ylabel=f'{key}(t)')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.grid(alpha=0.2)
for key, amplitude, exponent, color, label in [
    ('a_linear', selected_row.epsilon_a, selected_row.kappa_a, 'tab:orange', 'shape: VC'),
    ('u_linear', selected_row.epsilon_m, selected_row.kappa_m, 'tab:blue', 'mean: Vb/sqrt(V_ref)'),
]:
    axs[1, 0].plot(d, mode_profile[key], color=color, label=label)
    axs[1, 0].plot(d, amplitude * np.exp(-exponent * d), '--', color=color, label='exponential fit')
axs[1, 0].set_yscale('symlog', linthresh=0.01)
axs[1, 0].set(xlabel=r'$d=\Lambda-\ell$', ylabel='Signed normalized profile')
axs[1, 0].legend()
for key, label in [('mean_energy', 'mean'), ('affine_energy', 'linear'), ('residual_energy', 'nonlinear')]:
    axs[1, 1].plot(d, mode_profile[key] / np.maximum(mode_profile['total_energy'], 1e-30), label=label)
axs[1, 1].set(xlabel=r'$d=\Lambda-\ell$', ylabel='Fraction of score-error energy', ylim=(0, 1.02))
axs[1, 1].legend()
fig.suptitle(f'{run_to_inspect}, epoch {epoch_to_inspect}')
plt.show()


# %%
# All coordinates, including points outside the pre-existing axes.
display(Image(filename=str(mode_output / 'checkpoint_phase_coordinates.png')))
# The background uses the nearest amplitude-ratio panel; numeric predictions
# always use each checkpoint's own ratio. Crosses now carry measured KL colors
# from the controlled FULL-SCORE sampler, 64 bins, with its TRUE ODE baseline.
# This overlay therefore uses different measurements from the old sweeps below.
display(Image(filename=str(mode_output / 'checkpoint_phase_overlays.png')))


# %%
# Compare the selected checkpoint's observed and predicted gamma responses.
kl_comparison = np.genfromtxt(
    mode_output / 'kl_comparison.csv', names=True, delimiter=',',
    dtype=None, encoding='utf-8',
).view(np.recarray)
selected_kl = kl_comparison[
    (kl_comparison['run'] == run_to_inspect)
    & (kl_comparison['epoch'] == epoch_to_inspect)
]
selected_kl = np.sort(selected_kl, order='gamma')
fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
for column, label, style in [
    ('empirical_log10_ratio_q_p', 'Observed KL(q || p)', '-'),
    ('empirical_log10_ratio_p_q', 'Observed KL(p || q)', '-'),
    ('profile_log10_ratio_recorded_baseline', 'Gaussian kernels: measured signed profiles', '--'),
    ('exponential_finite_log10_ratio_recorded_baseline', 'Gaussian kernels: exponential fits', ':'),
]:
    ax.plot(selected_kl['gamma'], selected_kl[column], style, label=label)
ax.axhline(0, color='gray', linewidth=1)
ax.set(xscale='log', xlabel=r'Stochasticity $\gamma$',
       ylabel='log10(KL / KL at recorded baseline)',
       title=f'{run_to_inspect}, epoch {epoch_to_inspect}; baseline gamma = {selected_kl[0].empirical_baseline_gamma:.3g}')
ax.legend()
plt.show()
display(Image(filename=str(mode_output / 'empirical_vs_gaussian_profiles.png')))

# Descriptive association of nonlinear residual with prediction disagreement.
display(Image(filename=str(mode_output / 'nonlinearity_vs_disagreement.png')))


# %% [markdown]
# The nonlinear-energy fraction measures the part of the score error that an affine field cannot explain under the mixture. Its association with prediction disagreement is descriptive: large amplitudes, poor exponential fits, the non-Gaussian reference, the approximate sampling prior, finite-step integration and histogram KL estimation also contribute. A matched sampling ablation is needed to isolate the residual's causal effect. The 144 checkpoints belong to three training runs.
#

# %% [markdown]
# ## Controlled nonlinear-residual ablation
#
# We compare $s_0=s_{\mathrm{exact}}+b(t)+C(t)(x-\mu_t)$ with $s_1=s_\theta$, keeping the exact mixture prior, time steps, and random draws matched. The mixture score itself remains nonlinear in both arms. The experiment covers all 144 checkpoints, $\gamma\in\{0,0.01,0.2,1,5\}$, and four seeds with 4,096 particles each. Here **$\gamma=0$ is a measured ODE baseline**.
#
# The exact-score control measures numerical and finite-sample KL error. All arms use fixed exact-target quantile bins; results are stored at 32, 64 and 128 bins. These finite-partition KLs differ from the original saved histogram estimates. The target is the mixture at positive noise $\sigma=0.002$.
#
# Besides the first-order Gaussian profile prediction, we integrate the full Gaussian mean/variance equations with the same $b,C$ to examine the small-amplitude approximation separately. See [the experiment README](score_error_analysis/README.md#matched-nonlinear-residual-ablation).
#

# %%
# Load the paired intervention independently of the earlier notebook cells.
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image

repo_root = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'diffsci').is_dir())
paper_dir = repo_root / 'paper_odds'
ablation_dir = paper_dir / 'outputs/residual_ablation'
ablation_metadata = json.loads((ablation_dir / 'metadata.json').read_text())
assert ablation_metadata['complete'], 'Finish the ablation run before loading its aggregate results.'
ablation_results = np.genfromtxt(ablation_dir / 'ablation_results.csv', names=True, delimiter=',', dtype=None, encoding='utf-8').view(np.recarray)
ablation_effects = np.genfromtxt(ablation_dir / 'paired_effects.csv', names=True, delimiter=',', dtype=None, encoding='utf-8').view(np.recarray)
print(f"{ablation_metadata['completed_checkpoints']} checkpoints, "
      f"{ablation_metadata['particles'] * len(ablation_metadata['seeds']):,} pooled particles per arm, "
      f"prior={ablation_metadata['prior']}")


# %%
# Edit these to inspect a checkpoint, and compare histogram resolutions.
ablation_run = 'default3'
ablation_epoch = 24
ablation_bins = 64
selected_ablation = ablation_results[
    (ablation_results.run == ablation_run)
    & (ablation_results.epoch == ablation_epoch)
    & (ablation_results.bins == ablation_bins)
]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
for ax, direction in zip(axs, ['q_p', 'p_q']):
    for lam, label in [(0., 'Affine error only'), (1., 'Full learned score')]:
        rows = np.sort(selected_ablation[selected_ablation.residual_lambda == lam], order='gamma')
        ax.plot(rows.gamma, rows[f'log10_ratio_ode_{direction}'], 'o-', label=label)
    ax.plot(rows.gamma, rows[f'gaussian_profile_log10_ratio_ode_{direction}'], '--', label='Gaussian first-order response')
    ax.plot(rows.gamma, rows[f'gaussian_moment_log10_ratio_ode_{direction}'], ':', label='Gaussian full moments')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set(xlabel='Stochasticity gamma', ylabel='log10(KL_gamma / KL_ODE)', title=direction)
    ax.legend(fontsize=9)
fig.suptitle(f'{ablation_run}, epoch {ablation_epoch}, {ablation_bins} target-quantile bins')
plt.show()


# %%
# Positive reduction means removing the residual improves the Gaussian prediction.
# The bars quantify paired Monte Carlo uncertainty, not variation across training runs.
selected_effects = ablation_effects[
    (ablation_effects.run == ablation_run)
    & (ablation_effects.epoch == ablation_epoch)
    & (ablation_effects.bins == ablation_bins)
]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
for direction in ['q_p', 'p_q']:
    rows = np.sort(selected_effects[selected_effects.kl_direction == direction], order='gamma')
    axs[0].errorbar(rows.gamma, rows.nonlinear_effect_log10_ratio, yerr=rows.nonlinear_effect_mc_se, fmt='o-', label=direction)
    axs[1].errorbar(rows.gamma, rows.disagreement_reduction, yerr=rows.disagreement_reduction_mc_se, fmt='o-', label=direction)
for ax in axs:
    ax.set_xscale('symlog', linthresh=0.01)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Stochasticity gamma')
    ax.legend()
axs[0].set_ylabel('Full minus affine log10(KL ratio)')
axs[1].set_ylabel('Prediction-gap reduction after removing residual')
plt.show()


# %%
# All-checkpoint comparisons, with controls and bin-resolution checks in the files.
display(Image(filename=str(ablation_dir / 'ablation_prediction_gaps.png')))
display(Image(filename=str(ablation_dir / 'nonlinear_effect_vs_energy.png')))
display(Image(filename=str(ablation_dir / 'exact_score_control.png')))


# %% [markdown]
# ## Exponential-fit KL predictions and measured phase colors
#
# The finite- and infinite-horizon exponential predictions were already saved; the earlier aggregate correlations used **signed measured profiles**. This section compares all three predictors with measurements for all 144 checkpoints. Each exponential prediction uses the checkpoint's own fitted amplitudes and exponents, without replacing its amplitude ratio by a panel value.
#
# Choose `saved_sweep` for the original measurements (baseline $\gamma=0.01$), `controlled_full` for the full learned score, or `controlled_affine` for the residual-removed score (both with a measured ODE baseline). The controlled comparisons include 32, 64 and 128 bins; figures below use 64. Correlations are computed separately at each gamma and KL direction. Spearman is unchanged by taking logarithms; Pearson is reported on both scales. These are descriptive statistics from three correlated training trajectories.
#
# At $\gamma=1$ with the full score, finite-exponential Spearman is **0.198 / 0.207** (q to p / p to q), compared with **0.405 / 0.362** for signed profiles. Infinite-horizon fits give similar results to finite-horizon fits. Poor exponential fits remain included and flagged.
#
# The phase overlays now color crosses by the **measured full-score** $\log_{10}(\mathrm{KL}_\gamma/\mathrm{KL}_{ODE})$, with the same blue–white–red scale as the theory background. Blue means smaller KL than the ODE; red means larger. The original axes contain only 15 checkpoints; the companion symmetric-log plot shows all 144. Backgrounds use the panel's ratio; crosses still flag poor/boundary fits.
#
# Details and commands: [README](score_error_analysis/README.md#exponential-fit-predictions-versus-measured-kl-ratios). All correlations, subset checks and per-run results: [CSV](outputs/exponential_kl_comparison/correlations.csv).

# %%
# Load independently; change these choices to compare the saved experiments.
from pathlib import Path
import csv
from html import escape
from IPython.display import display, Image, HTML

comparison_repo = next(path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / 'paper_odds/outputs/exponential_kl_comparison/correlations.csv').exists())
comparison_paper = comparison_repo / 'paper_odds'
comparison_dir = comparison_paper / 'outputs/exponential_kl_comparison'
comparison_dataset = 'controlled_full'  # 'saved_sweep', 'controlled_full', 'controlled_affine'
comparison_direction = 'q_p'            # 'q_p' or 'p_q'
comparison_bins = 64                    # 32, 64, 128; ignored for saved_sweep
comparison_subset = 'all'               # 'common_valid', 'above_control_floor' (controlled only)
comparison_run = 'all'                  # 'default3', 'default4', 'default5'
with (comparison_dir / 'correlations.csv').open() as f:
    exponential_correlations = list(csv.DictReader(f))
comparison_table = [r for r in exponential_correlations
    if r['dataset'] == comparison_dataset and r['run'] == comparison_run
    and r['subset'] == comparison_subset
    and int(r['bins']) == (0 if comparison_dataset == 'saved_sweep' else comparison_bins)]
if not comparison_table:
    raise ValueError('No results for these selections; the control-floor subset requires controlled measurements.')
comparison_columns = ['gamma', 'kl_direction', 'model', 'n', 'spearman',
    'pearson_log10', 'pearson_ratio', 'sign_agreement', 'median_absolute_log10_gap']
def comparison_format(key, value):
    return f'{float(value):.3f}' if key in comparison_columns[4:] else escape(value)
comparison_header = '<tr>' + ''.join(f'<th>{escape(k)}</th>' for k in comparison_columns) + '</tr>'
comparison_body = ''.join('<tr>' + ''.join(f'<td>{comparison_format(k, r[k])}</td>'
    for k in comparison_columns) + '</tr>' for r in comparison_table)
print(f'{comparison_dataset}; baseline gamma={comparison_table[0]["baseline_gamma"]}; '
      f'run={comparison_run}; subset={comparison_subset}; bins={comparison_table[0]["bins"]}')
display(HTML('<div style="max-height:500px;overflow:auto"><table>'
    + comparison_header + comparison_body + '</table></div>'))


# %%
# Saved overview: all runs, all fit qualities; controlled figures use 64 bins.
# Rows compare direct signed profiles, finite fits, and infinite fits.
# Colors identify training runs; white centers flag estimates near the control floor.
# The table above also allows other bin counts and subsets.
display(Image(filename=str(comparison_dir / f'{comparison_dataset}_{comparison_direction}.png')))


# %%
# Phase colors always show the FULL learned score, 64 bins, with measured ODE baseline.
# Both figures use the KL direction selected above; they do not follow dataset/subset choices.
comparison_phase_name = ('checkpoint_phase_overlays.png' if comparison_direction == 'q_p'
                         else 'checkpoint_phase_overlays_p_q.png')
display(Image(filename=str(comparison_paper / 'outputs/score_error_modes' / comparison_phase_name)))
display(Image(filename=str(comparison_dir / f'phase_coordinates_measured_{comparison_direction}.png')))


# %% [markdown]
# ## Kappa sensitivity: smoothing and cumulative effects
#
# The current phase fit minimizes **signed profile squared error**; it does not average local log-log slopes. Narrow terminal peaks can dominate that loss even when their area is small. The log-variance clock also compresses low-sigma intervals. For the largest original mean exponent (default3, epoch 18; $\kappa_m\approx4924$), $d\le0.01$ accounts for **53.55% of squared profile energy but 1.22% of absolute area**.
#
# We compare the original fit with Gaussian smoothing at widths **0.01, 0.05, 0.2 in d units**, preserving signed area with conservative rebinning and reflecting boundaries. A separate **cumulative** fit matches $J(d)=\int_0^d e^{-\beta s}f(s)ds$ to $\epsilon F(\kappa+\beta,d)$, with $\beta=0$ for shape and $1/2$ for mean. All estimators use the original wide exponent bounds; no measured KL values are used for fitting.
#
# At smoothing width 0.05, the largest shape/mean exponents fall to **30.8 / 27.5**, and median kernel-response changes are about **3.1%**. The cumulative fit lowers typical exponents, but **18 shape and 36 mean modes have unresolved large-kappa alternatives** under the loss-envelope diagnostic. The cumulative estimator improves gamma=1 correlations but worsens them at gamma=5. These are sensitivity estimates, not uniquely recovered growth rates.
#
# The original analysis is preserved. New results: [kappa_sensitivity](outputs/kappa_sensitivity/), [definitions](score_error_analysis/README.md#kappa-estimation-smoothing-and-cumulative-response-sensitivity), [interpretation](score_error_analysis/RESULTS.md#why-some-kappa-estimates-are-huge).

# %%
# Load the sensitivity analysis independently of earlier notebook cells.
from pathlib import Path
import csv, json, sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image, HTML
from html import escape

kappa_repo = next(path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / 'paper_odds/outputs/kappa_sensitivity/summary.json').exists())
if str(kappa_repo) not in sys.path:
    sys.path.insert(0, str(kappa_repo))
kappa_paper = kappa_repo / 'paper_odds'
kappa_output = kappa_paper / 'outputs/kappa_sensitivity'
kappa_summary = json.loads((kappa_output / 'summary.json').read_text())
with (kappa_output / 'mode_fits.csv').open() as f:
    kappa_fits = list(csv.DictReader(f))
with (kappa_output / 'correlations.csv').open() as f:
    kappa_correlations = list(csv.DictReader(f))
kappa_html = '<tr><th>Method</th><th>Median shape kappa</th><th>Median mean kappa</th><th>Max shape</th><th>Max mean</th><th>Within phase axes</th></tr>'
for method, result in kappa_summary.items():
    values = [method, f"{result['shape']['median_kappa']:.3f}", f"{result['mean']['median_kappa']:.3f}",
              f"{result['shape']['max_kappa']:.1f}", f"{result['mean']['max_kappa']:.1f}",
              f"{result['inside_original_axes']}/144"]
    kappa_html += '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in values) + '</tr>'
display(HTML('<table>' + kappa_html + '</table>'))
display(Image(filename=str(kappa_output / 'kappa_distributions.png')))


# %%
# Inspect any checkpoint: signed profile, cumulative response, and alternative fits.
kappa_run = 'default3'
kappa_epoch = 18
kappa_mode = 'mean'          # 'shape' or 'mean'
kappa_bandwidth = 0.05      # 0.01, 0.05, 0.2 in log-variance units
from io import BytesIO
from paper_odds.score_error_analysis.kappa_estimators import (
    smooth_profile, primitive_at, cumulative_basis,
)
with np.load(kappa_paper / 'outputs/score_error_modes/clock.npz') as clock:
    kappa_d = clock['distance']
with np.load(kappa_paper / 'outputs/score_error_modes/profiles' / f'{kappa_run}_epoch{kappa_epoch:02d}.npz') as data:
    kappa_y = data['a_linear' if kappa_mode == 'shape' else 'u_linear']
kappa_beta = 0. if kappa_mode == 'shape' else 0.5
kappa_smoothed = smooth_profile(kappa_d, kappa_y, kappa_bandwidth)
kappa_methods = ['original', f'smooth_{kappa_bandwidth:g}', 'cumulative']
kappa_selected = [r for r in kappa_fits if r['run'] == kappa_run
    and int(r['epoch']) == kappa_epoch and r['mode'] == kappa_mode and r['method'] in kappa_methods]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.7), layout='constrained')
axs[0].plot(kappa_d, kappa_y, c='black', label='original signed profile')
axs[0].plot(np.r_[0., kappa_smoothed['distance'], kappa_d[-1]],
            np.r_[kappa_smoothed['profile'][0], kappa_smoothed['profile'], kappa_smoothed['profile'][-1]],
            label=f'smoothed: width {kappa_bandwidth:g}')
axs[1].plot(kappa_d, primitive_at(kappa_d, np.exp(-kappa_beta*kappa_d)*kappa_y, kappa_d),
            c='black', lw=2, label='actual signed cumulative response')
axs[2].plot(kappa_d, kappa_y, c='black', alpha=.6, label='original profile')
for row in kappa_selected:
    k, amp = float(row['kappa']), float(row['amplitude'])
    label = f"{row['method']}: kappa={k:.3g}"
    axs[1].plot(kappa_d, amp*cumulative_basis(k+kappa_beta, kappa_d), '--', label=label)
    axs[2].plot(kappa_d, amp*np.exp(-k*kappa_d), label=label)
    if row['method'] == 'cumulative':
        print('Cumulative objective-sensitivity envelope (not a confidence interval):',
              row['loss_envelope_low'], row['loss_envelope_high'],
              '; upper unresolved =', row['upper_unresolved'])
for i in (0, 2):
    axs[i].set_xscale('symlog', linthresh=.001)
    axs[i].set_yscale('symlog', linthresh=.01)
for ax, title in zip(axs, ['Conservative smoothing', 'Cumulative ODE mode response', 'Fitted exponential profiles']):
    ax.set(title=title, xlabel='Log-variance distance d; terminal time = 0')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.suptitle(f'{kappa_run}, epoch {kappa_epoch}: {kappa_mode}')
kappa_buffer = BytesIO()
fig.savefig(kappa_buffer, format='png', dpi=130, bbox_inches='tight')
display(Image(data=kappa_buffer.getvalue()))
plt.close(fig)


# %%
# Compare all checkpoints, then inspect separately saved phase coordinates.
display(Image(filename=str(kappa_output / 'estimator_correlations.png')))
kappa_phase_method = 'smooth_0.05'  # or 'cumulative'; both have precomputed overlays
# Marker colors still use full-score measured KL(q || p), 64 bins, true ODE baseline.
# Fit-quality flags refer to the selected method's objective.
# Cumulative upper-unresolved flags are in mode_fits.csv; inspect them before interpreting kappa.
display(Image(filename=str(kappa_output / kappa_phase_method / 'checkpoint_phase_overlays.png')))


# %% [markdown]
# ## Global least squares on log-log plots
#
# For each normalized mode $f=a_{\rm lin}$ or $u_{\rm lin}$, fit
# $$\log|f|=c+s\log(V/V_{\rm ref}),\qquad \kappa=-s,\quad |\epsilon|=e^c.$$
# This is one global straight-line fit, with no differentiation or smoothing. The horizontal variable is **variance ratio**: $d=\log(V/V_{\rm ref})$ is already a logarithm. Using $\log d$ or $\log\sigma$ would estimate a different exponent.
#
# `loglog` weights uniformly in log variance; `loglog_unweighted` is ordinary least squares on retained saved nodes. The hybrid grid is dense near terminal noise, so equal node weights emphasize that region. We exclude magnitudes below a relative cutoff rather than clipping them; cutoffs $10^{-8},10^{-6},10^{-4}$ are compared.
#
# The primary weighted fits give shape $\kappa\in[0.306,0.961]$ and mean $\kappa\in[-0.394,0.490]$, with all 144 points inside the old axes. Median log-space $R^2$ is **0.894 for shape / 0.230 for mean**. These are **magnitude envelopes**: 131 shape and 121 mean profiles change sign, and the fitted envelopes cannot preserve cancellation. Ordinary fits give weak measured-KL rank correlations (about 0.11–0.17); weighted fits give correlations near zero or slightly negative.
#
# [Definitions and reproduction](score_error_analysis/README.md#global-log-log-slope-fits), [results](score_error_analysis/RESULTS.md#global-log-log-least-squares), [all fit coefficients](outputs/loglog_kappa/mode_fits.csv). Previous experiments are preserved.

# %%
# Load the log-log experiment independently of earlier notebook cells.
from pathlib import Path
import csv, json, sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image, HTML
from html import escape

loglog_repo = next(path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / 'paper_odds/outputs/loglog_kappa/summary.json').exists())
if str(loglog_repo) not in sys.path:
    sys.path.insert(0, str(loglog_repo))
loglog_paper = loglog_repo / 'paper_odds'
loglog_output = loglog_paper / 'outputs/loglog_kappa'
loglog_summary = json.loads((loglog_output / 'summary.json').read_text())
with (loglog_output / 'mode_fits.csv').open() as f:
    loglog_fits = list(csv.DictReader(f))
with (loglog_output / 'correlations.csv').open() as f:
    loglog_correlations = list(csv.DictReader(f))
loglog_html = '<tr><th>Method</th><th>Mode</th><th>Min kappa</th><th>Median kappa</th><th>Max kappa</th><th>Median log R²</th></tr>'
for method, result in loglog_summary.items():
    for mode in ('shape', 'mean'):
        values = [method, mode] + [f'{result[mode][k]:.3f}' for k in
                  ('min_kappa', 'median_kappa', 'max_kappa', 'median_log_r2')]
        loglog_html += '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in values) + '</tr>'
display(HTML('<table>' + loglog_html + '</table>'))


# %%
# Inspect a global log-log fit and the original profile signs.
loglog_run = 'default3'
loglog_epoch = 18
loglog_mode = 'mean'                 # 'shape' or 'mean'
loglog_weighting = 'log_variance'    # 'points' for ordinary least squares
loglog_relative_cutoff = 1e-6       # compare 1e-8 and 1e-4
from io import BytesIO
from paper_odds.score_error_analysis.loglog_kappa import fit_loglog

with np.load(loglog_paper / 'outputs/score_error_modes/clock.npz') as clock:
    loglog_d = clock['distance']
with np.load(loglog_paper / 'outputs/score_error_modes/profiles' / f'{loglog_run}_epoch{loglog_epoch:02d}.npz') as data:
    loglog_y = data['a_linear' if loglog_mode == 'shape' else 'u_linear']
loglog_fit = fit_loglog(loglog_d, loglog_y, weighting=loglog_weighting,
                       relative_cutoff=loglog_relative_cutoff)
print(f"kappa={loglog_fit['kappa']:.6g}; slope={loglog_fit['slope']:.6g}; "
      f"amplitude magnitude={loglog_fit['amplitude']:.6g}; log R²={loglog_fit['log_r2']:.3f}")
print(f"Sign changes={loglog_fit['sign_changes']}; "
      f"retained clock fraction={loglog_fit['retained_clock_fraction']:.2%}; "
      f"excluded samples={loglog_fit['excluded_count']}")
fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
for mask, color, label in [(loglog_y > 0, 'C0', 'original profile positive'),
                           (loglog_y < 0, 'C1', 'original profile negative')]:
    ax.scatter(np.exp(loglog_d[mask]), abs(loglog_y[mask]), c=color, s=20, label=label)
loglog_retained = abs(loglog_y) > loglog_fit['absolute_cutoff']
loglog_excluded = (~loglog_retained) & (loglog_y != 0)
ax.scatter(np.exp(loglog_d[loglog_excluded]), abs(loglog_y[loglog_excluded]),
           marker='x', c='black', s=35, label='excluded by cutoff')
ax.plot(np.exp(loglog_d), loglog_fit['amplitude']*np.exp(-loglog_fit['kappa']*loglog_d),
        c='black', lw=2, label='global log-magnitude fit')
ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=.2)
ax.set(xlabel='Variance ratio V / V_ref', ylabel=f'Absolute {loglog_mode} profile',
       title=f'{loglog_run}, epoch {loglog_epoch}: {loglog_weighting}')
ax.legend()
loglog_buffer = BytesIO()
fig.savefig(loglog_buffer, format='png', dpi=130, bbox_inches='tight')
display(Image(data=loglog_buffer.getvalue()))
plt.close(fig)


# %%
# All-checkpoint evaluations; measured KL values were not used in fitting.
display(Image(filename=str(loglog_output / 'loglog_correlations.png')))
loglog_phase_method = 'loglog'   # or 'loglog_unweighted'
# Crosses flag poor magnitude-profile RMSE; sign changes are recorded separately.
# Colors use full-score measured KL(q || p), 64 bins, and the measured ODE baseline.
display(Image(filename=str(loglog_output / loglog_phase_method / 'checkpoint_phase_overlays.png')))


# %% [markdown]
# ## Prediction agreement across all kappa strategies
#
# The primary question here is **how close the predicted ratios are to the measured ratios**. We evaluate fixed predictions against the identity line, retaining correlations as separate diagnostics:
# $$\mathrm{RMSE}_{\log}=\sqrt{\frac1N\sum_i(\log_{10}\widehat R_i-\log_{10}R_i)^2},\qquad
# \mathrm{RMSE}_{R}=\sqrt{\frac1N\sum_i(\widehat R_i-R_i)^2}.$$
# Log errors treat reciprocal multiplicative errors symmetrically. Raw-ratio errors penalize large absolute ratio errors more strongly. No calibration slope or intercept is fitted to measured KL.
#
# We also report MAE, prediction bias, concordance, fixed-prediction R², and **skill versus predicting ratio=1**: $1-\mathrm{MSE}_{model}/\mathrm{MSE}_{ratio=1}$ (positive is better). Correlation can remain high with large bias; R² here is not Pearson r².
#
# This changes the assessment of log-log fitting. At $\gamma=1$, full-score q-to-p log-RMSE is **0.276** for weighted log-log versus **0.499** for the original exponential, although weighted log-log has weak rank correlation. With equal weight over $\gamma\in\{0.2,1,5\}$, ordinary log-log has the best overall error among evaluated strategies: **0.295 / 0.330** for q-to-p / p-to-q. The ratio=1 baseline is **0.299 / 0.340**, so the gain over this baseline is modest.
#
# The comparison uses the same checkpoints across all strategies, preserving each dataset's baseline. Tables include the original sweeps, controlled full/affine arms, per-run and above-control-floor subsets. These are descriptive evaluations of three correlated training trajectories. [Metric definitions](score_error_analysis/README.md#prediction-agreement-across-kappa-estimation-strategies), [results](score_error_analysis/RESULTS.md#prediction-agreement-across-kappa-strategies), [full metric table](outputs/prediction_agreement/metrics.csv).

# %%
# Select an evaluation; sort by a numerical agreement metric (lower is better).
from pathlib import Path
import csv, sys
import numpy as np
from html import escape
from IPython.display import display, HTML, Image

agreement_repo = next(path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / 'paper_odds/outputs/prediction_agreement/metrics.csv').exists())
if str(agreement_repo) not in sys.path:
    sys.path.insert(0, str(agreement_repo))
agreement_paper = agreement_repo / 'paper_odds'
agreement_output = agreement_paper / 'outputs/prediction_agreement'
from paper_odds.score_error_analysis.prediction_agreement import LABELS
with (agreement_output / 'metrics.csv').open() as f:
    agreement_metrics = list(csv.DictReader(f))
with (agreement_output / 'aggregate_metrics.csv').open() as f:
    agreement_aggregate = list(csv.DictReader(f))
agreement_dataset = 'controlled_full'  # 'controlled_affine' or 'saved_sweep'
agreement_direction = 'q_p'            # 'p_q' for reverse KL
agreement_target_gamma = 1.0           # nearest recorded gamma selected for saved_sweep
agreement_subset = 'all'               # or 'above_control_floor' (controlled datasets only)
agreement_run = 'all'                  # or 'default3', 'default4', 'default5'
agreement_sort_metric = 'rmse_log10'    # or 'rmse_ratio', 'mae_log10', 'mae_ratio'
agreement_gammas = sorted({float(r['gamma']) for r in agreement_metrics if r['dataset'] == agreement_dataset})
agreement_gamma = min(agreement_gammas, key=lambda g: abs(g-agreement_target_gamma))
def agreement_selected(r):
    return (r['dataset'] == agreement_dataset and r['kl_direction'] == agreement_direction
            and r['subset'] == agreement_subset and r['run'] == agreement_run)
agreement_rows = sorted([r for r in agreement_metrics if agreement_selected(r) and float(r['gamma']) == agreement_gamma],
                        key=lambda r: float(r[agreement_sort_metric]))
if not agreement_rows:
    raise ValueError('No matching comparison; saved sweeps have no exact-score control-floor subset.')
def agreement_table(rows, columns):
    header = '<tr>' + ''.join(f'<th>{escape(label)}</th>' for key, label in columns) + '</tr>'
    body = ''
    for row in rows:
        values = []
        for key, label in columns:
            if key == 'method':
                value = LABELS[row[key]]
            elif key == 'n':
                value = row[key]
            else:
                v = float(row[key]); value = f'{v:.4f}' if np.isfinite(v) else '—'
            values.append(f'<td>{escape(value)}</td>')
        body += '<tr>' + ''.join(values) + '</tr>'
    display(HTML('<div style="overflow:auto;max-height:600px"><table>' + header + body + '</table></div>'))
print(f"{agreement_dataset}; {agreement_direction}; actual gamma={agreement_gamma:g}; "
      f"baseline gamma={float(agreement_rows[0]['baseline_gamma']):g}; {agreement_subset}; run={agreement_run}")
print('RMSE/MAE: lower is better; bias: ideal 0; skill: positive beats ratio=1; concordance: ideal 1.')
agreement_table(agreement_rows, [('method', 'Strategy'), ('n', 'n'), ('rmse_log10', 'RMSE log10'),
    ('rmse_ratio', 'RMSE ratio'), ('mae_log10', 'MAE log10'), ('bias_log10', 'Bias log10'),
    ('skill_vs_no_change_log10', 'Skill vs ratio=1'), ('concordance_log10', 'Concordance log10'),
    ('spearman', 'Spearman'), ('pearson_log10', 'Pearson log10')])
agreement_overall = sorted([r for r in agreement_aggregate if agreement_selected(r)],
                          key=lambda r: float(r[agreement_sort_metric]))
print('Across-gamma summary: equal gamma weights for MSE; mean within-gamma correlations.')
agreement_table(agreement_overall, [('method', 'Strategy'), ('rmse_log10', 'Overall RMSE log10'),
    ('rmse_ratio', 'Overall RMSE ratio'), ('skill_vs_no_change_log10', 'Skill vs ratio=1'),
    ('mean_spearman', 'Mean Spearman'), ('mean_pearson_log10', 'Mean Pearson log10')])


# %%
# Overview figures always show controlled FULL-score measurements, all runs, 64 bins.
# The selected tables above retain other datasets, runs and control-floor subsets.
display(Image(filename=str(agreement_output / 'agreement_and_correlation.png')))
display(Image(filename=str(agreement_output / 'raw_ratio_agreement.png')))


# %%
# Identity plots follow the dataset, gamma, direction, subset and run selected above.
from io import BytesIO
import matplotlib.pyplot as plt
from paper_odds.score_error_analysis.prediction_agreement import (
    load_predictions, identity_figure,
)
agreement_records, _ = load_predictions(agreement_paper)
if agreement_run != 'all':
    agreement_records = [r for r in agreement_records if r['run'] == agreement_run]
fig = identity_figure(agreement_records, dataset=agreement_dataset, gamma=agreement_gamma,
                      direction=agreement_direction, subset=agreement_subset)
agreement_buffer = BytesIO()
fig.savefig(agreement_buffer, format='png', dpi=130, bbox_inches='tight')
display(Image(data=agreement_buffer.getvalue()))
plt.close(fig)


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
#     version: 3.10.15
# ---

# %% [markdown]
# # Where to inject noise: checkpoint-backed stochasticity schedules
#
# This notebook compares time-dependent stochasticity with the ODE and fixed positive γ using the
# **same 144 default3/default4/default5 checkpoints** as notebooks 056 and 057-stoch_tests.
# It runs the exact-score control, the exact mixture score plus affine projected error, and the **full learned score**.
# The checkpoint loader is connected and required; no requested arm is silently skipped.
#
# The original low/mid/high windows and burst are retained as exploratory candidates. They have different
# integrated stochasticity budgets, so their comparison combines placement and amount. Three additional
# windows have the **same amplitude and width in the log-variance clock**, isolating placement more directly.
# The ODE is included among constant competitors. “Best constant” always means best among this finite tested grid.
# Population and checkpoint-specific windows are selected using score profiles, not sampled KL; they remain
# in-sample theory-guided choices, not held-out validation. The original numerical window locations came from
# an earlier envelope analysis; the revised a,u fits do not by themselves establish those same optima.
#
# Sampling uses the validated conventions of the residual ablation: an exact mixture prior at σ=80,
# target σ=0.002, and Euler / Euler–Maruyama in **z=(x−μ)/√V and ℓ=log(Vmax/V)**. The starting grid is
# 500 intervals from the **EDM ρ=7 noise schedule**, not a uniform-ℓ grid. Every schedule edge is inserted,
# then a **common** stiffness refinement is applied to all schedules, checkpoints and score arms. Reported
# actual step counts therefore exceed 500. All comparisons use identical grid nodes, initial states and
# Brownian increments, with four independent replicate seeds and 4,096 particles per seed.
# Fresh affine projections are computed at every sampler node. Full-score sampling uses checked spatial
# lookup tables from the actual checkpoints, with direct inference for any points outside the table.
# KL uses the existing target-quantile estimator and caches per-seed counts for 32, 64 and 128 bins.
#
# The Gaussian theory remains a surrogate even for the affine-error arm: that arm still contains the
# nonlinear exact mixture score. The notebook tests benefit and agreement; it does not assume either.
# Settings and checkpoint/source hashes identify each result directory, so incompatible cached runs cannot mix.
# See [057-schedule-review.md](057-schedule-review.md) for corrections and the executed results.

# %%
from pathlib import Path
from io import BytesIO
from html import escape
import os, csv, json, time, math, sys, hashlib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr
from IPython.display import display, HTML, Image

# ------------------------------- configuration -------------------------------
n_steps = 500                  # base EDM rho=7 intervals; shared refinement is added
n_seeds, n_particles = 4, 4096 # pooled for KL, as in 056
n_bins, pseudocount = 64, 0.5  # target-quantile bins
max_step_contraction = 0.1     # substep when (1+gamma)*dV/(2*v_loc) exceeds this
stiffness_aware_substeps = True  # v_loc = narrowest component variance + sigma^2; False: v_loc = V (global)
arms_requested = ('exact', 'affine', 'full')  # required arms; no silent fallback
checkpoint_filter = lambda run, epoch: True    # e.g. lambda run, epoch: epoch >= 20
include_tailored = True        # per-checkpoint theory-optimal window
run_sampling = True            # False: theory predictions only
force_recompute = False
seeds = [170, 271, 372, 473]    # same independent sampling replicate seeds as 056
assert len(seeds) == n_seeds
workers = 4
inference_device = 'cpu'       # table inference is efficient on CPU; use 'cuda:6' if changing this
if inference_device not in ('cpu', 'cuda:6'):
    raise ValueError('Only CPU or the requested GPU 6 may be used.')
floor_factor = 3.0             # 'above floor' = KL > floor_factor * exact-score KL (as in 056)

relative_dir = Path('paper_odds')
if 'SCHEDULE_NB_ROOT' in os.environ:            # optional override (used for testing)
    root = Path(os.environ['SCHEDULE_NB_ROOT'])
else:
    root = next(p for p in (Path.cwd(), *Path.cwd().parents)
                if (p / relative_dir / 'outputs/score_error_modes').is_dir())
paper = root / relative_dir
if os.environ.get('SCHEDULE_NB_QUICK'):          # smoke-test settings
    n_steps, n_seeds, n_particles = 120, 2, 512
    seeds = seeds[:n_seeds]
    workers = 2
    checkpoint_filter = lambda run, epoch: epoch in (5, 30)
out_base = paper / f'outputs/schedule_experiment_euler{n_steps}'
out_base.mkdir(parents=True, exist_ok=True)
out_dir = out_base
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from paper_odds.score_error_analysis.schedule_experiment import (
    gamma_of, predict_schedule, split_profile, cell_response, common_clock, schedule_matrix,
    prepare_cache, content_hash, run_checkpoint, run_exact, result_rows, write_csv,
)
from paper_odds.score_error_analysis.score_error_modes import GaussianMixture1D
from paper_odds.score_error_analysis.residual_ablation import binned_kl

def read_table(path, columns=None):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, usecols=columns,
                                       dtype=None, encoding='utf-8')).view(np.recarray)

figure_index = 0

def show_figure(fig):
    global figure_index
    figure_index += 1
    fig.savefig(out_dir / f'figure_{figure_index:02d}.png', dpi=150, bbox_inches='tight')
    with BytesIO() as buffer:
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
    plt.close(fig)

def show_table(rows):
    if not rows:
        print('(empty table)'); return
    def fmt(v):
        return f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(v)
    header = ''.join(f'<th>{escape(str(k))}</th>' for k in rows[0])
    body = ''.join('<tr>' + ''.join(f'<td>{escape(fmt(v))}</td>' for v in r.values()) + '</tr>' for r in rows)
    display(HTML('<div style="overflow:auto"><table style="border-spacing:12px 4px">'
                 f'<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'))

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.2})
print('Result parent:', out_base, '; fingerprinted directory is set after schedule selection.')

# %% [markdown]
# ## Data, checkpoints and clock
#
# Load the unchanged mixture and cached affine profiles. Mixture weights are stored as `probabilities`;
# means, variances, coordinate identities and every checkpoint hash are checked before sampling.
# The affine coefficients are reprojected at the new integration nodes when running each checkpoint.
# The preliminary theory curves below use the existing dense profile grid. The sampling results also save
# predictions from the newly projected profiles for a direct agreement check.

# %%
MIXTURE_OVERRIDE = None   # e.g. dict(weights=[...], means=[...], stds=[...])

summary = read_table(paper / 'outputs/score_error_modes/checkpoint_summary.csv', ['run', 'epoch', 'checkpoint'])
with np.load(paper / 'outputs/score_error_modes/clock.npz') as saved:
    prof_sigma = saved['sigma'].astype(float).copy()
    prof_variance = saved['variance'].astype(float).copy()
meta = json.loads((paper / 'outputs/score_error_modes/metadata.json').read_text())
mu0 = float(meta['reference_mean'])
V_ref = float(meta['reference_variance'])
sigma0_sq = float(np.median(prof_variance - prof_sigma**2))
sigma0 = math.sqrt(sigma0_sq)
sigma_min, sigma_max = float(prof_sigma.min()), float(prof_sigma.max())

checkpoints = [(str(r.run), int(r.epoch)) for r in summary if checkpoint_filter(str(r.run), int(r.epoch))]
profiles = {}
for run, epoch in checkpoints:
    with np.load(paper / 'outputs/score_error_modes/profiles' / f'{run}_epoch{epoch:02d}.npz') as saved:
        profiles[(run, epoch)] = {k: saved[k].astype(float).copy() for k in ('b', 'C', 'a_linear', 'u_linear')}

def resolve_mixture(meta):
    if MIXTURE_OVERRIDE is not None:
        m = MIXTURE_OVERRIDE
    else:
        m = next((meta[k] for k in ('mixture', 'data', 'dataset', 'target', 'data_distribution')
                  if isinstance(meta.get(k), dict)), meta)
    def pick(*names):
        for name in names:
            if name in m:
                return np.atleast_1d(np.asarray(m[name], dtype=float))
        return None
    w = pick('probabilities', 'weights', 'mixture_weights', 'pi', 'probs', 'proportions')
    mu = pick('means', 'mixture_means', 'mu', 'locs', 'centers')
    sd = pick('stds', 'mixture_stds', 'sigmas', 'scales', 'std')
    var = pick('variances', 'mixture_variances', 'vars')
    if sd is None and var is not None:
        sd = np.sqrt(var)
    if w is None or mu is None or sd is None:
        raise KeyError('Mixture parameters not found in metadata.json; '
                       'set MIXTURE_OVERRIDE = dict(weights=..., means=..., stds=...).')
    if w.shape != mu.shape or sd.shape != mu.shape or np.any(w <= 0) or np.any(sd <= 0):
        raise ValueError('Invalid mixture parameters.')
    return w / w.sum(), mu, sd

mix_w, mix_mu, mix_sd = resolve_mixture(meta)
mix_mean = float(np.sum(mix_w * mix_mu))
mix_var = float(np.sum(mix_w * (mix_sd**2 + mix_mu**2)) - mix_mean**2)
print(f'{len(checkpoints)} checkpoints; mixture with {len(mix_w)} components')
print(f'mixture mean {mix_mean:.5f} (reference {mu0:.5f}); variance {mix_var:.5f} (clock sigma0^2 {sigma0_sq:.5f})')

np.testing.assert_allclose([mix_mean, mix_var], [mu0, sigma0_sq], rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(V_ref, mix_var+sigma_min**2)
np.testing.assert_allclose(mix_w, meta['mixture']['probabilities'])
np.testing.assert_allclose(mix_mu, meta['mixture']['means'])
np.testing.assert_allclose(mix_sd, meta['mixture']['scales'])
mixture = GaussianMixture1D(tuple(mix_mu), tuple(mix_sd), tuple(mix_w))
if not checkpoints:
    raise ValueError('No checkpoints selected.')
records = []
for row in summary:
    cp = (str(row.run), int(row.epoch))
    if cp not in profiles:
        continue
    with np.load(paper/'outputs/score_error_modes/profiles'/f'{cp[0]}_epoch{cp[1]:02d}.npz') as saved:
        expected_hash = str(saved['checkpoint_sha256'])
    path = root / str(row.checkpoint)
    if not path.is_file() or content_hash(path) != expected_hash:
        raise ValueError(f'Checkpoint missing or changed: {path}')
    records.append(dict(run=cp[0], epoch=cp[1], checkpoint=str(row.checkpoint), checkpoint_sha256=expected_hash))
    np.testing.assert_allclose(profiles[cp]['a_linear'], prof_variance*profiles[cp]['C'])
    np.testing.assert_allclose(profiles[cp]['u_linear'], prof_variance*profiles[cp]['b']/np.sqrt(V_ref))
Lambda = np.log((sigma0_sq+sigma_max**2)/V_ref)
print(f'Validated {len(records)} checkpoint hashes; σ0={sigma0:.6f}, Λ={Lambda:.6f}.')


# %% [markdown]
# ## Candidate schedules and noise budgets
#
# The original window edges are retained for review. The three `matched_*` schedules instead have identical
# height γ=5 and identical log-variance width. Their integrated stochasticity ∫γ dℓ is equal.
# Their common integration grid also gives equal numbers of score evaluations. Population and tailored
# windows are selected next using the signed score profiles alone. Constant competitors include γ=0.

# %%
def const(g):
    return dict(kind='const', gamma=float(g))

def window(g, lo, hi):
    return dict(kind='window', gamma=float(g), lo=float(lo), hi=float(hi))

s0 = sigma0
SCHEDULES = {
    'ode':      const(0.0),
    'c0.05':    const(0.05),
    'c0.2':     const(0.2),
    'c1':       const(1.0),
    'c5':       const(5.0),
    'low5':     window(5.0, 0.0, 0.86 * s0),
    'mid5':     window(5.0, 0.86 * s0, 2.6 * s0),
    'high5':    window(5.0, 2.6 * s0, np.inf),
    'mid1':     window(1.0, 0.75 * s0, 9.5 * s0),
    'burst50':  window(50.0, 1.03 * s0, 1.29 * s0),
}
CONSTANTS = ['ode', 'c0.05', 'c0.2', 'c1', 'c5']
LABELS = {'ode': 'ODE', 'c0.05': 'γ=0.05', 'c0.2': 'γ=0.2', 'c1': 'γ=1', 'c5': 'γ=5',
          'low5': 'low window γ=5', 'mid5': 'mid window γ=5', 'high5': 'high window γ=5',
          'mid1': 'wide window γ=1', 'burst50': 'burst γ=50',
          'pop5': 'population window γ=5', 'tail5': 'tailored window γ=5'}

def describe(spec):
    if spec['kind'] == 'const':
        return f"gamma = {spec['gamma']:g} everywhere"
    hi = '∞' if not np.isfinite(spec['hi']) else f"{spec['hi']:.3g}"
    return (f"gamma = {spec['gamma']:g} on sigma in [{spec['lo']:.3g}, {hi}] "
            f"= [{spec['lo'] / s0:.2f}, {spec['hi'] / s0:.2f}] sigma0; ODE elsewhere")

for key, spec in SCHEDULES.items():
    print(f'{key:8s} {describe(spec)}')

# Equal-height, equal-width windows in distance d=log(V/Vref).
def sigma_at_distance(d):
    return float(np.sqrt(max(V_ref*np.exp(d)-sigma0_sq, sigma_min**2)))
d_low = np.log((sigma0_sq+(0.86*s0)**2)/V_ref)
d_high = np.log((sigma0_sq+(2.6*s0)**2)/V_ref)
width = d_low
center = (d_low+d_high)/2
for key, dlo, dhi, label in [
    ('matched_low5', 0., width, 'equal-budget low γ=5'),
    ('matched_mid5', center-width/2, center+width/2, 'equal-budget mid γ=5'),
    ('matched_high5', d_high, d_high+width, 'equal-budget high γ=5')]:
    SCHEDULES[key] = window(5., sigma_at_distance(dlo), sigma_at_distance(dhi))
    LABELS[key] = label


# %% [markdown]
# ## First-order and Gaussian-moment predictions
#
# For η(d)=∫₀ᵈγ(e) de, the response kernels are
# $$L=\int(1+\gamma)e^{-\eta}a\,dd,\qquad
# M=\int\frac{1+\gamma}{2}e^{-(d+\eta)/2}u\,dd,\qquad \widehat h=L^2/4+M^2/2.$$
# Every schedule edge is inserted into the profile integration grid. Within each refined cell, coefficients
# are frozen at its midpoint and the response and Gaussian moment equations are advanced analytically.
# This avoids missing or widening narrow bursts. It is a convergent numerical approximation to the
# measured time-varying profiles, not an exact solution for those profiles.
#
# The population window minimizes the median predicted log KL ratio over the selected checkpoints.
# Tailored windows minimize the same quantity separately. Neither uses empirical sampled KL.
# Both are in-sample uses of profile information; “computed before sampling” does not make them held-out tests.

# %%
# --- theory-selected windows (gamma = 5) ---
edges = np.exp(np.linspace(np.log(0.05 * s0), np.log(40 * s0), 26))
cand = [(lo, hi) for i, lo in enumerate(edges) for hi in edges[i + 1:]]
candidate_specs = [window(5., lo, hi) for lo, hi in cand]
# Each candidate has the same integration grid for every checkpoint. Build it once.
profile_distance = np.log((sigma0_sq+prof_sigma**2)/(sigma0_sq+prof_sigma[0]**2))
zero_profile = np.zeros_like(prof_sigma)
def response_template(spec):
    grid, _, _, gamma = split_profile(prof_sigma,zero_profile,zero_profile,sigma0_sq,spec)
    return grid,(grid[:-1]+grid[1:])/2,gamma
candidate_templates = [response_template(spec) for spec in candidate_specs]
ode_template = response_template(const(0))
def from_template(a,u,template):
    grid,midpoint,gamma = template
    return cell_response(grid,np.interp(midpoint,profile_distance,a),np.interp(midpoint,profile_distance,u),gamma)[2]
log_gain = np.empty((len(checkpoints), len(cand)))
for i, cp in enumerate(checkpoints):
    a, u = profiles[cp]['a_linear'], profiles[cp]['u_linear']
    h0 = from_template(a,u,ode_template)
    for j, template in enumerate(candidate_templates):
        log_gain[i,j] = np.log10(from_template(a,u,template)/h0)
j_pop = int(np.argmin(np.median(log_gain, axis=0)))
SCHEDULES['pop5'] = window(5.0, *cand[j_pop])
print('population window:', describe(SCHEDULES['pop5']),
      f'; predicted median log10 gain {np.median(log_gain[:, j_pop]):+.3f}')
tailored = {}
if include_tailored:
    for i, cp in enumerate(checkpoints):
        tailored[cp] = window(5.0, *cand[int(np.argmin(log_gain[i]))])
    SCHEDULES['tail5'] = dict(kind='tailored', gamma=5.0)
    lows = np.array([t['lo'] for t in tailored.values()]) / s0
    highs = np.array([t['hi'] for t in tailored.values()]) / s0
    print(f'tailored windows: median [{np.median(lows):.2f}, {np.median(highs):.2f}] sigma0; '
          f'lower edge IQR [{np.percentile(lows, 25):.2f}, {np.percentile(lows, 75):.2f}] sigma0')

def spec_for(key, cp):
    return tailored[cp] if SCHEDULES[key]['kind'] == 'tailored' else SCHEDULES[key]

# --- predictions for every checkpoint and schedule ---
predictions = {}
for cp in checkpoints:
    for key in SCHEDULES:
        predictions[(cp, key)] = predict_schedule(prof_sigma, profiles[cp]['a_linear'], profiles[cp]['u_linear'],
                                                  sigma0_sq, spec_for(key, cp), refinement=2)

def predicted_log_ratio(cp, key, kind='first_order'):
    return math.log10(predictions[(cp, key)][kind] / predictions[(cp, 'ode')][kind])


specs_by_cp = {cp: {key: spec_for(key, cp) for key in SCHEDULES} for cp in checkpoints}
all_specs = [spec for specs in specs_by_cp.values() for spec in specs.values()]
# Deduplication reduces grid construction work but does not affect schedule order.
unique_specs = list({json.dumps(spec, sort_keys=True): spec for spec in all_specs}.values())
clock = common_clock(mixture, sigma_min, sigma_max, n_steps, unique_specs,
                     max_step_contraction, stiffness_aware_substeps)
source_names = ['schedule_experiment.py', 'residual_ablation.py', 'checkpoint_score.py',
                'fit_checkpoint_modes.py', 'score_error_modes.py']
config = dict(schema_version=1, root=str(root.resolve()), mixture=meta['mixture'],
    sigma_min=sigma_min, sigma_max=sigma_max, base_steps=n_steps, clock_ell=clock['ell'].tolist(),
    seeds=seeds, particles=n_particles, bins=[32,64,128], pseudocount=pseudocount,
    max_step_contraction=max_step_contraction, stiffness_aware_substeps=stiffness_aware_substeps,
    integrator='Euler/EM in standardized log-variance coordinates; common refined EDM rho=7 grid',
    lambdas=[0.,1.], inference_device=inference_device, quadrature_order=256,
    table_nodes=1025, table_tolerance=.003, z_limit=12., profile_fingerprint=meta['fingerprint'],
    checkpoints=records, schedules={f'{cp[0]}:{cp[1]}': specs for cp, specs in specs_by_cp.items()},
    sources={name:content_hash(paper/'score_error_analysis'/name) for name in source_names},
    profile_hashes={f'{r}_epoch{e:02d}':content_hash(paper/'outputs/score_error_modes/profiles'/f'{r}_epoch{e:02d}.npz') for r,e in checkpoints})
out_dir, fingerprint = prepare_cache(out_base, config, force=force_recompute)
print('Validated cache directory:', out_dir)
print(f'Common integration grid: {len(clock["ell"])-1} intervals from {n_steps} base intervals.')
np.savez_compressed(out_dir/'clock.npz', **clock)
schedule_budgets = []
for key in SCHEDULES:
    budgets = [float(schedule_matrix(clock, mixture, [spec_for(key, cp)])[0] @ np.diff(clock['ell'])) for cp in checkpoints]
    schedule_budgets.append(dict(schedule=key, label=LABELS[key], min_integrated_gamma=min(budgets),
                                 max_integrated_gamma=max(budgets), score_evaluations=len(clock['ell'])-1))
write_csv(out_dir/'schedule_budgets.csv', schedule_budgets)
show_table(schedule_budgets)
write_csv(out_dir/'pre_sampling_predictions.csv', [dict(run=cp[0],epoch=cp[1],schedule=key,**predictions[(cp,key)])
                                                  for cp in checkpoints for key in SCHEDULES])

rows = []
for key in SCHEDULES:
    if key == 'ode':
        continue
    lr = np.array([predicted_log_ratio(cp, key) for cp in checkpoints])
    lm = np.array([predicted_log_ratio(cp, key, 'moment_q_p') for cp in checkpoints])
    rows.append(dict(schedule=LABELS[key], predicted_median_log10_ratio=float(np.median(lr)),
                     predicted_win_rate=float(np.mean(lr < 0)),
                     moment_median_log10_ratio=float(np.median(lm))))
display(HTML('<p><b>Predictions before sampling</b> (Gaussian surrogate, affine error; ratio to the ODE):</p>'))
show_table(rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout='constrained')
plot_keys = [k for k in SCHEDULES if SCHEDULES[k]['kind'] != 'tailored']
for row, key in enumerate(plot_keys):
    spec = SCHEDULES[key]
    lo, hi = (sigma_min, sigma_max) if spec['kind'] == 'const' else (max(spec['lo'], sigma_min), min(spec['hi'], sigma_max))
    if spec['gamma'] > 0:
        axes[0].hlines(row, lo, hi, lw=2 + 2 * math.log10(1 + spec['gamma']), color='tab:blue')
        axes[0].text(math.sqrt(lo * hi), row - 0.25, f"γ={spec['gamma']:g}", ha='center', fontsize=7)
axes[0].axvline(s0, color='k', ls=':', lw=1)
axes[0].set_xscale('log')
axes[0].set_xlim(sigma_min, sigma_max)
axes[0].set_yticks(range(len(plot_keys)), [LABELS[k] for k in plot_keys])
axes[0].set(xlabel=r'$\sigma$', title=r'Where each schedule is stochastic (dotted: $\sigma_0$; ODE elsewhere)')
axes[0].invert_yaxis()
keys = [k for k in SCHEDULES if k != 'ode']
data = [[predicted_log_ratio(cp, k) for cp in checkpoints] for k in keys]
axes[1].boxplot(data, vert=False, showfliers=False)
axes[1].set_yticks(range(1, len(keys) + 1), [LABELS[k] for k in keys])
axes[1].axvline(0, color='k', lw=0.8)
axes[1].set(xlabel=r'predicted $\log_{10}(\hat h/\hat h_{\rm ODE})$', title='First-order predictions per checkpoint')
show_figure(fig)

# %% [markdown]
# ## Checkpoint loader and paired sampler
#
# The helper uses `CheckpointScore`, which strictly loads the production checkpoint by run and epoch,
# including the original custom EDM log-noise conditioning (½ log σ), σ_data=0.5 and float64 inference.
# Fresh Gauss–Hermite projections and learned-score lookup tables are constructed on the shared sampler grid.
# Spatial table error is checked relative to score-error RMS (tolerance 0.003); out-of-range particles use
# direct inference rather than clipping. The retained table diagnostics are inspected below.
#
# The integration update is the one from `residual_ablation.sample_paired`, with a gamma value per interval.
# Schedule edges are grid nodes and gamma is evaluated inside each interval. There are no schedule-specific
# substep RNGs: every schedule/arm/checkpoint/control consumes the same per-seed Brownian sequence.

# %%
# Explicit loader for interactive inspection; the worker uses this same class.
from paper_odds.score_error_analysis.checkpoint_score import CheckpointScore
record_lookup = {(r['run'], r['epoch']): r for r in records}
def load_network_score(run, epoch):
    record = record_lookup[(run, epoch)]
    model = CheckpointScore(root/record['checkpoint'], device=inference_device)
    if model.epoch != epoch:
        raise ValueError('Requested epoch does not match checkpoint.')
    return model

if set(arms_requested) != {'exact','affine','full'}:
    raise ValueError('This validated comparison requires all three score arms.')
arms = list(arms_requested)
# Fail before dispatching sampling if the checkpoint inference dependency is unavailable.
probe = load_network_score(*checkpoints[0])
probe_value = probe(np.array([mu0]), np.array([sigma0]))
assert np.isfinite(probe_value).all()
del probe
print('Full learned-score arm connected. Inference device:', inference_device)

# %% [markdown]
# ## Run and resume
#
# Each checkpoint cache is written atomically and contains per-seed histograms, actual step count,
# projection coefficients, both theoretical predictions and interpolation diagnostics. Configuration and
# source/checkpoint hashes isolate incompatible runs automatically. Exact-score controls are cached once
# per distinct schedule, with checkpoint-specific controls for tailored windows. All three arms must finish
# before results are summarized; partial or missing rows are never counted as losses.
#
# The default is the full 144-checkpoint experiment. `SCHEDULE_NB_QUICK=1` selects a separate smoke run
# (epochs 5 and 30, 2 seeds × 512 particles, 120 base steps). It never populates the production cache.
# Use the project's PyTorch environment; the notebook also requires pandas, NumPy, SciPy and Matplotlib.

# %%
results_path = out_dir/'results.csv'
exact_specs = {}
for key, spec in SCHEDULES.items():
    if spec['kind'] != 'tailored':
        exact_specs[f'exact_score|-1|{key}'] = spec
for cp, spec in tailored.items():
    exact_specs[f'{cp[0]}|{cp[1]}|tail5'] = spec

if run_sampling:
    started = time.time()
    print(f'Running exact controls ({len(exact_specs)} schedule entries)...', flush=True)
    run_exact(config, clock, exact_specs, out_dir, fingerprint)
    pending = [record for record in records if not (out_dir/f"{record['run']}_epoch{record['epoch']:02d}.npz").exists()]
    # CPU table inference is the default; a requested cuda:6 run uses one worker.
    effective_workers = workers if inference_device == 'cpu' else 1
    print(f'{len(pending)} uncached checkpoints; {effective_workers} workers.', flush=True)
    if effective_workers == 1:
        for i, record in enumerate(pending,1):
            cp = (record['run'],record['epoch'])
            result = run_checkpoint(record,config,clock,list(specs_by_cp[cp].values()),out_dir,fingerprint)
            print(f'{i}/{len(pending)}: {result}',flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(run_checkpoint,record,config,clock,
                                  list(specs_by_cp[(record['run'],record['epoch'])].values()),out_dir,fingerprint)
                       for record in pending]
            for i,future in enumerate(as_completed(futures),1):
                result = future.result()
                print(f'{i}/{len(pending)}: {result["run"]}:{result["epoch"]}, '
                      f'{result.get("seconds",0):.1f}s; elapsed {time.time()-started:.0f}s',flush=True)
    all_rows = result_rows(records,config,specs_by_cp,out_dir,fingerprint)
    (out_dir/'COMPLETE.json').write_text(json.dumps(dict(fingerprint=fingerprint,
        checkpoints=len(records), rows=len(all_rows), seconds=time.time()-started),indent=2)+'\n')
    print(f'Complete: {len(records)} checkpoints, {time.time()-started:.1f}s; {results_path}',flush=True)
else:
    if not (out_dir/'COMPLETE.json').exists():
        raise RuntimeError('No complete cache for these settings. Set run_sampling=True, or execute only through the theory cell.')

# %% [markdown]
# ## Results
#
# For each arm, schedule and KL direction the table reports the median of $\log_{10}[\mathrm{KL}(\text{schedule})/\mathrm{KL}(\text{ODE})]$ over checkpoints, the fraction of checkpoints where the schedule beats the ODE, and the same fraction restricted to checkpoints above the exact-score floor (both KLs larger than `floor_factor` times the exact-score KL of the same schedule). The bounded statistic $(h-h_{\rm ODE})/(h+h_{\rm ODE})\in[-1,1]$ is robust to tiny ODE KLs. The last two columns compare the measured sign with the first-order prediction.

# %%
res = {}
with open(results_path) as f:
    for row in csv.DictReader(f):
        if int(row['bins']) != n_bins:
            continue
        res[(row['arm'], row['run'], int(row['epoch']), row['schedule'])] = (float(row['kl_q_p']), float(row['kl_p_q']))
for cp in checkpoints:
    for arm in ('affine','full'):
        for key in SCHEDULES:
            if (arm,cp[0],cp[1],key) not in res:
                raise ValueError(f'Missing result: {arm}, {cp}, {key}')
directions = {'q_p': 0, 'p_q': 1}
dir_label = {'q_p': 'KL(q||p)', 'p_q': 'KL(p||q)'}

def kl(arm, cp, key, direction):
    return res.get((arm, cp[0], cp[1], key), (np.nan, np.nan))[directions[direction]]

def floor(cp, key, direction):
    if key == 'tail5':
        return kl('exact', cp, key, direction)
    return res.get(('exact', 'exact_score', -1, key), (np.nan, np.nan))[directions[direction]]

def log_ratio(arm, cp, key, direction):
    return math.log10(kl(arm, cp, key, direction) / kl(arm, cp, 'ode', direction))

def above_floor(cp, key, direction, arm):
    return (kl(arm, cp, key, direction) > floor_factor * floor(cp, key, direction) and
            kl(arm, cp, 'ode', direction) > floor_factor * floor(cp, 'ode', direction))

rows = []
for arm in [a for a in arms if a != 'exact']:
    for direction in directions:
        for key in SCHEDULES:
            if key == 'ode':
                continue
            lr = np.array([log_ratio(arm, cp, key, direction) for cp in checkpoints])
            ok = np.isfinite(lr)
            if not ok.any():
                continue
            h, h0 = (np.array([kl(arm, cp, k, direction) for cp in checkpoints]) for k in (key, 'ode'))
            bounded = (h - h0) / (h + h0)
            above = np.array([above_floor(cp, key, direction, arm) for cp in checkpoints])
            pred = np.array([predicted_log_ratio(cp, key) for cp in checkpoints])
            agree = np.mean(np.sign(pred[ok]) == np.sign(lr[ok]))
            rho = spearmanr(pred[ok], lr[ok]).statistic if ok.sum() > 2 else np.nan
            rows.append(dict(arm=arm, KL=dir_label[direction], schedule=LABELS[key],
                             median_log10_ratio=float(np.median(lr[ok])),
                             q25=float(np.percentile(lr[ok], 25)), q75=float(np.percentile(lr[ok], 75)),
                             win_rate=float(np.mean(lr[ok] < 0)),
                             win_rate_above_floor=float(np.mean(lr[ok & above] < 0)) if (ok & above).any() else np.nan,
                             n_above_floor=int((ok & above).sum()),
                             median_bounded=float(np.median(bounded[ok])),
                             sign_agreement=float(agree), spearman_vs_theory=float(rho)))
show_table(rows)
write_csv(out_dir/'schedule_summary.csv', rows)

floor_rows = []
for key in SCHEDULES:
    if SCHEDULES[key]['kind'] == 'tailored':
        continue
    fq, fp = res.get(('exact', 'exact_score', -1, key), (np.nan, np.nan))
    floor_rows.append(dict(schedule=LABELS[key], exact_KL_q_p=fq, exact_KL_p_q=fp))
display(HTML('<p><b>Exact-score floor</b> (discretization + sampling error of each schedule):</p>'))
show_table(floor_rows)
write_csv(out_dir/'exact_floor.csv', floor_rows)

# %% [markdown]
# ### Main test: windows vs. constant $\gamma$
#
# For each window schedule, the next table compares it per checkpoint with **each fixed constant $\gamma$**, with the **best fixed constant** (the single constant with the best median over checkpoints, chosen on the same data and therefore favourable to constant $\gamma$), and with the **per-checkpoint oracle constant** (the best constant for each checkpoint separately, an in-sample upper bound for constant $\gamma$). Negative medians mean the window is better.
#
# The tested constant set includes the ODE. These in-sample benchmarks use the median **log ratio to each checkpoint’s ODE**, and the oracle covers only the finite tested gamma grid; neither is a bound over all constant gamma values.

# %%
window_keys = [k for k in ('mid5', 'burst50', 'pop5', 'tail5', 'mid1', 'high5', 'low5', 'matched_low5', 'matched_mid5', 'matched_high5') if k in SCHEDULES]
main_rows = []
for arm in [a for a in arms if a != 'exact']:
    for direction in directions:
        const_lr = {c: np.array([log_ratio(arm, cp, c, direction) for cp in checkpoints]) for c in CONSTANTS}
        best_fixed = min(CONSTANTS, key=lambda c: np.nanmedian(const_lr[c]))
        oracle = np.nanmin(np.vstack([const_lr[c] for c in CONSTANTS]), axis=0)
        for key in window_keys:
            w = np.array([log_ratio(arm, cp, key, direction) for cp in checkpoints])
            row = dict(arm=arm, KL=dir_label[direction], window=LABELS[key],
                       best_fixed_constant=LABELS[best_fixed],
                       beats_best_fixed=float(np.nanmean(w < const_lr[best_fixed])),
                       median_diff_vs_best_fixed=float(np.nanmedian(w - const_lr[best_fixed])),
                       beats_oracle_constant=float(np.nanmean(w < oracle)),
                       median_diff_vs_oracle=float(np.nanmedian(w - oracle)))
            for c in CONSTANTS:
                row[f'beats {LABELS[c]}'] = float(np.nanmean(w < const_lr[c]))
            main_rows.append(row)
show_table(main_rows)
write_csv(out_dir/'constant_comparisons.csv', main_rows)

# %%
plot_arms = [a for a in arms if a != 'exact']
fig, axes = plt.subplots(len(plot_arms), 2, figsize=(14, 4.2 * len(plot_arms)), squeeze=False, layout='constrained')
keys = [k for k in SCHEDULES if k != 'ode']
for i, arm in enumerate(plot_arms):
    for j, direction in enumerate(directions):
        ax = axes[i, j]
        data = [[log_ratio(arm, cp, k, direction) for cp in checkpoints] for k in keys]
        ax.boxplot(data, vert=False, showfliers=False)
        for pos, values in enumerate(data, 1):
            ax.scatter(values, pos + 0.18 * (np.random.default_rng(pos).random(len(values)) - 0.5),
                       s=6, alpha=0.35, color='tab:blue')
        pred = [np.median([predicted_log_ratio(cp, k) for cp in checkpoints]) for k in keys]
        ax.scatter(pred, range(1, len(keys) + 1), marker='D', color='tab:red', s=25, zorder=5,
                   label='median first-order prediction')
        ax.set_yticks(range(1, len(keys) + 1), [LABELS[k] for k in keys])
        ax.axvline(0, color='k', lw=0.8)
        ax.set(xlabel='log10(KL / ODE KL)', title=f'{arm} score, {dir_label[direction]}')
        ax.legend(fontsize=8, loc='lower right')
show_figure(fig)

# %% [markdown]
# ### Paired view and theory agreement
#
# Left: per checkpoint, the mid window against the best fixed constant (points below the diagonal favour the window). Right: predicted vs. measured log ratio for the placement contrast (low / mid / high windows), which mixes placement and total injected stochasticity for the original windows. The equal-budget windows below provide a cleaner placement comparison.

# %%
direction = 'q_p'
fig, axes = plt.subplots(len(plot_arms), 2, figsize=(12, 4.6 * len(plot_arms)), squeeze=False, layout='constrained')
for i, arm in enumerate(plot_arms):
    const_lr = {c: np.array([log_ratio(arm, cp, c, direction) for cp in checkpoints]) for c in CONSTANTS}
    best_fixed = min(CONSTANTS, key=lambda c: np.nanmedian(const_lr[c]))
    w = np.array([log_ratio(arm, cp, 'mid5', direction) for cp in checkpoints])
    ax = axes[i, 0]
    ax.scatter(const_lr[best_fixed], w, s=14, alpha=0.6)
    lim = [np.nanmin([w, const_lr[best_fixed]]), np.nanmax([w, const_lr[best_fixed]])]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set(xlabel=f'{LABELS[best_fixed]}: log10(KL/ODE)', ylabel='mid window γ=5: log10(KL/ODE)',
           title=f'{arm}: window vs best fixed constant ({np.nanmean(w < const_lr[best_fixed]):.0%} below diagonal)')
    ax = axes[i, 1]
    for key, color in [('low5', 'tab:red'), ('mid5', 'tab:green'), ('high5', 'tab:gray'), ('burst50', 'tab:purple')]:
        x = np.array([predicted_log_ratio(cp, key) for cp in checkpoints])
        y = np.array([log_ratio(arm, cp, key, direction) for cp in checkpoints])
        ax.scatter(x, y, s=12, alpha=0.6, color=color, label=LABELS[key])
    ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
    ax.set(xlabel='predicted log10 ratio (first order)', ylabel='measured log10 ratio',
           title=f'{arm}: placement contrast, {dir_label[direction]}')
    ax.legend(fontsize=8)
show_figure(fig)


# %% [markdown]
# ## Decision summary

# %%
def summarize(arm, direction='q_p'):
    const_lr = {c: np.array([log_ratio(arm, cp, c, direction) for cp in checkpoints]) for c in CONSTANTS}
    best_fixed = min(CONSTANTS, key=lambda c: np.nanmedian(const_lr[c]))
    lines = [f'[{arm} score, {dir_label[direction]}] best fixed constant: {LABELS[best_fixed]} '
             f'(median log10 ratio {np.nanmedian(const_lr[best_fixed]):+.3f}, beats ODE in '
             f'{np.nanmean(const_lr[best_fixed] < 0):.0%})']
    for key in ('mid5', 'burst50', 'pop5', 'tail5'):
        if key not in SCHEDULES:
            continue
        w = np.array([log_ratio(arm, cp, key, direction) for cp in checkpoints])
        lines.append(f'  {LABELS[key]:24s} median {np.nanmedian(w):+.3f}; beats ODE {np.nanmean(w < 0):.0%}; '
                     f'beats best fixed constant {np.nanmean(w < const_lr[best_fixed]):.0%} '
                     f'(median diff {np.nanmedian(w - const_lr[best_fixed]):+.3f} dex)')
    placement = {k: np.nanmedian([log_ratio(arm, cp, k, direction) for cp in checkpoints])
                 for k in ('low5', 'mid5', 'high5')}
    ordered = placement['mid5'] < placement['high5'] and placement['mid5'] < placement['low5']
    lines.append('  placement medians: ' + ', '.join(f'{k} {v:+.3f}' for k, v in placement.items())
                 + (' -> mid window has the lowest measured median' if ordered else ' -> another placement has the lowest measured median'))
    return '\n'.join(lines)

for arm in plot_arms:
    for direction in directions:
        print(summarize(arm, direction))
        print()

# %% [markdown]
# ## Interpretation and sensitivity
#
# - **Affine-error arm:** Gaussian predictions are surrogates for nonlinear mixture dynamics, not guaranteed truths.
# - **Full learned score:** benefit here is the practical checkpoint-backed result; the nonlinear residual can change the ordering.
# - **Numerical floor:** above-floor filtering is a diagnostic, not a confidence test. All arms have the same actual step count.
# - **Selection:** population and tailored schedules use in-sample profiles. The best fixed and oracle constants use sampled KL and are also in-sample selections. None estimates out-of-sample performance without a separate evaluation design.
# - **Dependence:** repeated epochs from three training trajectories are not 144 independent networks. Report each run and the bin/seed sensitivity below.
# - **Scope:** mixture randomization and retraining remain deferred. No changes to the production integrators are needed for this experiment.

# %%
results_frame = pd.read_csv(results_path)
selected_results = results_frame[(results_frame.bins == n_bins) & (results_frame.arm != 'exact')].copy()
base = selected_results[selected_results.schedule == 'ode'][['arm','run','epoch','kl_q_p','kl_p_q']].rename(
    columns={'kl_q_p':'ode_q_p','kl_p_q':'ode_p_q'})
selected_results = selected_results.merge(base,on=['arm','run','epoch'],validate='many_to_one')
run_rows = []
for (arm, run, key), group in selected_results.groupby(['arm','run','schedule']):
    for direction in directions:
        lr = np.log10(group[f'kl_{direction}']/group[f'ode_{direction}'])
        run_rows.append(dict(arm=arm,run=run,schedule=key,direction=direction,n=len(group),
                             median_log_ratio=float(np.median(lr)),win_fraction=float(np.mean(lr<0))))
write_csv(out_dir/'by_training_run.csv',run_rows)
display(pd.DataFrame(run_rows).query("schedule in ['mid5','burst50','pop5','tail5']").round(4))

fig, axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
for ax,arm in zip(axes,['affine','full']):
    for key in ['matched_low5','matched_mid5','matched_high5']:
        vals = [log_ratio(arm,cp,key,'q_p') for cp in checkpoints]
        ax.hist(vals,bins=25,alpha=.4,label=LABELS[key])
    ax.axvline(0,color='black',ls=':');ax.set(title=f'{arm}: equal integrated γ',xlabel='log10(KL / ODE KL)',ylabel='Checkpoints')
    ax.legend(fontsize=8)
show_figure(fig)

diagnostic_rows, seed_rows, projection_rows = [], [], []
for cp in checkpoints:
    with np.load(out_dir/f'{cp[0]}_epoch{cp[1]:02d}.npz') as saved:
        diagnostic_rows.append(dict(run=cp[0],epoch=cp[1],actual_steps=int(saved['actual_steps']),
            table_nodes=int(saved['table_nodes']),table_error=float(saved['table_error']),
            fallback_points=int(saved['fallback_points']),seconds=float(saved['seconds'])))
        # Compare preliminary profile predictions with the fresh sampler-node projections.
        for j,key in enumerate(SCHEDULES):
            preliminary = predicted_log_ratio(cp,key)
            fresh = np.log10(saved['profile_kl'][j]/saved['profile_kl'][0])
            projection_rows.append(dict(run=cp[0],epoch=cp[1],schedule=key,preliminary=preliminary,fresh=fresh,
                                        difference=fresh-preliminary))
        counts = saved[f'counts_{n_bins}']
        leave = binned_kl(counts.sum(axis=2,keepdims=True)-counts,pseudocount)
        for j,key in enumerate(SCHEDULES):
            for k,arm in enumerate(['affine','full']):
                for direction,index in directions.items():
                    for seed_index,seed in enumerate(seeds):
                        seed_rows.append(dict(run=cp[0],epoch=cp[1],arm=arm,schedule=key,direction=direction,
                            deleted_seed=seed,win=bool(leave[j,k,seed_index,index]<leave[0,k,seed_index,index])))
write_csv(out_dir/'sampler_diagnostics.csv',diagnostic_rows)
write_csv(out_dir/'projection_prediction_check.csv',projection_rows)
projection_frame = pd.DataFrame(projection_rows)
print('Fresh-node minus cached-profile prediction differences (dex):')
display(projection_frame.groupby('schedule').difference.agg(['median',lambda x: np.max(np.abs(x))]).round(5))
seed_frame = pd.DataFrame(seed_rows)
seed_rates = seed_frame.groupby(['arm','schedule','direction','deleted_seed']).win.mean().reset_index()
seed_sensitivity = seed_rates.groupby(['arm','schedule','direction']).win.agg(['min','max']).reset_index()
seed_sensitivity.to_csv(out_dir/'seed_sensitivity.csv',index=False)
display(seed_sensitivity.query("schedule in ['mid5','burst50','pop5','tail5']").round(4))

bin_rows = []
for bins in config['bins']:
    data = results_frame[(results_frame.bins == bins) & (results_frame.arm != 'exact')]
    baseline = data[data.schedule=='ode'][['arm','run','epoch','kl_q_p','kl_p_q']].rename(columns={'kl_q_p':'ode_q_p','kl_p_q':'ode_p_q'})
    data = data.merge(baseline,on=['arm','run','epoch'],validate='many_to_one')
    for (arm,key),group in data.groupby(['arm','schedule']):
        for direction in directions:
            lr=np.log10(group[f'kl_{direction}']/group[f'ode_{direction}'])
            bin_rows.append(dict(bins=bins,arm=arm,schedule=key,direction=direction,
                                 win_fraction=float(np.mean(lr<0)),median_log_ratio=float(np.median(lr))))
write_csv(out_dir/'bin_sensitivity.csv',bin_rows)
print('Maximum learned-table relative RMS error:',max(row['table_error'] for row in diagnostic_rows))
print('Direct fallback evaluations:',sum(row['fallback_points'] for row in diagnostic_rows))
print('Results and per-seed counts:',out_dir)

# %% [markdown]
# ## Numerical resolution and fresh-profile agreement
#
# The separate paired diagnostic compares **869 and 1,738 intervals** on three representative checkpoints.
# Fine Brownian increments sum to the coarse increments, and all four replicate seeds are retained.
# Read its changes in log KL ratio alongside the size of each claimed benefit: the largest above-floor
# change is about **0.021 dex**, with larger shifts for near-floor ratios. This is a limited convergence
# diagnostic, not proof that every checkpoint is converged.
#
# The agreement table below uses projections freshly evaluated at the sampler nodes. Earlier predictions
# and window choices use the cached 288-node profile grid, and their differences are exported separately.
# Both remain Gaussian-surrogate predictions for this nonlinear mixture.

# %%
resolution_path = out_dir/'resolution_validation.csv'
if resolution_path.exists():
    resolution = pd.read_csv(resolution_path)
    selected = resolution[(resolution.arm!='exact') & (resolution.schedule!='ode')]
    display(selected.groupby(['arm','direction']).log_ratio_change.agg(
        median_abs=lambda x:np.median(abs(x)),max_abs=lambda x:np.max(abs(x))).round(5))
    above = selected[selected.above_floor]
    print(f'Largest above-floor refinement shift: {abs(above.log_ratio_change).max():.5f} dex; '
          f'all comparisons: {abs(selected.log_ratio_change).max():.5f} dex.')
else:
    print('Resolution validation has not been run for this fingerprint. See 057-schedule-review.md.')

fresh_baseline = selected_results[selected_results.schedule=='ode'][['arm','run','epoch','profile_kl']].rename(
    columns={'profile_kl':'profile_ode'})
fresh = selected_results.merge(fresh_baseline,on=['arm','run','epoch'],validate='many_to_one')
fresh_rows = []
for (arm,key),group in fresh.groupby(['arm','schedule']):
    if key=='ode':continue
    predicted=np.log10(group.profile_kl/group.profile_ode)
    for direction in directions:
        measured=np.log10(group[f'kl_{direction}']/group[f'ode_{direction}'])
        good=np.isfinite(predicted)&np.isfinite(measured)
        x,y=np.asarray(predicted)[good],np.asarray(measured)[good]
        fresh_rows.append(dict(arm=arm,schedule=key,direction=direction,n=int(good.sum()),
            log_RMSE=float(np.sqrt(np.mean((x-y)**2))),log_bias=float(np.mean(x-y)),
            sign_agreement=float(np.mean((x<0)==(y<0))),
            Spearman=float(spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else np.nan))
write_csv(out_dir/'fresh_profile_agreement.csv',fresh_rows)
display(pd.DataFrame(fresh_rows).query("schedule in ['mid5','burst50','pop5','tail5']").round(4))

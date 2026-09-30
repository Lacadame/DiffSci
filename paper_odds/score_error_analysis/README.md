# Least-squares score-error modes for the mixture checkpoints

This analysis evaluates all 144 production checkpoints in `default3`, `default4`,
and `default5` (48 epochs each), matching them to the existing KL measurements by
**(run, epoch)**. These are **one-dimensional** mixture models. The generic
`affine_projection` function accepts vector errors and matrix coefficients, but
the checkpoint loader and mixture evaluator here intentionally implement the
trained 1D experiment.

Run from the repository root with NumPy, SciPy, Matplotlib and PyTorch:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /opt/conda/envs/pytorch/bin/python -m paper_odds.score_error_analysis.fit_checkpoint_modes
```

The inference-only loader does not require Lightning or diffusers. It reproduces
the MLP, `residual=False`, EDM preconditioning with `sigma_data=0.5`, and the
`0.5*log(sigma)` time conditioning used in
`stochasticity_paper/scripts/test-time_profile-correlation.py`. Computations use
float64, including a cancellation-resistant expression for the score.

The analysis code, tests, documentation, and generated results live together
under `paper_odds`. Results default to `paper_odds/outputs/score_error_modes`.
The new cells at the end of `055-stoch_tests.ipynb` load and explore those results.

All generated figures, tables, and experiment caches live under `paper_odds/outputs/`,
which Git ignores. Keep the Python modules, tests, notebook `.py` sources, Markdown notes, and
reference PDFs in version control. Local `.ipynb` files are ignored. Back up `outputs/` separately: rebuilding the
checkpoint analyses requires `savedmodels/production/` and the original
`stochasticity_paper/stats/output_default{3,4,5}/results.npy` measurements.

Existing caches were moved without changing their contents. Their manifests retain
historical paths and source hashes as provenance. Path edits change the source
fingerprints used by some generators, so regenerating those experiments may require
a new `--output-dir` under `paper_odds/outputs/`; do not overwrite an old manifest
to bypass a configuration mismatch. Reading and plotting the moved results does
not require rerunning sampling.

## Notebook sources and rebuilding

The four notebooks in `paper_odds/` use the same Jupytext workflow as `geoworld1`:
Git tracks the adjacent `.py` source, and the `.ipynb` remains local with its
outputs. `# %%` marks code cells and `# %% [markdown]` marks prose cells.
Authoring metadata is preserved; execution outputs, counts, and timing metadata
are excluded from the text source.

From the repository root, install the optional tooling and rebuild missing notebooks:

```bash
python -m pip install -e ".[notebooks]"
python scripts/notebooks.py build
```

To install just the conversion tool into an existing environment, use
`python -m pip install "jupytext>=1.19,<2"`. Building never executes cells and
leaves existing notebooks, including local edits and outputs, untouched.

After editing and saving a notebook or its `.py` source, run:

```bash
python scripts/notebooks.py sync
```

Sync uses the newer file's inputs and preserves local notebook outputs. Sync
before switching between editors: independently editing both files can replace
edits in the older file. Commit the `.py` sources; keep the `.ipynb` files local.
A Jupytext-enabled Jupyter server uses `paper_odds/jupytext.toml` to save the pair
automatically. VS Code users can run the sync command explicitly.

The helper is scoped to `paper_odds/`, including new nested notebooks whose
companion `.py` files are eligible for Git. It skips `outputs/`, checkpoint
folders, and other ignored workspaces. Other DiffSci notebook directories keep
their existing tracking rules. Rebuilding produces notebooks without outputs;
running the experiments still requires the scientific environment and local data.

## Notebook 056: Euler with 500 steps

`../056-stoch_tests-wo_fits.ipynb` uses **Euler for the ODE and
Euler–Maruyama for the SDE, with 500 steps**. The original saved sweeps
already use these methods: all three `results.npy` files contain 501 time
nodes with initial step zero. They retain the original EDM noise clock,
zero endpoint, and recorded gamma=0.01 baseline.

The controlled residual experiment is regenerated in a separate cache:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/conda/envs/pytorch/bin/python -m paper_odds.score_error_analysis.run_residual_ablation \
  --integrator euler --steps 500 --workers 8 --no-plots \
  --output-dir paper_odds/outputs/residual_ablation_euler500
```

For this experiment Euler/Euler–Maruyama advances the existing standardized
log-variance clock, on 500 intervals from sigma=80 to sigma=0.002. Each step
uses the current drift and one shared Brownian increment, with no corrector.
Gamma=0 gives the Euler ODE. The exact-score control uses the same method,
clock, and step count. The Gaussian moment equations also use explicit Euler
on those 500 intervals; the signed-profile kernel prediction remains a
quadrature calculation. Thus the controlled experiment retains its own
positive endpoint, exact prior and target-partition KL estimator.

Sampler and moment integrators are included in the cache fingerprint, and
notebook setup checks both methods and the step count. The earlier cache
remains available to notebook 055. The refinement utility reads the selected
integrator from metadata; existing refinement results for the earlier cache
do not establish convergence of this new experiment.

## Notebook 056: piecewise RMS magnitude envelopes

The notebook compares continuous power laws with one, two, and three exponents
for RMS of the exact draft coordinates `u = (mu_theta - mu) / sqrt(Vref)` and
`a = log(alpha_theta)`. Exponents and breakpoints are fitted jointly within each
family, with nonnegative decay powers allowing plateaus. The same
`minimum_epoch` selection is used throughout.

`piecewise_fit_space = 'log'` selects fits by least squares in log magnitude;
`'magnitude'` selects least squares on the RMS itself. Both objectives are
computed, with trapezoidal weights in log noise, and both magnitude and log
errors are reported. The conditional valid-at-each-time RMS and the fixed cohort
valid throughout are kept separate. These are in-sample numerical fits, not
estimates of independent training replicates or evidence for Gaussian fluctuations.

The comparison figure has three columns: RMS of the mean coordinate, RMS of
`log(alpha_theta)`, and signed `log(alpha_theta)` profiles with symmetric
`± fitted_RMS_a` curves for all three families. The second and third columns
use the same fitted magnitudes; both signs share exponents and breakpoints.
Each row uses its stated validity cohort. RMS preparation lives in this cell,
so the removed single-power/exponential experiment is no longer a dependency.

The following parameter-profile reprise overlays `mu ± sqrt(Vref) * fitted_RMS_u`,
`exp(± fitted_RMS_a)`, and `± fitted_RMS_a` on the three corresponding epoch-colored
checkpoint panels. These are RMS guides, not confidence bands or pointwise bounds.

[rms_power_laws.py](rms_power_laws.py) uses exact nonnegative slope solves for
fixed breakpoints in log space, analytic amplitude solves in magnitude space,
and restarted breakpoint searches. Synthetic recovery checks run with:

```bash
/opt/conda/envs/pytorch/bin/python -m unittest \
  paper_odds.score_error_analysis.tests.test_rms_power_laws
```

## Spatial least squares

At each strictly positive EDM noise level `t = sigma`, evaluate

```text
p_t = sum_k w_k N(mu_k, s_k² + t²)
error(x,t) = score_checkpoint(x,t) - score_exact_mixture(x,t)
error(x,t) = b(t) + C(t)(x - mu_t) + residual(x,t).
```

The mixture has means `[-1, 0.1]`, standard deviations `[0.2, 0.1]`, and weights
`[0.1, 0.9]`; its mean is `-0.01` and data variance is `0.1219`.
Use the actual mixture variance, **not** the denoiser's `sigma_data² = 0.25`.

For population-centered data the minimizer satisfies

```text
b = E_p[error]
C = E_p[(error-b)(X-mu)^T] Cov_p(X)^(-1).
```

The implementation solves weighted least squares after centering and column
scaling, without explicitly inverting the covariance. In 1D, expectations use
256 Gauss-Hermite nodes **per component**, with positive mixture weights.
This approximates least squares under `p_t` deterministically. It does not use a
uniform spatial grid or approximate forward SDE paths. Each checkpoint is also
checked at twice the quadrature order on at least 21 noise levels, including the
largest coefficient/error locations. The exact mixture score uses log-sum-exp
responsibilities, avoiding unstable density ratios in the tails.

Stored diagnostics include both normal equations and the identity

```text
E||error||² = ||b||² + tr(C Cov_p(X) C^T) + E||residual||².
```

With finite Monte Carlo samples and a supplied population center, the energy
identity uses the fitted mean error and weighted sample covariance. The returned
intercept is still evaluated at the supplied center.

For vector inputs the generic function additionally returns the isotropic,
symmetric traceless, and skew parts of `C`. Their energies need not add under an
anisotropic covariance. All non-isotropic parts are identically zero in this 1D
experiment. The residual is the component orthogonal to affine functions under
the integration distribution, not the nonlinear part of the exact score itself.

## From b and C to the draft's phase coordinates

The draft's Eqs. (7)--(10) require a variance normalization and clock. Here

```text
V(t) = 0.1219 + t²
V_ref = V(t_min), t_min = 0.002
d = log(V(t)/V_ref) = Lambda - ell
Lambda = log(V(80)/V_ref).
```

The reference is the **last positive evaluation noise** rather than mathematical
zero. No network score is evaluated at zero noise. The grid combines 161 uniform
log-variance points and 129 log-sigma points; the latter resolve the narrow
low-noise interval where an imperfect denoiser can have large score error.
The highest noise is 80, matching the saved experiments.

The Gaussian *tangent* profiles corresponding to small shape and mean errors are

```text
a_linear(d) = V(t) C(t)
u_linear(d) = V(t) b(t) / sqrt(V_ref).
```

They are fitted separately by signed, weighted least squares:

```text
a_linear(d) ≈ epsilon_a exp(-kappa_a d)
u_linear(d) ≈ epsilon_m exp(-kappa_m d).
```

Trapezoidal integration weights give uniform weight per unit **log variance**;
the extra low-noise grid points do not receive extra weight just because there
are more of them. The signed amplitude is solved analytically at each exponent,
and a coarse search followed by all local refinements finds the best exponent
within the configurable range `[-4, 1e6]`. A dense grid near the original phase
range plus an asinh-spaced search resolves both ordinary exponents and very
narrow low-noise profiles. The unrestricted coordinate plot uses symmetric-log
axes; the original phase overlays retain their original linear axes.
Sign changes are preserved. This is not
a log-log fit of absolute values or a fit to the scalar score-error norm.

The resulting coordinates are `(kappa_a, kappa_m, abs(epsilon_m/epsilon_a))`.
The absolute ratio is appropriate for the second-order phase formula, which
squares both amplitudes. A zero mode has undefined localization; exponents at a
search boundary and poor relative RMSE are recorded. Early (`d >= 2`) and late
(`d <= 2`) fits show sensitivity to the fitting window; they use the same endpoint
coordinate d. Use `--fit-distance MIN MAX` to change the main fit window.

For an affine Gaussian score model, the **exact** algebraic mapping is

```text
alpha = 1 / (1 - V C)
log_alpha = -log(1 - V C)
u_exact = V b / (sqrt(V_ref) (1 - V C)).
```

These exact-mapping profiles and their exponential fits are saved as sensitivity
checks. They are undefined wherever `1-VC <= 0`; such times are recorded as
invalid, never silently excluded from an otherwise full-window fit. Tangent and
exact coordinates agree only for small errors. Large `max_abs_a_linear` or
`max_abs_u_linear` is evidence against a quantitatively accurate small-error
phase prediction.

**For the mixture these are moment-matched Gaussian surrogate coordinates.**
The Gaussian moment dynamics and KL formulas are not exact for a mixture even if
the projected score error is affine. A phase-space point is a summary, and its
fit quality and residual should be inspected alongside it.

## Predictions and the empirical baseline

Three predictions are stored:

1. Direct integration of the **signed measured profiles**, using Eqs. (26), (70),
   and (83), at the finite observed horizon. This preserves temporal sign
   cancellation and avoids the exponential-fitting assumption.
2. The finite-horizon exponential-profile prediction, Eqs. (39), (40), (77), (83).
3. The infinite-horizon exponential prediction, Eq. (85), masked unless
   `kappa_a > 0` and `kappa_m > -1/2` for nonzero modes.

All three use actual fitted/profile amplitudes. Each checkpoint's own ratio is
used for numerical predictions. The existing phase overlay groups checkpoints
into the nearest available ratio panel in log space, purely for display: its
background is for the **panel ratio**, not the individual checkpoint ratio.
Coordinates outside the old axes remain in the CSV and in the unrestricted
coordinate plots, with an explicit omitted-count annotation on the overlays.

The saved sweeps contain 50 gamma values starting at **0.01**, with no saved
gamma-zero entropy. Thus `entropies[0]` is **not** an ODE measurement. The empirical
comparison uses each run's actual smallest recorded gamma as the denominator,
and evaluates predictions at exactly the corresponding recorded gammas. Columns
ending in `_recorded_baseline` use that denominator; `_ode` columns are Gaussian
predictions relative to zero. No observed ODE KL is fabricated or extrapolated.

The observed sweeps use a Gaussian approximate initial prior, finite Euler steps,
and histogram KL estimates. The leading Gaussian theory assumes an exact prior
and continuous time. Consequently, disagreement cannot be attributed solely to
nonlinear residual energy. It can also reflect the non-Gaussian reference,
large errors, profile fitting, prior mismatch, discretization, and KL estimation.
There are three independent training runs, not 144 independent replicates.

The profile-only outputs measure nonlinearity and test its association with
surrogate prediction error. The matched residual intervention described below
now tests what changes when that residual is removed from the sampling score.

## Outputs and verification

- `checkpoint_summary.csv`: one row per checkpoint, including localization,
  amplitudes, fit quality, signed-profile diagnostics, energy fractions, exact
  mapping, window sensitivity, and quadrature diagnostics.
- `profiles/<run>_epochNN.npz`: signed `b`, `C`, normalized profiles, mode energies,
  orthogonality diagnostics, quadrature checks, all prediction curves, and original
  empirical KL curves. Arrays are ordered like `clock.npz`, from low to high noise.
- `clock.npz`: `sigma`, `variance`, `reference_variance`, `distance`, `ell`, `horizon`.
- `kl_comparison.csv`: one row per checkpoint and recorded gamma, with both
  observed KL directions and clearly named predicted ratios.
- `metadata.json`: configuration, source/code/checkpoint provenance policy,
  completion status, and approximation assumptions. Each profile contains its
  checkpoint/statistics SHA256 and configuration fingerprint.
- `checkpoint_phase_coordinates`, `checkpoint_phase_overlays`,
  `example_mode_profiles`, `empirical_vs_gaussian_profiles`,
  `nonlinearity_vs_disagreement`: PDF and PNG figures.
- `diagnostics.json`: aggregate numerical checks and descriptive comparisons.

Completed per-checkpoint files are reused on a matching rerun; provenance changes
require a new output directory. A smoke run can use
`--limit 1 --output-dir /tmp/diffsci-modes-smoke`. Run `--help` for other options.
To regenerate plots without checkpoint inference:

```bash
python -m paper_odds.score_error_analysis.plot_checkpoint_modes \
  paper_odds/outputs/score_error_modes
```

When the controlled ablation results are available, `checkpoint_phase_overlays`
colors its checkpoint markers by the **measured full-score log10 KL ratio** at
the exact panel gamma, using the measured ODE denominator and 64 target-quantile
bins. The main figure uses KL(q || p); `_p_q` uses KL(p || q). Marker and theory
background colors share `RdBu_r` and limits [-2, 2]; colorbar extensions indicate
saturation outside these limits. The outlines keep crosses visible against the
background. Crosses still mean poor/boundary fits, and panel backgrounds still
use the panel's amplitude ratio. Only 15/144 checkpoints are inside these axes.

### Exponential-fit predictions versus measured KL ratios

The original CSV already stored finite- and infinite-horizon exponential
predictions, but the original aggregate correlation figures used signed profiles
only. The added comparison evaluates **all three predictors side by side**:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/conda/envs/pytorch/bin/python -m \
  paper_odds.score_error_analysis.compare_exponential_predictions
```

This uses existing results and repeats no training, profile fitting, or sampling.
The signed, full-window least-squares amplitudes and exponents from
`checkpoint_summary.csv` are used directly, at each checkpoint's **own** amplitude
ratio. The finite-horizon formula uses the saved horizon. The infinite-horizon
formula keeps its convergence mask; all 144 present checkpoints satisfy that
mask, although all have a poor fit in at least one mode.

Two measurement sources are kept separate:

- `saved_sweep`: the recorded gammas nearest 0.2, 1, and 5, normalized by their
  actual recorded gamma=0.01 baseline. Recomputed exponential predictions are
  checked against the original saved prediction columns.
- `controlled_full` and `controlled_affine`: exact gamma=0.2, 1, and 5, normalized
  by each arm's measured gamma=0 ODE KL. Both arms and all three bin counts
  (32, 64, 128) are included. The main figures use the full score and 64 bins.

Correlations are computed separately for each dataset, gamma, KL direction,
partition, and predictor. Outputs report Spearman (the same on positive ratios
and their logarithms), Pearson on **both ratio and log10-ratio scales**, sign
agreement, median absolute log10 prediction error, and log10 RMSE. Per-run
statistics and a Pearson correlation after subtracting each run's mean are
included to inspect between-run differences. These do not make successive
checkpoints statistically independent; no independent-checkpoint p-values are
reported. No gamma-zero self-ratios are included in correlations.

Every comparison includes all finite predictions, a common-valid subset across
the three predictors, and, for the controlled experiment, an above-control-floor
subset. The floor flag requires numerator and denominator KL to exceed three
times the corresponding exact-score estimate. It is a descriptive diagnostic.

Outputs live in `../outputs/exponential_kl_comparison/`:

- `predictions_vs_measurements.csv`: 6,048 checkpoint/gamma/direction comparisons,
  each with all three predictions, baseline, fit-quality, and floor information.
- `correlations.csv`: 1,440 descriptive summaries, including per-run and subset
  comparisons. Select `run=all`, `subset=all`, and `bins=64` for the primary
  controlled comparison; saved sweeps use `bins=0` as a not-applicable sentinel.
- `saved_sweep_{q_p,p_q}`, `controlled_full_{q_p,p_q}`, and
  `controlled_affine_{q_p,p_q}`: PNG/PDF scatter grids, with predictor rows and
  gamma columns. White marker centers identify controlled estimates near the
  exact-score floor.
- `phase_coordinates_measured_{q_p,p_q}`: all 144 coordinates on symmetric-log
  axes, colored by the same measured full-score log ratios as the phase overlays.
- `metadata.json`: input/code hashes, conventions, and completion status.

The command also updates the two colored overlays in `../outputs/score_error_modes/`.
The notebook has a new **Exponential-fit KL predictions and measured phase
colors** section with selectable dataset, KL direction, and bin count. See
[RESULTS.md](RESULTS.md#exponential-fit-kl-predictions) for the observed correlations.

To check temporal resolution as well as spatial quadrature, rerun the first,
middle, and last epoch of each training run with both grids and quadrature order
doubled. This writes `resolution_validation.json` with the changes in energy
fractions, response integrals, KL ratios, and fitted exponents:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /opt/conda/envs/pytorch/bin/python -m paper_odds.score_error_analysis.validate_checkpoint_modes
```

Numerical tests recover known weighted vector/matrix affine coefficients, recover
a known orthogonal nonlinear Hermite mode, check Gaussian-mixture moments and
Stein identities, recover signed exponential parameters, test the exact Gaussian
parameter mapping, compare kernels to independent quadrature, test convergence
domains, and verify checkpoint inference against the original architecture:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /opt/conda/envs/pytorch/bin/python -m unittest paper_odds.score_error_analysis.tests.test_score_error_modes -v
```

## Kappa estimation: smoothing and cumulative-response sensitivity

The phase-coordinate estimator in `fit_exponential` minimizes **signed profile
squared error** with trapezoidal weights in log-variance distance. It does not
differentiate a log-log curve or average local slopes. Narrow high-amplitude
terminal features can nevertheless dominate this objective. The clock itself
compresses the low-noise region: for `d=log(V/V_ref)`,
`dd/dlog(sigma)=2*sigma²/V`. A modest slope against log sigma can thus correspond
to a large local exponent against d. Large kappa is not automatically a numerical
error, but can be a poor summary of the cumulative effect.

Run the independent sensitivity analysis on all 144 saved profiles:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/conda/envs/pytorch/bin/python -m \
  paper_odds.score_error_analysis.analyze_kappa_sensitivity
```

The original profiles, fits, predictions, and figures are preserved. New files
go into `../outputs/kappa_sensitivity/`. No training or sampling is repeated, and no
measured KL values enter any fit. Five alternatives are compared:

- `original`: the existing pointwise weighted least-squares fit.
- `smooth_0.01`, `smooth_0.05`, `smooth_0.2`: Gaussian smoothing with these standard
  deviations in **d units**, followed by the same signed pointwise fit. These
  widths are experimental scale choices, not optimized hyperparameters. First
  integrate the original piecewise-linear profile into uniform bins, then apply
  reflecting Gaussian smoothing. This conserves signed area and avoids dropping
  narrow peaks when moving off the hybrid grid. Eight bins per bandwidth are
  used; all kappa search bounds remain `[-4, 1e6]`.
- `cumulative`: fit the cumulative **signed ODE mode response** directly:

```text
beta = 0 for shape, 1/2 for mean
J(d) = integral_0^d exp(-beta*s) profile(s) ds
J_fit(d) = epsilon * F(kappa+beta, d)
F(c,d) = (1-exp(-c*d))/c, with F(0,d)=d.
```

The loss is `integral (J-J_fit)² dd`, with the signed amplitude eliminated
analytically. The constant 1/2 prefactor for the mean ODE response cancels from
the fit. This preserves sign cancellation in the target and emphasizes
accumulated contributions. It still compresses a potentially nonmonotone
cumulative curve into a single exponential, so it cannot reproduce every shape
or every gamma response. A cumulative-fit RMSE is on **J**, whereas the other
RMSEs are on the original or smoothed profiles; these RMSEs are not directly
comparable across objectives.

The cumulative fit also reports an objective-sensitivity envelope: the range
covered by search-grid kappa values whose MSE is within 0.01 times the cumulative
target energy of the optimum. This is **not a confidence interval**. If it reaches
the upper search limit, `upper_unresolved=True` warns that the objective does not
meaningfully constrain large kappa. Report the range/flag alongside such phase
coordinates; do not replace them by an arbitrary small exponent.

Use `mode_fits.csv` for per-mode estimates, fit quality, cancellation, envelope
flags, and kernel errors. `checkpoint_coordinates.csv` gives all five sets of
phase coordinates. `predictions_vs_measurements.csv` and `correlations.csv`
compare their **finite-horizon** KL predictions against the original saved sweep
and the controlled full/affine samplers at 64 bins, with correct baselines,
per-run statistics, and control-floor subsets. A finite horizon permits fitted
exponents outside the infinite-horizon convergence domain; such coordinates are
counted explicitly and omitted from the original infinite-horizon axes.

The mode-response error is the relative Euclidean error across gamma
`[0, 0.2, 1, 5]`, relative to integration of the original signed profile. The
smoothing-response change compares the smoothed profile itself with the original,
before fitting; a small signed-area error alone does not guarantee small changes
under nonconstant response kernels. Neither measure uses observed sampling KL.

Figures show kappa distributions, the largest-kappa examples in both profile and
cumulative coordinates, prediction correlations, and separately saved colored
phase overlays for smoothing width 0.05 and cumulative fitting. The smoothing
overlay's fit flags concern its smoothed-profile objective; cumulative-overlay
fit flags concern its cumulative objective. They are not interchangeable with
the original raw-profile flags. The new notebook section loads these outputs
and lets you inspect any checkpoint and smoothing width.

`resolution_validation.json` doubles the smoothing-bin density and refines the
cumulative objective grid on the same piecewise-linear profiles for epochs
0, 24, and 47 in each run. This checks the estimator's numerical resolution;
it does not add new model evaluations. Seven added tests cover conservative
smoothing, known signed exponentials, narrow peaks, sign cancellation, and
unresolved localization. Run the complete analysis test directory with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/conda/envs/pytorch/bin/python -m unittest discover \
  -s paper_odds/score_error_analysis/tests -t . -v
```

## Global log-log slope fits

The additional `loglog_kappa` experiment tests global least squares **after**
taking logarithms of the magnitude, without smoothing or numerical derivatives:

```text
x = log(V/V_ref) = d
z = log(abs(f)), with f = a_linear or u_linear
z ≈ intercept + slope*x
kappa = -slope, amplitude_magnitude = exp(intercept).
```

This is a log-log fit of profile magnitude against the **variance ratio**. The
same logarithm base on both axes gives the same slope. The clock d is already a
logarithm, so `log(d)` would be a different model. A fit against log sigma also
estimates a different exponent: only in the high-noise limit, where V is
approximately sigma², does its slope approach `-2*kappa`. Likewise, raw b or C
must first be normalized to the draft's u or a profiles.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/conda/envs/pytorch/bin/python -m \
  paper_odds.score_error_analysis.analyze_loglog_kappa
```

Outputs are saved separately in `../outputs/loglog_kappa/`. The primary `loglog` variant
uses trapezoidal weights in d, so the dense terminal part of the hybrid grid does
not get extra weight just because it contains more samples. The
`loglog_unweighted` variant is ordinary least squares on all retained saved nodes,
matching a standard straight-line regression on the log-log scatter plot.

The primary fit excludes `abs(f) <= 1e-6*max(abs(f))`; excluded points are not
clipped to a floor and their original quadrature weights are not redistributed
across gaps. Cutoffs 1e-8 and 1e-4 are also evaluated. Their retained clock
fractions are exported: a large peak can make even a relative cutoff discard a
substantial part of the profile. Zero modes or fewer than three retained points
have undefined fits. There are no kappa bounds.

Both positive and negative values enter via their magnitudes. The result is a
**magnitude envelope**; it cannot reproduce signed cancellations in the Gaussian
response integrals. Mode amplitudes are the positive exponentials of the fitted
intercepts, not amplitudes refitted in linear space. The per-mode sign does not
affect this envelope's second-order KL formula, which squares the mode responses;
sign changes *within* a profile do matter and are recorded separately. Log-space
R² and RMSE describe magnitude fits, not signed-response accuracy.

`mode_fits.csv` stores slopes, exponents, amplitudes, fit diagnostics, sign changes,
and retained support. `checkpoint_coordinates.csv` includes the original and four
log-fit variants. `correlations.csv` and `predictions_vs_measurements.csv` compare
finite-horizon envelope predictions with the original sweeps and the controlled
full/affine samplers at 64 bins, retaining their respective baselines. No observed
KL is used in fitting. The original results remain unchanged.

Figures show example log-log plots (point colors distinguish signs), measured-KL
correlations, and colored phase overlays for weighted and ordinary log fits.
Overlay crosses use relative **magnitude-profile** RMSE > 0.25; mixed-sign flags
must be inspected separately. The notebook includes a selectable checkpoint,
mode, and weighting, plus the all-checkpoint comparisons.

Six tests cover known slopes and log bases, sign changes, cutoff handling,
nonuniform-grid weights, true large exponents, and undefined profiles. An
independent weighted design-matrix solve verifies all 288 primary mode fits.
See [results](RESULTS.md#global-log-log-least-squares).

## Prediction agreement across kappa estimation strategies

The unified evaluation makes **numerical agreement** the main criterion and keeps
correlations alongside it. It uses the existing predictions from the original
exponential fits (finite and infinite horizon), signed profiles, three smoothing
widths, cumulative fits, and four log-log variants. Predictions and coefficients
are held fixed: no calibration slope/intercept is fitted against measured KL.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/conda/envs/pytorch/bin/python -m \
  paper_odds.score_error_analysis.prediction_agreement
```

Let `R_hat` be a predicted KL ratio, `R` its measured counterpart, and
`x=log10(R_hat)`, `y=log10(R)`. The principal least-squares criterion is

```text
MSE_log10  = mean((x-y)²)
RMSE_log10 = sqrt(MSE_log10)
MSE_ratio  = mean((R_hat-R)²).
```

RMSE in log units measures multiplicative mismatch symmetrically: being a factor
of ten high or low gives the same absolute log error. Raw-ratio MSE/RMSE is also
reported, as requested, and penalizes large absolute ratio errors more strongly.
No predictions or large errors are clipped. Both scales also have MAE, signed
bias (prediction minus observation), median absolute error, and 90th-percentile
absolute error.

Additional diagnostics distinguish closeness from association:

- **No-change baseline:** predict `R_hat=1`, or `x=0`, for every checkpoint.
  `skill_vs_no_change = 1 - MSE_model/MSE_baseline`. Positive values mean lower
  squared error than this baseline. The score is undefined when baseline MSE is
  zero.
- **Fixed-prediction R²:** `1 - MSE/variance(measured)`. This is agreement against
  the identity line and can be negative. It is not Pearson correlation squared.
  Its reference is the observed group's mean, so it is a descriptive in-sample
  reference, not an independently evaluated deployable predictor.
- **Concordance:** `2*cov(predicted,measured) / (var(predicted)+var(measured)
  +(mean(predicted)-mean(measured))²)`. It penalizes location and scale mismatch
  that Pearson/Spearman correlation can overlook. Identical constants receive 1;
  their Pearson/Spearman correlations remain undefined.
- **Correlations:** Pearson on log ratios and raw ratios, and Spearman. They remain
  in the tables and are displayed beside agreement errors.

Joins use dataset, run, epoch, exact gamma, baseline gamma, bin count, and KL
direction. Repeated predictions/measurements must agree to 1e-12; duplicate copies
of the original fit are counted once. Each comparison uses the same finite
checkpoint support across all 11 estimators plus the no-change baseline. The
current data have no common-support exclusions. Original sweeps retain their
gamma=0.01 denominator; controlled full/affine measurements retain their measured
ODE denominator and 64 bins.

Metrics are calculated separately by gamma, KL direction, dataset, training run
(each run and all runs), and control-floor subset. The optional across-gamma
summary gives each of the three gamma settings equal weight: its RMSE is the
square root of the mean gamma-specific MSE. Correlations are averaged within
gamma, never recomputed after pooling distinct gamma responses. The above-floor
subset uses the existing numerator/denominator exact-score control diagnostic.

Outputs in `../outputs/prediction_agreement/`:

- `metrics.csv`: 1,440 per-gamma metric rows.
- `aggregate_metrics.csv`: 480 summaries with equal weight per gamma.
- `checkpoint_errors.csv`: 31,104 prediction/measurement pairs with both squared
  errors and the common-support flag, covering 2,592 distinct measurements.
- `agreement_and_correlation`: side-by-side numerical RMSE and rank correlations.
- `raw_ratio_agreement`: raw-ratio RMSE, using logarithmic colors for readability.
- `identity_gamma1_{q_p,p_q}`: shared-axis plots with the identity line and each
  estimator's RMSE/bias/correlation.
- `metadata.json`: input/code hashes, matching rules, metrics, and completion.

The notebook's **Prediction agreement across all kappa strategies** section
provides sortable per-gamma and aggregate tables plus selectable identity plots.
These are descriptive evaluations of three correlated training trajectories;
choosing a strategy from this table is not a held-out generalization result.
Seven tests include perfect correlation with severe bias, exact predictions,
raw/log error differences, baseline skill, common support, and gamma weighting.

## Matched nonlinear-residual ablation

The new experiment uses

```text
s_lambda(x,t) = exact_score(x,t) + b(t) + C(t)(x-mu) + lambda*r(x,t).
lambda=0: the affine score error alone, added to the exact mixture score.
lambda=1: the full learned score.
```

The reference mixture score remains nonlinear in both arms. This removes the
**nonlinear score error**, not the non-Gaussian mixture itself. Coefficients b and
C are recomputed by spatial least squares at every integration node, without
exponential fitting or temporal coefficient interpolation.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/diffsci-mpl \
  /opt/conda/envs/pytorch/bin/python -m \
  paper_odds.score_error_analysis.run_residual_ablation \
  --workers 8
```

Defaults are all 144 checkpoints, gamma = `[0, 0.01, 0.2, 1, 5]`, lambda =
`[0, 1]`, four independent seeds, 4,096 particles per seed (16,384 pooled), and
600 steps. All arms and gamma values share initial states and Gaussian draws;
the same draws are reused across checkpoints. A separate exact-score control
uses those same draws. Gamma zero is a **measured ODE baseline** in this experiment.
The sampling prior is the exact high-noise mixture, and the target is the exact
mixture at sigma=0.002. `--prior gaussian` additionally allows the original
approximate N(0, sigma_max²) prior; the Gaussian surrogate predictions include
its initial mean and variance mismatch.

The sampler uses standardized coordinates `z=(x-mu)/sqrt(V)` and the increasing
log-variance clock ell, in which the Brownian noise is additive:

```text
dz = [z/2 + (1+gamma)*sqrt(V)*s_lambda/2] d ell + sqrt(gamma) dW.
```

Stochastic Heun evaluates the drift at the beginning and predicted end of each
step, using the same Brownian increment in both evaluations. The positive EDM
rho=7 time grid ends at sigma_min; no singular zero-noise score is evaluated.
There is no final denoising step to zero. The new results therefore define their
own controlled experiment and should not be substituted for the original saved
Euler sweep with its approximate prior and histogram estimator.

For practical CPU execution, the learned normalized score is tabulated on a
regular standardized spatial grid, starting with 1,025 nodes over [-12,12].
Independent quadrature-node checks compare it to direct network inference at
11 time levels. The grid doubles if relative RMS interpolation error exceeds
0.003 of the actual score-error RMS. The exact mixture score is evaluated
analytically. Any sampled point outside the table is evaluated directly by the
network; particles are never clipped or discarded. Table errors and fallback
counts are saved per checkpoint.

### KL estimates and uncertainty

All arms use the **same fixed target-quantile bins**, including infinite tail
bins. Target probabilities are known analytically, so no random reference
histogram or sampler-dependent edges are used. The experiment records 32, 64,
and 128 bins; figures use 64 by default. A Jeffreys pseudocount of 0.5 per bin
keeps reverse KL finite if an arm has an empty bin.

These estimates are KL divergences of a finite partition, not unbiased estimates
of continuous-density KL. Inspect their sensitivity to bin count and the
exact-score control. Flags identify ratios where either terminal or ODE KL is
within three times the corresponding control estimate. This is a descriptive
floor diagnostic, not a statistical significance test. Estimates near that floor
are particularly sensitive to finite sample size.

The main paired quantities are

```text
nonlinear_effect = log10(KL_full,gamma / KL_full,0)
                 - log10(KL_affine,gamma / KL_affine,0)

disagreement_reduction = |full_log_ratio - Gaussian_profile_log_ratio|
                      - |affine_log_ratio - Gaussian_profile_log_ratio|.
```

A positive disagreement reduction means removing the residual improves agreement
with the first-order Gaussian profile prediction. A negative nonlinear effect
means retaining the residual makes stochasticity more favorable relative to the
ODE. Removing the residual can change the ODE and stochastic KL simultaneously;
both absolute KL values are saved, not only their ratios.

Pooled histogram counts give the primary estimates. Paired delete-one-seed
jackknife standard errors quantify Monte Carlo uncertainty in these two effects.
They do not account for uncertainty across training runs, and four seeds provide
only a modest uncertainty estimate. The 144 checkpoints still come from three
training runs.

### Separating two Gaussian approximations

Alongside the small-error response formula, the experiment integrates the full
Gaussian mean/variance equations using the same measured b and C:

```text
r' = -(1+gamma)/2 * [(1-VC)r - Vb/sqrt(V_ref)]
v' = [-gamma + (1+gamma)VC]v + gamma.
```

Their exact terminal Gaussian KLs remove the small-amplitude expansion while
retaining the moment-matched Gaussian reference. Comparing these predictions to
the affine-only mixture sampler helps distinguish finite-amplitude effects from
the remaining Gaussian-reference approximation. It is not an additive
decomposition of every source of error.

### Files and numerical checks

Results live in `paper_odds/outputs/residual_ablation/`:

- `ablation_results.csv`: each checkpoint, gamma, lambda and bin count, with
  pooled KLs, true-ODE ratios, Gaussian predictions, and control-floor flags.
- `paired_effects.csv`: residual effects and changes in prediction disagreement,
  with paired Monte Carlo standard errors, for both KL directions.
- `replicate_results.csv`: individual seed estimates.
- `checkpoints/*.npz`: counts for every arm/seed/binning, fresh b and C, Gaussian
  moments and responses, sample moments, numerical checks, and provenance.
- `exact_control.npz`: exact-score control counts and sample moments.
- `metadata.json`, `summary.json`: configuration, completeness, descriptive
  aggregate comparisons, and run-stratified summaries.
- `ablation_prediction_gaps`, `nonlinear_effect_vs_energy`,
  `ablation_example_curves`, `exact_score_control`: PNG/PDF figures.

To reproduce the numerical refinement on nine representative checkpoints:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/conda/envs/pytorch/bin/python -m \
  paper_odds.score_error_analysis.validate_residual_ablation
```

It doubles the time steps, initial score-table resolution and spatial quadrature.
Brownian bridges make each pair of fine noise increments sum to its original
coarse increment. `resolution_validation.json` reports changes in the measured
residual effects and whether they meet the explicit resolution criterion.

The added tests check lambda=0/1 identities, identical-arm noise pairing, exact
Gaussian ODE and SDE marginals, the nonperturbative Gaussian moment equations,
analytic target bin masses, complete tail accounting, direct score-table
fallback, and Brownian-bridge increments. Run all analysis tests with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/conda/envs/pytorch/bin/python -m unittest discover \
  -s paper_odds/score_error_analysis/tests -t . -v
```

For a quick independent smoke run use `--limit 1 --particles 1024 --seeds 170 271
--steps 300 --workers 1 --output-dir /tmp/diffsci-ablation-example`. Matching runs
resume from saved checkpoint files; changed configurations require another output
directory. Individual checkpoints for validation can be selected using, e.g.,
`--checkpoint-ids default3:0 default4:24 default5:47`.

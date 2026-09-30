# Results for the 144 mixture checkpoints

The completed analysis contains 48 checkpoints from each of `default3`,
`default4`, and `default5`. Every checkpoint has signed b(t) and C(t) curves at
288 positive noise levels, nonlinear residual energies, temporal fits, and
Gaussian surrogate predictions. See [README.md](README.md) for definitions and
the reproduction commands. The [controlled residual ablation](#controlled-residual-ablation)
below now measures what changes when the nonlinear score error is removed.
The [exponential-fit comparison](#exponential-fit-kl-predictions) also reports
correlations for the fitted profiles and adds measured colors to the phase plots.
The latest [prediction-agreement evaluation](#prediction-agreement-across-kappa-strategies)
compares squared errors and correlations across every estimator on matched data.

## Main observations

- The median nonlinear fraction is **61.09%** of total score-error energy
  integrated uniformly in log-variance time. Across checkpoints it ranges from
  **8.93% to 97.68%**. This is a projection-energy diagnostic, not a fraction of
  final sampling KL.
- A single signed exponential is usually a poor summary. Median relative RMSE
  is **50.17% for the normalized shape profile** and **58.93% for the normalized
  mean profile**. Every checkpoint exceeds 25% RMSE in at least one mode.
- The shape profile changes sign in **131/144** checkpoints; the mean profile
  changes sign in **121/144**. Integrating the signed measured profiles retains
  cancellation that fitting magnitudes would discard.
- **129/144** full-window fitted coordinates lie outside the original phase
  axes (`kappa_a` from 0 to 2, `kappa_m` from -0.5 to 2). The unrestricted plot
  includes all 144, using symmetric-log axes. None of the final fits hits the
  exponent search bounds. Some exponents are hundreds or thousands because
  the lowest-noise error is concentrated in a narrow interval of log variance.
- Small-error assumptions are also strained: median maximum absolute tangent
  amplitudes over time are **5.72 for shape** and **15.05 for mean**. Phase labels
  therefore should not be interpreted as exact predictions for these mixtures.

## Comparison with existing sampling measurements

Both predictions and measurements below use the **recorded gamma = 0.01
baseline**. The old saved sweeps do not contain an ODE KL measurement.

| Recorded gamma | Sign agreement, KL(q \|\| p) | Sign agreement, KL(p \|\| q) | Spearman prediction/measurement, q \|\| p | Spearman prediction/measurement, p \|\| q |
| --- | ---: | ---: | ---: | ---: |
| 0.208766 | 52.1% | 52.1% | 0.128 | 0.087 |
| 0.984605 | 43.1% | 50.7% | 0.402 | 0.368 |
| 5.024117 | 65.3% | 77.1% | 0.484 | 0.476 |

These use the finite-horizon Gaussian kernels applied to the **measured signed
profiles**, without exponential fitting. They compare whether KL increases or
decreases relative to the recorded baseline. The 144 observations are correlated
checkpoints of only three training runs, so these are descriptive statistics.

Nonlinear residual fraction alone does **not** explain the prediction gap: its
Spearman association with absolute predicted-versus-observed log-ratio difference
is weakly negative (between -0.18 and -0.06 across the six comparisons). Those
correlations alone cannot isolate the residual's effect. The matched ablation
below addresses that question; finite
amplitudes, non-Gaussian reference dynamics, prior and numerical effects remain
confounders. The analysis measures the nonlinear error and makes this limitation
visible rather than attributing all Gaussian-theory disagreement to it.

## Numerical verification

- Ten unit tests pass, including recovery of known affine and nonlinear fields,
  correct Gaussian normalization, signed exponential fits, independent kernel
  quadrature, and equivalence to the original checkpoint inference code.
- For every checkpoint, 256-node/component quadrature is compared with
  512-node/component quadrature on at least 21 time points. The maximum relative
  change in the affine field RMS is **5.87e-7**, normalized by total score-error
  RMS; maximum relative change in total score-error energy is **2.94e-4**.
- The orthogonal energy identity holds to a maximum relative error of
  **1.29e-15**.
- Doubling both temporal grids and spatial quadrature for epochs 0, 24, and 47
  of all three runs changes the combined response integrals by at most **0.31%**,
  nonlinear energy fractions by **0.00079 absolute**, and predicted log10 KL
  ratios by **0.02575**. Scaled exponent changes `abs(delta kappa)/(1+abs(kappa))`
  are at most **0.31%**. Full details are in `resolution_validation.json`.
- The appended notebook cells execute against all 144 saved results without
  importing Lightning or diffusers.

Tables, curves, figures, and numerical checks are in
[`../outputs/score_error_modes`](../outputs/score_error_modes). The original
phase-diagram artifacts and existing notebook cells are preserved.

## Controlled residual ablation

The experiment has now run for **all 144 checkpoints**, with gamma = 0, 0.01,
0.2, 1, and 5, four shared seeds, and 4,096 particles per seed and arm (16,384
pooled). It compares the full learned score with the exact mixture score plus
only its fitted affine error. The exact mixture prior, random draws, integration
grid, and target-defined KL partitions are matched. There is a true ODE baseline
and a separate exact-score control. Exponential profile fits are not used.

**Removing the residual usually improves agreement with the Gaussian profile
prediction, even though residual-energy fraction poorly predicts the size of
that improvement.** At 64 target-quantile bins:

| Gamma | KL direction | Checkpoints with smaller prediction gap after removal | Median full-score gap | Median affine-only gap | Spearman with prediction: full → affine |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.2 | q to p | 68.8% | 0.132 | 0.064 | 0.150 → 0.457 |
| 0.2 | p to q | 67.4% | 0.128 | 0.063 | 0.111 → 0.440 |
| 1 | q to p | 69.4% | 0.318 | 0.177 | 0.405 → 0.540 |
| 1 | p to q | 70.1% | 0.300 | 0.171 | 0.362 → 0.537 |
| 5 | q to p | 68.8% | 0.514 | 0.338 | 0.496 → 0.615 |
| 5 | p to q | 66.0% | 0.447 | 0.317 | 0.468 → 0.621 |

Gaps are absolute differences between predicted and measured
`log10(KL_gamma / KL_ODE)`. Comparing medians is not the same as taking the median
of paired differences; `paired_effects.csv` contains both the paired reduction
and its Monte Carlo standard error for every checkpoint.

The qualitative conclusion persists at 32 and 128 bins: the fraction improving
is between **61.8% and 73.6%** across all six comparisons and the three partitions.
At 64 bins, between 105 and 113 checkpoints have both arms' numerator and
denominator KL above three times the exact-score control estimate. Among these
checkpoints, the improvement fraction is **66.1% to 74.3%**. This floor threshold
is a diagnostic, not a significance test.

The residual also affects absolute sampling quality. At gamma=1, removing it
reduces measured KL in **96.5%** of checkpoints for q to p and **97.2%** for p to
q. Median q-to-p KL falls from **0.0380 to 0.0137**. These absolute-KL changes
should be distinguished from changes in the stochastic/ODE ratio, since the
residual also affects the ODE.

The residual-energy fraction remains weakly correlated with the **absolute
controlled effect on the log ratio**: Spearman coefficients range from **-0.089
to -0.011** at 64 bins. Thus the original weak energy/disagreement correlation
was not evidence that the residual had little effect. Its integrated energy
fraction does not capture the temporal and spatial structure that governs the
effect.

### What remains unexplained

The affine-only sampler still disagrees appreciably with the Gaussian
prediction. Integrating the full Gaussian moment equations, rather than their
small-amplitude expansion, changes median affine-only gaps only modestly: for
q-to-p KL they are **0.061, 0.179, and 0.331** at gamma=0.2, 1, and 5, versus
**0.064, 0.177, and 0.338** for the first-order formula. The finite-amplitude
correction alone therefore does not remove the remaining discrepancy in this
experiment. Non-Gaussian mixture dynamics and finite-partition/sample effects
remain relevant limitations; this comparison does not uniquely quantify each.

### Verification and limits of the estimates

- The original profile tables were reproduced after relocation and are
  byte-identical; every profile data array is also unchanged. Updated provenance
  is documented in `outputs/score_error_modes/relocation_validation.json`.
- All 19 analysis and ablation tests passed, covering intervention identities,
  noise pairing, exact Gaussian marginals and moments, bin masses, tail
  accounting, Brownian bridges, and paired uncertainty calculations.
- The maximum checked score-table interpolation RMS error is **0.2974%** of
  the score-error RMS. No sampled state required direct evaluation outside the
  table in the main run; fallback remains implemented and tested.
- Doubled time steps, score-table resolution and projection quadrature passed
  the specified paired-refinement criterion on epochs 0, 24 and 47 of all three
  runs. Fine Brownian increments sum to the original coarse increments. The
  largest change in the measured residual log-ratio effect is **0.0596 dex**;
  it remains within the criterion `max(0.03 dex, 2 paired MC standard errors)`.
  Read the per-checkpoint results in `outputs/residual_ablation/resolution_validation.json`.
- The reported KLs are fixed-partition estimates with analytic target masses
  and a 0.5 count pseudocount. Near the exact-score control floor, ratios remain
  sensitive to sampling and binning. Four-seed jackknife errors describe Monte
  Carlo uncertainty, not training-run uncertainty.
- These are three training runs with correlated checkpoints. Percentages and
  correlations are descriptive, rather than population-level significance claims.

Open the [prediction-gap figure](../outputs/residual_ablation/ablation_prediction_gaps.png),
[paired effects table](../outputs/residual_ablation/paired_effects.csv), or
[full sampling results](../outputs/residual_ablation/ablation_results.csv).

## Exponential-fit KL predictions

The exponential predictions were already saved, but **their aggregate
correlations had not been reported**. The earlier correlation tables used the
signed measured profiles. The new experiment compares all three predictions
against both sets of sampling measurements without repeating sampling or fitting.

For the controlled **full learned score**, measured ODE baseline and 64 bins,
Spearman correlations across all 144 checkpoints are:

| Gamma | KL direction | Signed profiles | Exponential, finite horizon | Exponential, infinite horizon |
| --- | --- | ---: | ---: | ---: |
| 0.2 | q to p | 0.150 | 0.124 | 0.116 |
| 0.2 | p to q | 0.111 | 0.088 | 0.080 |
| 1 | q to p | 0.405 | 0.198 | 0.189 |
| 1 | p to q | 0.362 | 0.207 | 0.198 |
| 5 | q to p | 0.496 | 0.403 | 0.401 |
| 5 | p to q | 0.468 | 0.432 | 0.429 |

The exponential fits retain some ranking information, especially at gamma=5,
but give weaker rank correlations than the signed profiles in all six primary
comparisons. At gamma=1, median absolute log10-ratio errors are **0.446 and 0.427**
for the finite exponential fits (q to p and p to q), versus **0.318 and 0.300** for
the signed profiles. Correlation and quantitative calibration are different
questions; neither validates the Gaussian surrogate as an exact mixture model.

Finite and infinite exponential predictions behave similarly here. Switching
horizon limits does not recover the ranking information retained by the signed
profiles. This is consistent with the poor temporal fits and frequent sign
changes, although this comparison does not isolate every source of mismatch.
All 144 fitted points satisfy the infinite-horizon convergence conditions; none
is removed by that mask. All remain flagged for poor fit in at least one mode.

The original sweeps give a similar pattern using their own recorded baseline:

| Recorded gamma | KL direction | Signed profiles | Exponential, finite horizon | Exponential, infinite horizon |
| --- | --- | ---: | ---: | ---: |
| 0.208766 | q to p | 0.128 | 0.116 | 0.108 |
| 0.208766 | p to q | 0.087 | 0.098 | 0.089 |
| 0.984605 | q to p | 0.402 | 0.190 | 0.181 |
| 0.984605 | p to q | 0.368 | 0.226 | 0.217 |
| 5.024117 | q to p | 0.484 | 0.321 | 0.317 |
| 5.024117 | p to q | 0.476 | 0.430 | 0.426 |

Here the denominator is gamma=0.01, **not an ODE measurement**. The two sampling
experiments have different priors, integrators, and KL estimators, so they remain
separate datasets in every table and plot.

Removing the nonlinear residual improves the exponential fits' association with
measurements too: at gamma=1, finite-exponential Spearman rises from **0.198 to
0.403** for q to p and **0.207 to 0.407** for p to q. The direct signed-profile
predictions still rank the affine-only outcomes better (0.540 and 0.537).

The exported statistics include Pearson on ratios and log ratios, sign agreement,
prediction errors, all three KL partitions, control-floor subsets, and per-run
correlations. For example, the gamma=1 finite-exponential q-to-p correlation is
0.201, 0.162, and 0.270 within the individual training runs. These remain
descriptive correlations among related checkpoints, not 144 independent trials.

### Measured colors on the phase diagrams

The [updated phase overlay](../outputs/score_error_modes/checkpoint_phase_overlays.png)
colors each checkpoint cross by the **measured full-score
log10(KL_gamma/KL_ODE)** at exactly the panel gamma, using 64 bins. The background
and markers share the same blue–white–red scale: blue means stochastic sampling
has lower KL than the ODE; red means higher KL. The background is still the
infinite-horizon Gaussian prediction at the **panel's** amplitude ratio, while
each marker color is a sampling measurement. Crosses retain their poor-fit
meaning. Outlines make pale markers visible; values beyond [-2, 2] saturate.

There is also a [reverse-KL overlay](../outputs/score_error_modes/checkpoint_phase_overlays_p_q.png)
and an [all-144-checkpoint view](../outputs/exponential_kl_comparison/phase_coordinates_measured_q_p.png).
The original phase axes contain only 15/144 checkpoints, so their visible colors
should not be read as representative of the entire checkpoint collection.

See the [full-score comparison figure](../outputs/exponential_kl_comparison/controlled_full_q_p.png),
[correlation table](../outputs/exponential_kl_comparison/correlations.csv), and
[individual comparisons](../outputs/exponential_kl_comparison/predictions_vs_measurements.csv).
Reconstructed exponential predictions match the original saved columns to
1e-12 absolute tolerance. Five additional tests cover baseline matching,
convergence masks, correlation scales, within-run centering, and exact-gamma
phase-color joins.

## Why some kappa estimates are huge

The current estimator performs signed, weighted least squares on the normalized
profiles. It does **not** average log-log slopes. Its squared-error objective can
nevertheless prefer a narrow terminal peak over the broader, low-amplitude part
that contributes most of the integrated response. These features were stable in
the existing projection/time-resolution checks; their size is not evidence of
finite-difference noise.

For `default3`, epoch 18, the original mean exponent is **4,924**. The interval
`0 <= d <= 0.01` contributes **53.55%** of squared profile energy but only
**1.22%** of absolute profile area. The clock `d=log(V/V_ref)` strongly compresses
small sigma intervals too. This makes a high fitted exponent understandable,
while weakening its interpretation as an overall error-growth descriptor.

### Smoothing and cumulative fits across all 144 checkpoints

Gaussian smoothing is performed in d, preserving signed area by conservative
rebinning and reflecting boundaries. The cumulative estimator fits the signed
ODE mode integrals, using kernel 1 for shape and exp(-d/2) for mean. All methods
retain the original wide exponent bounds; none is clipped to the old phase axes.

| Estimator | Median shape kappa | Median mean kappa | Maximum shape kappa | Maximum mean kappa | Inside original axes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original squared-profile fit | 9.426 | 15.577 | 3254 | 4924 | 15/144 |
| Smoothing width 0.01 | 6.177 | 0.572 | 86.7 | 106.3 | 25/144 |
| Smoothing width 0.05 | 2.770 | 0.165 | 30.8 | 27.5 | 45/144 |
| Smoothing width 0.2 | 1.158 | 0.079 | 7.27 | 7.17 | 71/144 |
| Cumulative ODE response fit | 0.749 | 0.474 | 5673 | 12203 | 66/144 |

The signed-area drift under smoothing is at most **2.1e-15**, normalized by
absolute profile area. Smoothing also changes time localization, so the kernel
responses need separate inspection. The median relative response changes are
about **0.4%**, **3.1%**, and **15–17%** for widths 0.01, 0.05, and 0.2,
respectively. These are Euclidean changes across gamma `[0, 0.2, 1, 5]`, not
changes in observed KL. Wider smoothing produces smaller kappa partly by imposing
a broader temporal scale; smaller exponents alone are not a validation criterion.

The cumulative fit lowers the typical exponent but does not eliminate large or
ambiguous values. Its loss permits exponents up to the search limit within the
specified 1%-of-energy tolerance for **18 shape and 36 mean modes**. These are
flagged `upper_unresolved` in the fit table. They are objective-sensitivity flags,
not statistical confidence statements. Cumulative curves with substantial sign
cancellation can also fit a single exponential poorly. In this fit, 21
checkpoints fall outside the infinite-horizon convergence domain; all finite-
horizon predictions remain evaluated.

For the controlled full-score sampler, 64 bins and measured ODE baseline,
Spearman correlations for **KL(q || p)** are:

| Gamma | Original fit | Smooth 0.01 | Smooth 0.05 | Smooth 0.2 | Cumulative fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.124 | 0.160 | 0.177 | 0.171 | 0.172 |
| 1 | 0.198 | 0.272 | 0.281 | 0.242 | 0.311 |
| 5 | 0.403 | 0.408 | 0.381 | 0.243 | 0.243 |

At gamma=1, smoothing width 0.05 reduces median absolute log10 prediction gaps
from **0.446 to 0.301** for q to p and **0.427 to 0.311** for p to q. The cumulative
fit gives **0.252 and 0.248**. Its rank correlations, however, deteriorate at
gamma=5. Thus smoothing addresses the terminal-peak dominance, but neither
bandwidth selection nor cumulative fitting supplies a universally better phase
coordinate. The direct signed-profile kernels remain useful for predictions
without compressing the temporal structure into one exponent. All measured-KL
comparisons here are evaluations after fitting, not fitting targets.

The new [profile/cumulative example figure](../outputs/kappa_sensitivity/large_kappa_profiles.png)
shows exactly what the large exponents capture and miss. See the
[kappa distributions](../outputs/kappa_sensitivity/kappa_distributions.png),
[correlation comparison](../outputs/kappa_sensitivity/estimator_correlations.png), and
[smoothed colored phase overlay](../outputs/kappa_sensitivity/smooth_0.05/checkpoint_phase_overlays.png).
The original analysis remains intact; the notebook has a separate sensitivity
section with selectable checkpoint, mode and smoothing width.

All 72 estimator-resolution checks on nine representative checkpoints passed:
the largest `abs(delta kappa)/(1+abs(kappa))` is **0.00233** after doubling the
smoothing-bin density or refining the cumulative quadrature grid. These checks
use the same projected profiles; the earlier independent projection-resolution
validation is separate. All **31 analysis tests** pass, including seven new
tests for the alternative estimators.

## Global log-log least squares

Global least squares on `log(abs(profile))` versus `log(V/V_ref)` gives much more
moderate slopes, without smoothing or restricting kappa. The exponent is **minus
the fitted slope**. This estimates the growth of the profile's magnitude; it
cannot capture temporal sign cancellation. The primary fit weights uniformly in
log variance and excludes magnitudes at or below 1e-6 of the mode's maximum.
Ordinary unweighted least squares on the saved nodes is included separately.

| Primary weighted log fit | Minimum kappa | Median kappa | Maximum kappa | Median log-space R² |
| --- | ---: | ---: | ---: | ---: |
| Shape | 0.306 | 0.693 | 0.961 | 0.894 |
| Mean | -0.394 | -0.014 | 0.490 | 0.230 |

All **144/144** coordinates now lie within the original phase axes. The shape
magnitude often has an approximately power-law trend in variance; the mean
magnitude is generally less well described by a single slope. The sign-changing
profiles remain: **131 shape and 121 mean modes**. Good magnitude R² and moderate
kappa do not imply accurate signed-response or sampling predictions.

For the controlled full-score experiment (64 bins, measured ODE baseline),
Spearman correlations of finite-horizon predictions with measured ratios are:

| Gamma | KL direction | Original signed linear-space fit | Weighted log-log fit | Ordinary log-log fit |
| --- | --- | ---: | ---: | ---: |
| 0.2 | q to p | 0.124 | 0.042 | 0.152 |
| 0.2 | p to q | 0.088 | 0.080 | 0.167 |
| 1 | q to p | 0.198 | -0.030 | 0.150 |
| 1 | p to q | 0.207 | -0.008 | 0.145 |
| 5 | q to p | 0.403 | -0.086 | 0.131 |
| 5 | p to q | 0.432 | -0.109 | 0.106 |

The weighted log fits reduce median absolute log10 prediction gaps at gamma=1
to **0.158 / 0.163** (q to p / p to q), but have almost no rank association with
the outcomes. Reduced median error and useful checkpoint ranking are distinct
properties. The ordinary fits retain a weak positive association. These
comparisons do not isolate sign cancellation from other Gaussian-surrogate
limitations; they show that removing extreme exponents does not by itself
produce strong checkpoint rankings. Numerical prediction agreement is assessed
separately in the unified evaluation below.

### Weighting and cutoff sensitivity

Ordinary regression gives median shape/mean kappa **0.820 / 0.194**, compared
with **0.693 / -0.014** under uniform log-variance weighting. The saved grid has
many low-noise nodes, so equal node weights emphasize that region. Neither fit
averages local slopes; the distinction is the measure used by global regression.

Lowering the relative cutoff to 1e-8 changes an individual kappa by at most
**0.107**, with median change zero. Raising it to 1e-4 changes kappa by up to
**0.856**, and in the most affected shape profile retains only **22.2%** of the
clock's weight. At the primary 1e-6 cutoff the minimum retained shape/mean weights
are **87.7% / 99.1%**. Near-zero handling therefore must be reported even for
apparently well-behaved log-log fits.

The [log-log example plots](../outputs/loglog_kappa/loglog_profile_examples.png) retain
positive/negative profile signs as different point colors. The
[correlation figure](../outputs/loglog_kappa/loglog_correlations.png) compares the cutoff
and weighting variants; the [phase overlay](../outputs/loglog_kappa/loglog/checkpoint_phase_overlays.png)
shows all 144 magnitude-fit coordinates with measured KL colors. These fits
describe magnitude trends and are stored as a separate experiment. All previous
fits, comparisons and figures are preserved.

All **37 analysis tests** pass, including six added log-log tests. All 288 primary
mode fits are also checked against an independent weighted least-squares
design-matrix solve; see `validation.json`.

## Prediction agreement across kappa strategies

**Log-log fits agree much better numerically than their weak correlations alone
suggested.** Earlier comparisons emphasized ranking; RMSE and median errors were
saved, but were not presented consistently across every strategy. The new
evaluation scores the fixed predictions against the identity line, using all 144
matched checkpoints, and keeps Pearson/Spearman correlations alongside the losses.

The primary loss is mean squared error in log10 KL ratios, reported as RMSE.
Raw-ratio MSE/RMSE and MAE are also included, along with signed bias, concordance,
and fixed-prediction R². No line is fitted to recalibrate the predictions.

For the full learned-score sampler, KL(q || p), 64 bins and measured ODE baseline:

| Strategy | RMSE, gamma=0.2 | RMSE, gamma=1 | RMSE, gamma=5 |
| --- | ---: | ---: | ---: |
| Signed profiles | 0.265 | 0.568 | 0.802 |
| Original finite exponential fit | 0.207 | 0.499 | 0.756 |
| Smoothed, width 0.05 | 0.184 | 0.415 | 0.597 |
| Smoothed, width 0.2 | 0.166 | 0.350 | 0.484 |
| Cumulative response fit | 0.172 | 0.402 | 0.635 |
| Weighted log-log | **0.128** | **0.276** | 0.530 |
| Ordinary log-log | 0.151 | 0.296 | **0.387** |
| No-change baseline: ratio=1 | 0.150 | 0.284 | 0.406 |

All entries are log10-ratio RMSE; this table shows the principal strategies.
The full files also contain infinite-horizon fits, smoothing width 0.01, and both
log-cutoff sensitivity variants. For example, cutoff 1e-4 gives 0.127 at gamma=0.2.

At gamma=1, the weighted log-log q-to-p prediction has RMSE **0.276** versus
**0.499** for the original exponential and **0.568** for direct signed profiles.
Its mean prediction bias is **+0.018 dex**, compared with **+0.343** and **+0.309**,
respectively. This is a substantial improvement in numerical agreement, even
though its Spearman correlation is -0.030. The original correlation-led assessment
therefore understated the log-fit approach's accuracy.

### Comparison with a no-change baseline

The baseline predicts `KL_gamma/KL_ODE=1` for every checkpoint. This exposes how
much improvement comes beyond a simple constant prediction. At gamma=1 the
weighted log fit reduces q-to-p log-MSE by **5.7%** relative to this baseline;
at gamma=0.2 the reduction is **27.0%**. At gamma=5 it performs worse, while
ordinary log-log fitting improves baseline log-MSE by **9.3%**.

Giving each of gamma=0.2, 1, and 5 equal weight:

| Strategy | Overall log-RMSE, q to p | Overall log-RMSE, p to q |
| --- | ---: | ---: |
| Signed profiles | 0.588 | 0.564 |
| Original finite exponential | 0.536 | 0.508 |
| Smoothed 0.2 | 0.358 | 0.370 |
| Weighted log-log | 0.353 | 0.404 |
| Ordinary log-log | **0.295** | **0.330** |
| No-change baseline | 0.299 | 0.340 |

Ordinary log-log fitting has the lowest overall log-RMSE among the evaluated
strategies, but its improvement over the no-change baseline is modest:
**3.0% / 6.0% in MSE** for q to p / p to q. For q to p the per-run improvements
are +11.4%, -3.9%, and +0.9%; this is not a uniform improvement across runs.
The aggregate gives equal gamma weight and does not pool observations to compute
correlations. These are descriptive comparisons, not a held-out selection result.

### The error scale matters

Raw-ratio agreement is reported as well. At gamma=1, q-to-p raw-ratio RMSE is
**0.606** for the weighted log fit, **0.600** for ordinary log fitting,
**1.782** for the original exponential, and **0.571** for the no-change baseline.
Thus the weighted fit improves on the baseline in log-MSE here, but not in raw
ratio MSE. Both comparisons are valid objectives with different penalties.

Signed-profile predictions have relatively stronger correlations but a few very
large ratio predictions produce high squared error. For q to p at gamma=5, their
raw-ratio RMSE is **807.3**, versus **15.6** for the original exponential and
**1.93** for ordinary log fitting. No large predictions are clipped or removed.
The log-scale score reduces that asymmetry and measures multiplicative mismatch.

The tables also include concordance and `1-MSE/variance(observed)` R². This R²
is evaluated on fixed predictions and can be negative; it is not Pearson r².
All principal methods have negative log-scale identity R² in the primary full-
score comparisons, so the improvement over earlier models should not be confused
with strong agreement for checkpoint-to-checkpoint variation.

See the [agreement/correlation comparison](../outputs/prediction_agreement/agreement_and_correlation.png),
[identity plots](../outputs/prediction_agreement/identity_gamma1_q_p.png),
[raw-ratio errors](../outputs/prediction_agreement/raw_ratio_agreement.png), and
[complete metrics](../outputs/prediction_agreement/metrics.csv). The original sweep,
affine-only arm, per-run comparisons, and above-control-floor subsets are all
included in the exported tables. Repeated predictions and observations are
checked for consistency; no checkpoint is excluded from the current common
finite support. All **44 analysis tests** pass, including seven additional tests
for the scoring and aggregation rules. The new notebook cells execute and their
tables/figures are saved in the notebook.

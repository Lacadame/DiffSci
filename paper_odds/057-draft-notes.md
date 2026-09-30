# Notes on the 28 September 2026 draft

These notes accompany [057-stoch_tests.ipynb](057-stoch_tests.ipynb). Numbers below use its default settings: 64 bins, the measured γ = 0 baseline, all 144 checkpoints for sampler agreement, and the 84 epoch ≥ 20 checkpoints for envelope and GP fitting. The original mixture, models and sampler caches are unchanged. Mixture randomization and retraining remain deferred.

## Replace the provisional envelope results

The revised coordinates are **a = VC** and **u = Vb/√Vref**. These are exact affine score coefficients; the Gaussian response and second-order KL are approximations. Notebook 056 already used these coefficients for its direct predictions, but fitted envelopes in different, singular coordinates. Its direct cached response predictions therefore remain valid inputs; its envelope tables do not carry over.

All 84 selected checkpoints now enter every noise level. The 329 points with a ≥ 1, across 27 checkpoints, remain in the analysis. There is no conditional-validity cohort.

Weighted log-space RMS fits, with nonnegative exponents and free continuous breakpoints:

| Coordinate | Segments | Exponents, increasing σ | Breaks in σ | log10 RMSE |
|---|---:|---|---|---:|
| a | 2 | 0.888, 1.340 | 1.019 | 0.0589 |
| a | 3 | 0.889, 1.605, 0.897 | 1.723, 14.844 | 0.0279 |
| u | 2 | 1.158, 0 | 0.286 | 0.0278 |
| u | 3 | 1.189, 0.829, 0 | 0.112, 0.382 | 0.0198 |

The shape's first break is now about 2.92 or 4.94 data standard deviations, depending on the family. The old claim that it occurs at 1.5–2.5 data standard deviations should be replaced. The old mean break near 0.006 also disappears in these coordinates. Neither old break should currently be used to argue scaling with a component variance.

The ensemble mean carries **13.6% of the log-noise-weighted a mean square and 48.5% of the u mean square**. The previous 32–68% versus 2–26% contrast no longer holds. Both means should be retained in the GP model. Centered SD envelopes are fitted separately: RMS² = mean² + SD², so fitting RMS alone does not identify a GP amplitude.

Sources: [envelope fits](outputs/stoch_tests_057/envelope_fits.csv), [profiles](outputs/stoch_tests_057/envelope_profiles.csv), [clock norms](outputs/stoch_tests_057/envelope_clock_norms.csv).

## Replace Table 4 and Figure 4

Using the two- and three-segment RMS fits, the finite-horizon, independent **unit Gaussian amplitude** benchmark gives:

| γ | Shape response ratio | Mean response ratio | log10 ratio of expected KLs | Two-amplitude win probability |
|---:|---:|---:|---:|---:|
| 0.2 | 0.960–0.965 | 1.057–1.058 | 0.0054–0.0071 | 0.442–0.459 |
| 1 | 0.958–0.973 | 1.268–1.272 | 0.0930–0.0971 | 0.198–0.240 |
| 5 | 1.292–1.313 | 2.143–2.172 | 0.478–0.491 | 0 |

The shape optimum on the declared numerical grid moves to γ ≈ 0.49–0.54. The fitted shape L¹ norms are 0.158–0.160. The notebook also locates positive shape/ODE crossings within its declared γ ≤ 30 search interval. These are predictions from fitted profiles, not gamma choices made using sampler KL.

**Section 9.4 needs an amplitude-law correction.** Independent random signs with fixed magnitudes leave both squared responses unchanged and give a deterministic leading-order comparison. The arctangent odds require random magnitudes, such as independent standard Gaussian amplitudes. Suggested wording: “We first use the fitted RMS curves as deterministic profiles multiplying independent standard Gaussian amplitudes. This centered benchmark ignores the measured systematic component.” Also distinguish a ratio of expected KLs from an expected KL ratio or a win probability.

Sources: [envelope predictions](outputs/stoch_tests_057/envelope_predictions.csv), [replacement figure](outputs/stoch_tests_057/envelope_predictions.png).

## Fixed-gamma empirical odds

Strict improvement fractions, using a measured ODE baseline for each arm:

| Cohort | γ | Full q‖p / p‖q | Affine-error q‖p / p‖q |
|---|---:|---:|---:|
| All 144 | 0.2 | 81.25% / 84.72% | 33.33% / 33.33% |
| All 144 | 1 | 70.14% / 68.75% | 28.47% / 27.78% |
| All 144 | 5 | 30.56% / 27.78% | 13.89% / 13.19% |
| Epoch ≥ 20, n = 84 | 0.2 | 80.95% / 83.33% | 36.90% / 35.71% |
| Epoch ≥ 20, n = 84 | 1 | 65.48% / 66.67% | 34.52% / 32.14% |
| Epoch ≥ 20, n = 84 | 5 | 38.10% / 36.90% | 17.86% / 17.86% |

Use the epoch-selected rows when comparing with the fitted envelopes or GPs. At γ = 1, epoch-selected per-run q‖p fractions range from 60.7% to 71.4% for the full score and 25.0% to 46.4% for affine error. These are three trajectories with repeated epochs, not 84 independent draws. The notebook provides bin-count, control-floor and paired delete-one-seed sensitivity, without binomial intervals over checkpoints. Near γ = 0.01, deleting a sampling seed changes classifications considerably; treat those strict near-tie counts cautiously.

**Section 9.6's set-inclusion statement needs qualification.** For a smooth function, h′(0) < 0 implies improvement at some sufficiently small positive γ relative to h(0). It does not imply improvement on a particular finite gamma grid, relative to γ₀ ≈ 0.01, with the first three grid values excluded, or by more than 1%. Thus the recorded sweep-minimum event does not in general contain {h′(0) < 0}. The new notebook uses the fixed-gamma event directly.

Sources: [pooled odds](outputs/stoch_tests_057/empirical_odds.csv), [per-run odds](outputs/stoch_tests_057/empirical_odds_by_run.csv), [seed sensitivity](outputs/stoch_tests_057/seed_deletion_sensitivity.csv).

## Agreement and the linearization claim

The original cached direct response and paired residual conclusions reproduce. At γ = 1, removing the residual reduces absolute log-ratio prediction error for 66.7% / 68.1% of checkpoints (q‖p / p‖q). Signed-profile classification accuracy is 50.7% / 52.1% for the full score and 70.1% / 70.8% for affine error. Bounded-change RMSE is 0.458 / 0.455 versus 0.330 / 0.327. The notebook includes the full confusion counts and balanced accuracy.

**“Integrates exactly (500 Euler steps)” in Section 9.5 is incorrect.** The moment equations are exact for the affine Gaussian surrogate; Euler is a numerical approximation. The new notebook integrates the same linearly interpolated cached coefficients with refined exponential midpoint steps. Refinements 4 and 8 change log ratios by at most 2.83 × 10⁻⁵ dex.

The refined moment/profile gap at γ = 1 has median 0.0129 / 0.0125 dex, but maximum 0.0729 / 0.0704 dex. At γ = 5 its maximum is 0.111 / 0.097 dex. Cached 500-step Euler moment ratios can differ from the refined solution by about **0.347 dex** in an outlying checkpoint at γ = 1. Replace “indistinguishable” or “identical to ±0.03” with a quantified statement about typical and tail discrepancies. Small-error linearization is often a smaller error source than mixture mismatch, but the numerical evidence does not justify an unqualified identity claim.

For the epoch-selected cohort, median ∫|a| dd is 0.117 and median ∫|u| dd is 0.571; the maximum mean norm is 2.089. The shape-envelope L¹ norm alone does not verify the small-δ hypothesis of Proposition 2.3, which includes both modes. Direct convergence/comparison diagnostics are more informative here. A small L₀[a] alone also does not make the *total* KL denominator small when M₀[u] contributes appreciably.

Sources: [agreement metrics](outputs/stoch_tests_057/agreement_metrics.csv), [moment checks](outputs/stoch_tests_057/moment_linearization_check.csv), [cancellation diagnostics](outputs/stoch_tests_057/cancellation_diagnostics.csv).

## Gaussian-process comparison

The fitted OU lengths in log σ are **4.73 for a and 3.70 for u**; pooled ACF RMSEs are 0.069 and 0.087. These are descriptive fits under a stationarity assumption. Both modes have nonzero ensemble means. The fitted-SD, independent-mode GP predicts both-mode win probabilities 0.390, 0.279 and 0.113 at γ = 0.2, 1 and 5. Using measured SD gives 0.405, 0.303 and 0.132. A Gaussian fitted to the four response coordinates, retaining cross-mode covariance, gives 0.436, 0.339 and 0.162.

At γ = 1, the observed signed Gaussian-surrogate responses win for 32.1% of the selected checkpoints. This is much closer to the affine-error mixture sampler (32.1–34.5%) than the full sampler (65.5–66.7%). It is an aggregate comparison, not evidence of accurate checkpoint-by-checkpoint prediction. Three leave-one-run-out checks are included. Shape-only GP odds are compared with **projected shape-only responses**; there is no cached shape-only mixture sampling experiment.

The standard errors in the GP table quantify numerical scrambled-Sobol integration only. They do not estimate uncertainty over training, finite-sample GP calibration or model misspecification. Quadrature refinement is checked separately.

Sources: [GP calibration](outputs/stoch_tests_057/gp_fit.csv), [GP odds](outputs/stoch_tests_057/gp_odds.csv), [holdout runs](outputs/stoch_tests_057/gp_run_holdout.csv), [quadrature check](outputs/stoch_tests_057/gp_quadrature_check.csv).

## Setup and other wording corrections

- **Mixture setup:** weights (0.1, 0.9), means (−1, 0.1), component standard deviations (0.2, 0.1). The data variance is 0.1219; it is different from the denoiser's σ_data² = 0.25.
- **Architecture/training provenance:** the strict checkpoint loader uses width 128, 64 sinusoidal time-embedding coordinates, three two-linear-layer blocks with SiLU, and no residual connections. The training setup saved in [notebook 054](../notebooks/exploratory/bps/054-bps-entropy_paper-investigating_scores-80.ipynb), cells 76–78 by zero-based index, specifies 4,000 training samples, 10,000 validation samples, batch size 16, AdamW at 0.04/128 = 0.0003125, no LR scheduler, and 48 epochs. This is the saved `default5` configuration, not a complete independently verified historical record of all three runs.
- **Noise embedding:** Section 7.1 cites the conventional EDM log σ / 4 embedding. The experiment's custom `mode='default'` uses **log σ / 2**, as confirmed by [the inference loader](score_error_analysis/checkpoint_score.py) and notebook 054. Both motivate log-noise stationarity, but the experimental coefficient should be stated correctly.
- **Positive endpoint:** the empirical target is p at σ = 0.002, with Vref = 0.121904, not exactly p₀. Shift the clock and normalization together. The relative standard-deviation difference is about 1.64 × 10⁻⁵. Finite-horizon/finite-endpoint predictions should not be presented as the infinite-horizon pure-power threshold results.
- **Remark 3.6:** convergence to probability one as Λ → ∞ for each fixed atomless prior does not give a prior-uniform guarantee of probability near one at Λ ≈ 10. An atomless prior concentrated in the finite-horizon unfavorable region is a counterexample to the stated universal practical-horizon conclusion. Also Rα = exp(−Λ/(2α)); replacing it by exp(−Λ/2) presumes α near one. Qualify both statements by restrictions on the prior.
- **Scope of the affine arm:** removing the projected residual leaves the exact *mixture* score plus affine error. This is not an affine Gaussian score, so remaining disagreement cannot all be assigned to linearization.
- **References/typesetting:** repair `??5.4` in Proposition 5.5 and the discussion. The conjecture remains a conjecture here; these notebook computations do not establish its proof.

## Reproduction

Execute [057-stoch_tests.ipynb](057-stoch_tests.ipynb) from the repository root or the notebook directory with a Python kernel providing NumPy, SciPy, pandas, Matplotlib and IPython. Default execution takes about a minute on CPU and uses no GPU. Artifacts are saved to `outputs/stoch_tests_057/`; [manifest.json](outputs/stoch_tests_057/manifest.json) records settings and source/cache hashes. These notes describe the default run and must be updated if the notebook settings change.

Scientific helper checks:

```bash
OPENBLAS_NUM_THREADS=1 /opt/conda/envs/pytorch/bin/python -m unittest \
  paper_odds.score_error_analysis.tests.test_draft057 \
  paper_odds.score_error_analysis.tests.test_rms_power_laws \
  paper_odds.score_error_analysis.tests.test_score_error_modes
```

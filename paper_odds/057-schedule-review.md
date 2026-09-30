# Review of 057-schedule_tests

The original notebook was a useful experiment outline, but it was not directly runnable against this repository and did not reproduce the finite-step sampler conventions of 056. The corrected [notebook](057-schedule_tests.ipynb) connects all three arms to the existing experiment and saves actual results for all 144 checkpoints. The original input notebook was preserved during review at `/tmp/057-schedule_tests.original.ipynb`.

## Corrections

1. **Mixture metadata:** the original resolver omitted `probabilities`, the key used in our metadata. It would fail before sampling. The corrected version accepts it and checks the complete component distribution, clock moments, and coordinate identities. A changed mixture cannot silently reuse old checkpoint profiles.
2. **Real checkpoint inference:** the full-score loader was a `NotImplementedError` placeholder and silently removed the full arm. The notebook now uses the existing `CheckpointScore`, resolves checkpoints by run/epoch from `checkpoint_summary.csv`, checks all stored SHA-256 hashes, and requires all three requested arms. This preserves the original width-128 model, float64 score inference, σ_data=0.5 and custom noise conditioning log σ / 2.
3. **Sampler convention:** 056 uses an EDM ρ=7 noise grid and Euler/EM updates in standardized state z=(x−μ)/√V and log-variance clock ℓ. The original schedule notebook instead used a uniform-ℓ grid with Euler updates in physical x and added-variance increments. That is a different legitimate discretization of the same SDE, but not the claimed finite-step match. The corrected sampler uses the existing standardized update. Regression tests give **identical arrays** to the old sampler for constant γ on the same grid.
4. **Window boundaries:** endpoint-only gamma checks can miss a narrow burst entirely or apply it beyond its stated window. Every finite schedule edge is now inserted into both the sampler and theory grids, and gamma is evaluated inside each interval. The response weights and Gaussian moments are exact for the frozen cell coefficients; approximating measured time-varying profiles still has integration error.
5. **Pairing and cost:** the old bridge seeds depended on schedule and checkpoint, and different schedules could use different substeps. Shared coarse increments do not then give a shared Brownian path on all subintervals. Every schedule, arm and checkpoint now uses one common grid and the same per-seed Brownian sequence. Exact controls use that same coupling. The main run has **869 actual intervals**, starting from 500 EDM intervals and adding edges/stability refinements. All candidates have equal score-evaluation counts.
6. **Actual replicate seeds and estimator:** four independent seeds `[170,271,372,473]`, 4,096 particles per seed, exact mixture prior, target σ=0.002, and the existing target-quantile KL estimator with 0.5 pseudocount. Per-seed counts are retained for 32/64/128 bins. Fresh affine projections are made at every sampler node; the full score uses checked spatial tables with direct inference outside the table. Nonfinite trajectories raise an error rather than being counted in a histogram tail.
7. **Cache integrity:** the original row keys contained no settings/source/checkpoint fingerprint and could silently reuse incompatible runs, even while overwriting metadata. Corrected caches live in a fingerprinted directory, validate provenance, and write each checkpoint atomically. Missing rows cannot enter win-rate denominators as losses.
8. **Constant competitors:** γ=0 is now included in the tested constant set `{0,0.05,0.2,1,5}`. “Best constant” and “oracle constant” refer only to this finite grid, not all constant gamma values. Their selection is in-sample and is labeled accordingly.
9. **Placement versus amount:** the original low/mid/high windows have integrated stochasticity budgets about **2.77, 7.48 and 44.10**, respectively. They are not placements of the same-width window in the natural clock and cannot by themselves isolate placement. Three added windows have the same γ=5 and the same clock width, each with budget **2.7681**.
10. **Interpretation:** the affine-error arm still includes the nonlinear exact mixture score; Gaussian theory is a surrogate there. Computing a profile-selected schedule before sampling does not make its selection out-of-sample. Population/tailored windows use the same checkpoint profiles being evaluated. Repeated epochs are not independent trained models. The notebook exports results by run and bin/seed sensitivity rather than attaching binomial intervals over checkpoints.

The original numerical window candidates remain in the experiment so their intended test is preserved. Their asserted ordering is now treated as a hypothesis. Indeed, before sampling, the updated signed-profile calculation predicts median log KL ratios of approximately **+0.018 for the mid window and +0.030 for the burst**, versus **−0.005 for the population window and −0.195 for tailored windows**. The original claim that the mid window and burst must win was not supported by these measured profiles.

## Numerical validation

Five targeted tests cover constant-sampler equivalence, constant-profile response/moment formulas, narrow-window edge inclusion, identical noise for identical schedules/arms, and cache separation by settings. A complete six-checkpoint smoke run exercised actual model inference and all score arms before the production experiment.

A separate diagnostic uses checkpoints `default3:0`, `default4:24`, and `default5:47`, the same four seeds and 4,096 particles per seed, and **869 versus 1,738 intervals**. Fine Brownian-bridge increments sum to the coarse increments, so differences are paired. For above-floor comparisons across both learned arms and both KL directions, the largest change in a log KL ratio was **0.0207 dex**. The largest change including floor-limited ratios was **0.1138 dex**. Typical absolute changes were about 0.0054 dex in the affine arm and 0.0068 dex in the full arm. These three checkpoints are a resolution diagnostic, not a population convergence guarantee; advantages of a few hundredths of a dex should not be declared robust solely from the main run.

The resolution CSV includes a paired delete-one-seed standard error for the coarse/fine difference. It measures Monte Carlo variability of that diagnostic, not uncertainty over training runs.

## Artifacts and reproduction

The experiment is in [outputs/schedule_experiment_euler500/825cda211589](outputs/schedule_experiment_euler500/825cda211589). It contains pooled results, per-seed checkpoint histograms, exact-score controls, source/settings hashes, the actual clock, pre-sampling theory predictions, and the resolution check. The notebook exports the comparison tables and figures to this same directory.

The numerical implementation is [schedule_experiment.py](score_error_analysis/schedule_experiment.py). It reuses the existing projection, checkpoint, exact-prior, Brownian-generator, histogram, KL, and affine-field code. No production integrator, checkpoint, mixture, or older sampler cache is changed. Both the smoke and full experiment use CPU inference/sampling; no GPU is required. Device 6 remains the permitted GPU if inference is changed later.

Use the repository's PyTorch environment to run the notebook. A full rerun reuses compatible checkpoint caches; `SCHEDULE_NB_QUICK=1` uses separate smoke settings. Tests:

```bash
OPENBLAS_NUM_THREADS=1 /opt/conda/envs/pytorch/bin/python -m unittest \
  paper_odds.score_error_analysis.tests.test_schedule_experiment
```

Paired resolution check:

```bash
OPENBLAS_NUM_THREADS=1 /opt/conda/envs/pytorch/bin/python -m \
  paper_odds.score_error_analysis.validate_schedules \
  paper_odds/outputs/schedule_experiment_euler500/825cda211589
```

## Executed results

All **144 checkpoints**, **15 schedules**, both learned-score arms and the exact controls completed. Each result pools four seeds × 4,096 particles. All learned-score lookup tables passed the 0.003 relative RMS tolerance; the largest measured error was **0.001773**, and no particles required out-of-table fallback.

The tested schedules do **not** establish a general benefit over the best tested constant gamma. For the full score, the best fixed constant by median log KL ratio is γ=1 in q‖p and γ=0.2 in p‖q. For the affine-error arm, it is the ODE in both directions. The selected population window is [0.432, 0.564] in σ (about [1.24, 1.62] data standard deviations).

Each paired entry below is **q‖p / p‖q**. Positive median differences mean the window is worse than the stated best fixed constant.

| Arm | Schedule | Beats ODE | Beats best fixed constant | Median log10(KL window / KL best fixed) |
|---|---|---:|---:|---:|
| affine | mid window γ=5 | 27.1% / 28.5% | 27.1% / 28.5% | +0.085 / +0.091 |
| affine | burst γ=50 | 30.6% / 30.6% | 30.6% / 30.6% | +0.099 / +0.096 |
| affine | population window γ=5 | 36.1% / 34.0% | 36.1% / 34.0% | +0.046 / +0.043 |
| affine | tailored window γ=5 | 48.6% / 49.3% | 48.6% / 49.3% | +0.004 / +0.003 |
| full | mid window γ=5 | 46.5% / 54.9% | 25.7% / 14.6% | +0.137 / +0.097 |
| full | burst γ=50 | 45.1% / 52.8% | 25.7% / 13.2% | +0.135 / +0.112 |
| full | population window γ=5 | 60.4% / 63.9% | 30.6% / 5.6% | +0.118 / +0.083 |
| full | tailored window γ=5 | 61.1% / 62.5% | 40.3% / 29.2% | +0.046 / +0.050 |

Tailored schedules show the most promise among these candidates, but their full-score median disadvantage versus the best fixed constant is still about **0.046–0.050 dex**. Their approximately 49% win rate in the affine arm is also far below their strong in-sample Gaussian prediction. The floor-filtered affine subset is more favorable (56–60% tailored wins versus ODE), which should be reported as a different, smaller diagnostic subset, not substituted for the all-checkpoint result.

Equal-budget placement does matter: for the full score the low window has median log KL changes **+0.202 / +0.328 dex**, the middle window **−0.0095 / −0.0272 dex**, and the high window **+0.0043 / +0.0006 dex**. This supports distinguishing early and late noise injection, but the modest middle-window advantage over ODE does not establish superiority to a well-chosen constant. Its size is also comparable with the measured step-refinement sensitivity.

The original cached-profile and fresh sampler-node predictions agree closely in typical cases (median absolute log-ratio difference **0.000212 dex**), but the maximum difference is **0.195 dex**. The notebook reports both and uses freshly projected coefficients for an additional agreement table; near-cancelling Gaussian responses are especially sensitive to profile discretization.

Use comparisons within this notebook. Constant-gamma values need not numerically equal the older 500-step cache because the shared grid now has 869 intervals and therefore a different discretized Brownian path, despite the same seed list and exact-prior construction. The constant-update equivalence regression checks the sampler on the **same** grid.

See [schedule summary](outputs/schedule_experiment_euler500/825cda211589/schedule_summary.csv), [constant comparisons](outputs/schedule_experiment_euler500/825cda211589/constant_comparisons.csv), [training-run results](outputs/schedule_experiment_euler500/825cda211589/by_training_run.csv), [bin sensitivity](outputs/schedule_experiment_euler500/825cda211589/bin_sensitivity.csv), [seed sensitivity](outputs/schedule_experiment_euler500/825cda211589/seed_sensitivity.csv), and [paired resolution check](outputs/schedule_experiment_euler500/825cda211589/resolution_validation.csv).

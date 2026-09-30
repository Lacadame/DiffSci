# Section 7: combined mean and shape error phase diagrams

The figures evaluate the **second-order KL comparison** in Eqs. (83)--(85) of
`The_odds_favor_noise_in_generative_diffusion_sampling.pdf`, Section 7 (page 13).
They compare each fixed stochasticity level with the probability-flow ODE,
which has gamma = 0.

## Figures and reproduction

- `outputs/phase_diagrams/phase_diagrams_overview.pdf`: all 15 panels, arranged with
  gamma = 1, 5, 20 along rows and epsilon_m / epsilon_a = 0.1, 0.5, 1, 2, 10
  along columns. A PNG version is also provided.
- `outputs/phase_diagrams/phase_gamma_*_ratio_*.pdf` and `.png`: individual panels.
- `outputs/phase_diagrams/phase_diagrams_all.pdf`: a 15-page PDF with the individual panels.
- `outputs/phase_diagrams/phase_diagram_data.npz`: coordinate arrays and unclipped color
  values, ordered as `[gamma, ratio, kappa_m, kappa_a]`. Color values are stored
  as float32 to reduce file size; calculations and contours use float64.
  Points outside the infinite-horizon convergence domain contain NaN, and
  `valid_domain` supplies the corresponding two-dimensional Boolean mask.
- `outputs/phase_diagrams/phase_boundaries.csv`: interpolated equal-KL contours, including
  separate component and point indices. These are numerical grid contours.
- `outputs/phase_diagrams/metadata.json`, `summary.csv`, and `validation.json`: configuration,
  source PDF checksum, color extrema, and numerical verification results.

From the repository root:

```bash
python paper_odds/plot_phase_diagrams.py --validate
```

Dependencies: NumPy and Matplotlib, plus SciPy for `--validate`.
The source PDF is only used for its provenance checksum; generation does not
require a PDF-reading library. The script uses a noninteractive plotting backend.

## Assumptions and choices

1. **Large horizon:** Lambda tends to infinity, as in Eq. (85). No finite Lambda
   is selected for these figures.
2. **Small errors:** only the second-order terms are retained. The overall error
   amplitude cancels in the comparison, leaving the ratio
   `rho = epsilon_m / epsilon_a`. The sign of either amplitude also cancels at
   this order. Dividing out epsilon_a squared does **not** set epsilon_a to one.
3. **Exact initial sampling prior:** r_T = 0 and beta_T = 1, as assumed in Section 7.
4. **Both KL directions:** their second-order approximations agree; the figures
   therefore apply to both h_q and h_c at this order.
5. **Linear axes:** both kappa_a and kappa_m in [-0.5, 2]. The infinite-horizon
   formulas require kappa_a > 0 and kappa_m > -1/2. Points outside this domain,
   including both singular edges, are masked in gray. The negative-kappa_a strip
   is also hatched and labeled. It is not classified as an ODE or stochastic phase.
   A finite Lambda is needed to evaluate the response integrals over the full square.
6. **Resolution:** 801 uniformly spaced samples along each axis, with kappa_a = 1
   and kappa_m = 0 included exactly (801 points per axis for the default ranges).

The remaining scientific specifications would be **Lambda for a finite-horizon
diagram**, or **an absolute error amplitude for the exact nonlinear comparison**.
The latter would also require choosing a KL direction and the sign of the shape error.
Prior mismatch would introduce additional parameters and is outside these figures.

## Quantity shown

Write H_gamma = h_gamma^(2) / epsilon_a^2. Equation (83) gives

```text
H_gamma = (1/4) L_gamma^2 + (rho^2/2) M_gamma^2,

L_0     = 1 / kappa_a,
L_gamma = (1 + gamma) / (kappa_a + gamma),
M_0     = 1 / (2 kappa_m + 1),
M_gamma = (1 + gamma) / (2 kappa_m + 1 + gamma).
```

The color is `log10(H_gamma / H_0)`:

- **Blue, negative:** the stochastic sampler has lower KL.
- **Red, positive:** the ODE has lower KL.
- **Black contour, zero:** equal KL, the phase boundary in Eq. (84).
- **Gray:** outside the convergence domain of the large-horizon formulas.
- A value of -1 means ten times smaller KL; +1 means ten times larger KL.

All panels use the same color scale, from -2 to +2. Colors saturate outside this
range (a factor of 100); the underlying values and saved arrays remain unclipped.
The colorbar extensions indicate saturation. Change `--color-limit` to widen it.
These are deterministic preference diagrams, not probabilities over error profiles.

Dashed reference lines show kappa_a = 1 and kappa_m = 0. In the large-horizon limit,
all phase boundaries pass through their intersection, marked by a white circle:
both individual error modes are neutral there. Within the convergence domain, kappa_a < 1,
kappa_m < 0 favors stochasticity, and kappa_a > 1, kappa_m > 0 favors the ODE.
In the other two quadrants the two modes compete and the amplitude ratio matters.

The infinite-horizon result is obtained from the fixed-horizon second-order
coefficients. It is not an assertion that finite, nonzero amplitudes are uniformly
small over an infinite interval: for negative kappa_m the mean profile grows
toward the early end, and near kappa_a = 0 the shape response becomes large.

## Finite-horizon and other variants

For a finite Lambda the script uses the exact linear response integrals in
Eqs. (39), (40), and (77), still in the second-order KL approximation:

```text
F(c, Lambda) = (1 - exp(-c Lambda)) / c,    F(0, Lambda) = Lambda,
L_gamma     = (1 + gamma) F(kappa_a + gamma, Lambda),
M_gamma     = (1 + gamma)/2 F(kappa_m + (1 + gamma)/2, Lambda).
```

For example, to choose Lambda = 10 and evaluate the full requested square:

```bash
python paper_odds/plot_phase_diagrams.py \
  --horizon 10 --kappa-a -0.5 2 --kappa-m -0.5 2 \
  --output-dir paper_odds/outputs/phase_diagrams_lambda10
```

This is an optional example; the supplied 15 figures use the infinite-horizon limit.
Finite-horizon boundaries need not pass through (1, 0). Convergence to the limit
can be slow near kappa_a = 0 and kappa_m = -1/2, so a moderately large Lambda
need not reproduce the infinite-horizon diagram near those edges.

Use `--gammas`, `--ratios`, `--kappa-a`, `--kappa-m`, `--resolution`, `--dpi`,
and `--color-limit` to change the grid or presentation. Choose a separate
`--output-dir` to preserve an existing set of figures.

## Verification

`--validate` checks the projection formulas against independent numerical
quadrature at finite and infinite horizons (48 parameter combinations), verifies
removable singularities and known phase signs, and integrates the exact Gaussian
mean/variance equations (12)--(13) for three mixed-error configurations and both
samplers. Both exact KL directions approach the predicted second-order expression
with decreasing amplitudes. Relative remainders must decrease when the two error
amplitudes are halved. Validation does not assume that the second-order boundary
is the exact boundary at finite error amplitude.

## Checkpoint overlays and measured KL colors

The [checkpoint analysis](score_error_analysis/README.md) fits the 144 trained
models and compares exponential-profile KL predictions with sampling results.
The [colored overlay](outputs/score_error_modes/checkpoint_phase_overlays.png) retains
these theoretical backgrounds and colors checkpoint crosses by measured
`log10(KL_gamma / KL_ODE)` from the controlled full-score experiment, at the exact
panel gamma and 64 bins. Blue means lower measured KL than the ODE; red means
higher. Markers and backgrounds share one color scale. Crosses flag poor fits.

Only 15 checkpoints fit inside these original axes. The
[all-checkpoint plot](outputs/exponential_kl_comparison/phase_coordinates_measured_q_p.png)
uses symmetric-log axes to show all 144. Both KL directions and the
[prediction correlations](score_error_analysis/RESULTS.md#exponential-fit-kl-predictions)
are available in `055-stoch_tests.ipynb`.

#!/usr/bin/env python3
"""Plot the second-order KL phase diagram in Section 7, Eqs. (83)--(85).

Run with --help for ranges, finite-horizon support, and output options.
Only NumPy and Matplotlib are needed; --validate additionally uses SciPy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "diffsci-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
DRAFT = HERE / "The_odds_favor_noise_in_generative_diffusion_sampling.pdf"


def exponential_integral(rate, horizon):
    """Integral of exp(-rate*d) over [0, horizon], including rate == 0."""
    rate = np.asarray(rate, dtype=float)
    if np.isinf(horizon):
        if np.any(rate <= 0):
            raise ValueError("Infinite-horizon integrals require strictly positive rates.")
        return 1.0 / rate
    result = np.full_like(rate, horizon)
    np.divide(-np.expm1(-rate * horizon), rate, out=result, where=rate != 0)
    return result


def projections(kappa_a, kappa_m, gamma, horizon=np.inf):
    """Return L_gamma and M_gamma from Eqs. (39), (40), and (77)."""
    shape = (1.0 + gamma) * exponential_integral(np.asarray(kappa_a) + gamma, horizon)
    mean = 0.5 * (1.0 + gamma) * exponential_integral(
        np.asarray(kappa_m) + 0.5 * (1.0 + gamma), horizon
    )
    return shape, mean


def scaled_kl(kappa_a, kappa_m, gamma, ratio, horizon=np.inf):
    """h_gamma^(2) / epsilon_a^2; ratio = epsilon_m / epsilon_a (Eq. 83)."""
    shape, mean = projections(kappa_a, kappa_m, gamma, horizon)
    return 0.25 * shape**2 + 0.5 * ratio**2 * mean**2


def log_kl_ratio(kappa_a, kappa_m, gamma, ratio, horizon=np.inf):
    """Log KL ratio, with NaN outside the infinite-horizon convergence domain."""
    ka, km = np.broadcast_arrays(np.asarray(kappa_a), np.asarray(kappa_m))
    valid = (ka > 0) & (km > -0.5) if np.isinf(horizon) else np.ones(ka.shape, dtype=bool)
    # Avoid evaluating divergent integrals at masked coordinates.
    safe_a, safe_m = np.where(valid, ka, 1.0), np.where(valid, km, 0.0)
    stochastic = scaled_kl(safe_a, safe_m, gamma, ratio, horizon)
    deterministic = scaled_kl(safe_a, safe_m, 0.0, ratio, horizon)
    return np.where(valid, np.log10(stochastic) - np.log10(deterministic), np.nan)


def validate_formulas():
    """Independent quadrature and full moment-ODE checks, plus known limits."""
    import scipy
    from scipy.integrate import quad, solve_ivp

    quadrature_errors = []
    for horizon in (0.3, 7.0, np.inf):
        for gamma in (0.0, 1.0, 5.0, 20.0):
            for ka, km in ((0.07, -0.45), (0.8, 0.0), (1.0, 0.8), (3.0, 2.0)):
                shape, mean = projections(ka, km, gamma, horizon)
                shape_quad = quad(
                    lambda d: (1 + gamma) * np.exp(-(ka + gamma) * d),
                    0, horizon, epsabs=1e-11, epsrel=1e-11,
                )[0]
                mean_quad = quad(
                    lambda d: (1 + gamma) / 2 * np.exp(-(km + (1 + gamma) / 2) * d),
                    0, horizon, epsabs=1e-11, epsrel=1e-11,
                )[0]
                np.testing.assert_allclose([shape, mean], [shape_quad, mean_quad], rtol=1e-10)
                quadrature_errors.extend([
                    abs(shape / shape_quad - 1), abs(mean / mean_quad - 1)
                ])
    # Removable singularities in finite-horizon formulas.
    np.testing.assert_allclose(exponential_integral(np.array([0, 1e-14]), 7), [7, 7], rtol=1e-12)
    # Domain masking must include the singular edges and leave valid points intact.
    domain_values = log_kl_ratio(np.array([-0.5, 0, 0.5, 1]),
                                 np.array([0, 0, -0.5, 0]), 1, 1)
    assert np.isnan(domain_values[:3]).all()
    np.testing.assert_allclose(domain_values[3], 0, atol=1e-14)
    assert np.isfinite(log_kl_ratio(np.array([-0.5, 0, 0.5]), -0.5, 1, 1, 7)).all()
    for gamma in (1.0, 5.0, 20.0):
        for ratio in (0.1, 0.5, 1.0, 2.0, 10.0):
            # All asymptotic contours pass through the two pure-mode thresholds.
            np.testing.assert_allclose(log_kl_ratio(1.0, 0.0, gamma, ratio), 0, atol=1e-14)
            assert log_kl_ratio(0.5, -0.1, gamma, ratio) < 0
            assert log_kl_ratio(2.0, 0.5, gamma, ratio) > 0
        # A pure shape perturbation changes preference at kappa_a = 1.
        assert log_kl_ratio(0.5, 0.3, gamma, 0.0) < 0
        assert log_kl_ratio(2.0, 0.3, gamma, 0.0) > 0

    # Integrate the exact moments, Eqs. (12)--(13), with v-1 as the state
    # to resolve small errors. Compare both exact KL directions with Eq. (83).
    moment_checks = []
    for ka, km, gamma, ratio in ((0.6, -0.2, 1.0, 0.5), (1.7, 0.8, 5.0, 2.0), (0.3, 0.2, 20.0, 10.0)):
        horizon = 7.0
        errors = []
        for epsilon_a in (1e-3, 5e-4):
            for sampler_gamma in (0.0, gamma):
                def rhs(s, state):
                    r, delta_v = state
                    profile = np.exp(-ka * (horizon - s))
                    inv_alpha = np.exp(-epsilon_a * profile)
                    mean_target = ratio * epsilon_a * np.exp(-km * (horizon - s))
                    q = (1 + sampler_gamma) * (-np.expm1(-epsilon_a * profile))
                    return [
                        -(1 + sampler_gamma) / 2 * inv_alpha * (r - mean_target),
                        -sampler_gamma * delta_v + q * (1 + delta_v),
                    ]

                solution = solve_ivp(rhs, (0, horizon), (0, 0), rtol=2e-11, atol=2e-14)
                assert solution.success, solution.message
                r, delta_v = solution.y[:, -1]
                hq = 0.5 * (delta_v - np.log1p(delta_v) + r*r)
                hc = 0.5 * (np.log1p(delta_v) - delta_v / (1 + delta_v) + r*r / (1 + delta_v))
                prediction = epsilon_a**2 * scaled_kl(ka, km, sampler_gamma, ratio, horizon)
                rel_errors = np.abs(np.array([hq, hc]) / prediction - 1)
                assert np.max(rel_errors) < 0.005, rel_errors
                errors.append(rel_errors.tolist())
        # Halving both errors should approximately halve the relative remainder.
        assert np.max(errors[2:]) < 0.7 * np.max(errors[:2])
        moment_checks.append({"kappa_a": ka, "kappa_m": km, "gamma": gamma,
                              "ratio": ratio, "relative_errors_hq_hc": errors})
    return {"passed": True, "scipy": scipy.__version__, "quadrature_cases": 48,
            "max_projection_relative_error": float(max(quadrature_errors)),
            "exact_moment_checks": moment_checks}


def number_tag(value):
    return f"{value:g}".replace("-", "minus").replace(".", "p")


def draw_panel(ax, ka, km, values, gamma, ratio, norm, *, compact=False, asymptotic=True):
    colors = plt.get_cmap("RdBu_r").with_extremes(bad="#dedede")
    masked_values = np.ma.masked_invalid(values)
    mesh = ax.pcolormesh(ka, km, masked_values, cmap=colors, norm=norm,
                         shading="auto", rasterized=True)
    if asymptotic and ka[0] < 0:
        right = min(0.0, ka[-1])
        ax.axvspan(ka[0], right, facecolor="#dedede", edgecolor="#b8b8b8", hatch="///", lw=0)
        ax.text((ka[0] + right) / 2, (km[0] + km[-1]) / 2,
                "Outside convergence domain", rotation=90, ha="center", va="center",
                color="0.35", fontsize=8 if compact else 10)
    ax.axvline(1, color="0.35", lw=0.8, ls=(0, (3, 3)), alpha=0.65)
    ax.axhline(0, color="0.35", lw=0.8, ls=(0, (3, 3)), alpha=0.65)
    segments = []
    if np.nanmin(values) < 0 < np.nanmax(values):
        contour = ax.contour(ka, km, masked_values, levels=[0], colors="#181818",
                             linewidths=1.5, corner_mask=False)
        segments = [s for s in contour.allsegs[0] if len(s) > 1]
    if asymptotic and ka[0] <= 1 <= ka[-1] and km[0] <= 0 <= km[-1]:
        ax.plot(1, 0, "o", ms=4 if compact else 5, color="white",
                markeredgecolor="#181818", markeredgewidth=1.0, zorder=5)
    ax.set_xlim(ka[0], ka[-1])
    ax.set_ylim(km[0], km[-1])
    visible_y_ticks = ax.get_yticks()
    visible_y_ticks = visible_y_ticks[(visible_y_ticks >= km[0]) & (visible_y_ticks <= km[-1])]
    if km[0] < 0 and not np.any(visible_y_ticks < 0):
        ax.set_yticks(np.r_[km[0], visible_y_ticks])
    ax.set_title(rf"$\gamma={gamma:g}$,  $\epsilon_m/\epsilon_a={ratio:g}$", fontsize=12 if compact else 15)
    ax.set_xlabel(r"Shape localization $\kappa_a$", fontsize=10 if compact else 12)
    ax.set_ylabel(r"Mean localization $\kappa_m$", fontsize=10 if compact else 12)
    ax.tick_params(labelsize=9 if compact else 11)
    return mesh, segments


def colorbar_label():
    return r"$\log_{10}\!\left[h^{(2)}(\gamma)/h^{(2)}(0)\right]$"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.2, 1, 5])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.5, 1, 2, 10])
    parser.add_argument("--kappa-a", nargs=2, type=float, default=[0, 2], metavar=("MIN", "MAX"))
    parser.add_argument("--kappa-m", nargs=2, type=float, default=[-0.5, 2], metavar=("MIN", "MAX"))
    parser.add_argument("--horizon", type=float, default=np.inf, help="Lambda; inf selects Eq. (85)")
    parser.add_argument("--resolution", type=int, default=801, help="Samples along each linear axis")
    parser.add_argument("--color-limit", type=float, default=2, help="Symmetric color limits in log10 units; data remain unclipped")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/phase_diagrams")
    parser.add_argument("--validate", action="store_true", help="Also check quadrature, thresholds, and exact moment dynamics")
    args = parser.parse_args()
    for name, bounds in (("kappa-a", args.kappa_a), ("kappa-m", args.kappa_m)):
        if not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            parser.error(f"--{name} requires finite MIN < MAX")
    if args.horizon <= 0 or np.isnan(args.horizon):
        parser.error("--horizon must be positive (or inf)")
    if np.isinf(args.horizon) and (args.kappa_a[1] <= 0 or args.kappa_m[1] <= -0.5):
        parser.error("Plot ranges must intersect kappa_a > 0, kappa_m > -0.5; other points are masked")
    if not all(np.isfinite(g) and g > 0 for g in args.gammas):
        parser.error("--gammas must be finite and positive")
    if not all(np.isfinite(r) and r >= 0 for r in args.ratios):
        parser.error("--ratios must be finite and nonnegative")
    if len(set(args.gammas)) != len(args.gammas) or len(set(args.ratios)) != len(args.ratios):
        parser.error("Gamma and ratio values must be unique")
    if args.resolution < 3 or args.dpi < 50 or not np.isfinite(args.color_limit) or args.color_limit <= 0:
        parser.error("Require resolution >= 3, dpi >= 50, and a finite positive color limit")
    return args


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    validation = validate_formulas() if args.validate else None
    if validation:
        (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print("Validation passed: quadrature, threshold signs, and exact Gaussian moment dynamics.")

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    # Include the two asymptotic thresholds exactly in the coordinate arrays.
    ka = np.unique(np.r_[np.linspace(*args.kappa_a, args.resolution),
                         [1.0] if args.kappa_a[0] <= 1 <= args.kappa_a[1] else []])
    km = np.unique(np.r_[np.linspace(*args.kappa_m, args.resolution),
                         [0.0] if args.kappa_m[0] <= 0 <= args.kappa_m[1] else []])
    norm = Normalize(vmin=-args.color_limit, vmax=args.color_limit)
    asymptotic = np.isinf(args.horizon)
    valid_domain = ((ka[None, :] > 0) & (km[:, None] > -0.5)
                    if asymptotic else np.ones((len(km), len(ka)), dtype=bool))
    has_mask = not valid_domain.all()
    horizon_label = r"$\Lambda\to\infty$" if asymptotic else rf"$\Lambda={args.horizon:g}$"
    model_label = f"Second-order KL comparison  |  {horizon_label}  |  exact prior"
    fields, summary, boundary_rows = [], [], []
    overview, axes = plt.subplots(len(args.gammas), len(args.ratios), squeeze=False,
                                  sharex=True, sharey=True,
                                  figsize=(3.5 * len(args.ratios) + 1.2, 3.55 * len(args.gammas) + 1.0))
    overview.subplots_adjust(left=0.052, right=0.912, bottom=0.11, top=0.90, wspace=0.13, hspace=0.30)

    with PdfPages(out / "phase_diagrams_all.pdf") as multipage:
        for i, gamma in enumerate(args.gammas):
            row_fields = []
            for j, ratio in enumerate(args.ratios):
                values = log_kl_ratio(ka[None, :], km[:, None], gamma, ratio, args.horizon)
                if not np.all(np.isfinite(values[valid_domain])):
                    raise ValueError("Nonfinite KL ratios: reduce finite-horizon profile growth or axis ranges.")
                row_fields.append(values)
                stem = f"phase_gamma_{number_tag(gamma)}_ratio_{number_tag(ratio)}"
                fig, ax = plt.subplots(figsize=(7.4, 6.4))
                fig.subplots_adjust(left=0.115, right=0.80, bottom=0.18, top=0.84)
                mesh, segments = draw_panel(ax, ka, km, values, gamma, ratio, norm, asymptotic=asymptotic)
                cb = fig.colorbar(mesh, cax=fig.add_axes([0.835, 0.18, 0.030, 0.66]), extend="both")
                cb.set_label(colorbar_label(), fontsize=12, labelpad=10)
                fig.suptitle(model_label, fontsize=12, y=0.955)
                fig.text(0.12, 0.094, "Blue: stochastic sampler has lower KL", color="#2166ac", fontsize=10)
                fig.text(0.12, 0.065, "Red: ODE has lower KL", color="#b2182b", fontsize=10)
                fig.text(0.12, 0.036, rf"Black: equal KL   |   Colors saturate at $\pm{args.color_limit:g}$", fontsize=9)
                if has_mask:
                    fig.text(0.12, 0.010, r"Gray: outside $\kappa_a>0$, $\kappa_m>-1/2$ convergence domain", fontsize=9, color="0.35")
                fig.savefig(out / f"{stem}.png", dpi=args.dpi)
                fig.savefig(out / f"{stem}.pdf", dpi=args.dpi)
                multipage.savefig(fig, dpi=args.dpi)
                plt.close(fig)

                mesh, _ = draw_panel(axes[i, j], ka, km, values, gamma, ratio, norm,
                                     compact=True, asymptotic=asymptotic)
                if i != len(args.gammas) - 1:
                    axes[i, j].set_xlabel("")
                if j != 0:
                    axes[i, j].set_ylabel("")
                for segment_id, segment in enumerate(segments):
                    for point_id, (a, m) in enumerate(segment):
                        boundary_rows.append((gamma, ratio, segment_id, point_id, a, m))
                summary.append({"gamma": gamma, "epsilon_m_over_epsilon_a": ratio,
                                "min_log10_kl_ratio": float(np.nanmin(values)),
                                "max_log10_kl_ratio": float(np.nanmax(values)),
                                "boundary_components_in_window": len(segments), "file_stem": stem})
                print(f"Saved {stem}: log10 KL ratio in [{np.nanmin(values):.3f}, {np.nanmax(values):.3f}]")
            fields.append(row_fields)

    overview.suptitle("Combined mean and shape errors: which sampler has lower KL?", fontsize=20, y=0.98)
    overview.text(0.5, 0.935, model_label, ha="center", fontsize=13)
    cb = overview.colorbar(mesh, cax=overview.add_axes([0.934, 0.20, 0.012, 0.62]), extend="both")
    cb.set_label(colorbar_label(), fontsize=13, labelpad=12)
    overview.text(0.052, 0.035, "Blue: stochastic sampler favored", color="#2166ac", fontsize=13)
    overview.text(0.31, 0.035, "Red: ODE favored", color="#b2182b", fontsize=13)
    overview.legend(handles=[Line2D([0], [0], color="#181818", lw=1.5, label="Equal KL"),
                             Line2D([0], [0], color="0.4", lw=0.8, ls="--", label=r"Reference: $\kappa_a=1$, $\kappa_m=0$")],
                    loc="lower center", bbox_to_anchor=(0.70, 0.017), ncol=2, frameon=False, fontsize=10)
    domain_note = r"  Gray: outside $\kappa_a>0$, $\kappa_m>-1/2$ convergence domain." if has_mask else ""
    overview.text(0.052, 0.009, rf"All panels share the color scale; colors saturate at $\pm{args.color_limit:g}$ (a factor $10^{{{args.color_limit:g}}}$)." + domain_note, fontsize=10)
    overview.savefig(out / "phase_diagrams_overview.png", dpi=args.dpi)
    overview.savefig(out / "phase_diagrams_overview.pdf", dpi=args.dpi)
    plt.close(overview)

    np.savez_compressed(out / "phase_diagram_data.npz", kappa_a=ka, kappa_m=km,
                        gammas=args.gammas, ratios=args.ratios, horizon=args.horizon,
                        valid_domain=valid_domain,
                        log10_kl_ratio=np.asarray(fields, dtype=np.float32))
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (out / "phase_boundaries.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "epsilon_m_over_epsilon_a", "component", "point", "kappa_a", "kappa_m"])
        writer.writerows(boundary_rows)
    metadata = {"source_pdf": DRAFT.name,
                "source_pdf_sha256": hashlib.sha256(DRAFT.read_bytes()).hexdigest() if DRAFT.exists() else None,
                "source_equations": [83, 84, 85] if asymptotic else [39, 40, 77, 83, 84],
                "approximation": "second order in error amplitudes, exact sampling prior",
                "horizon": "infinity" if asymptotic else args.horizon,
                "gammas": args.gammas, "ratios": args.ratios,
                "kappa_a_range": args.kappa_a, "kappa_m_range": args.kappa_m,
                "axis_scale": "linear", "array_order": ["gamma", "ratio", "kappa_m", "kappa_a"],
                "data_shape": list(np.asarray(fields).shape), "color_quantity": "log10(h_gamma^(2) / h_0^(2))",
                "saved_color_dtype": "float32 (calculations and contours use float64)",
                "invalid_domain_values": "NaN outside kappa_a > 0, kappa_m > -0.5 at infinite horizon",
                "masked_grid_points_per_panel": int(np.count_nonzero(~valid_domain)),
                "color_limits": [-args.color_limit, args.color_limit], "colormap": "RdBu_r",
                "data_clipped": False, "boundary_method": "linear grid contour at zero log10 KL ratio",
                "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "validation_passed": validation["passed"] if validation else None}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Overview, individual figures, multi-page PDF, data, and metadata saved to {out}")


if __name__ == "__main__":
    main()

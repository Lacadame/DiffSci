"""
test-correlation-thresholds.py

Scans a grid of time thresholds and, for each threshold combination, computes the
Pearson and Spearman correlations between:

  * improvement   I  vs.  early/middle error ratio   E_{early/mid}
  * deterioration D  vs.  late/total   error ratio   E_{late/tot}

for both KL directions  H(~p||p)  and  H(p||~p).

This reproduces the correlation computation of notebook
  entropy_paper/notebooks/02-entropy_paper-learned_scores.ipynb  (cells #67-#73)
but without any plotting.

Inputs are one or more directories, each of which must contain:
  all_entropies.pt  – dict: epoch -> (gamma_values, sde_entropies, inv_sde_entropies)
  all_errors.pt     – dict: epoch -> (error_values, dsm_losses)
(these are produced by test-time-profile-correlation.py)

The improvement / deterioration scalars depend only on the entropy curves, while
the error ratios depend on the thresholds:

  step_initial = step_from_time(initial_threshold)
  step_final   = step_from_time(final_threshold)
  step_late    = step_from_time(late_threshold)

  early_error  = sum(error_values[:step_initial])
  middle_error = sum(error_values[step_initial:step_final])
  late_error   = sum(error_values[step_late:])
  total_error  = sum(error_values)

  E_{early/mid} = early_error / middle_error      (uses initial & final thresholds)
  E_{late/tot}  = late_error  / total_error       (uses late threshold)

Because larger times map to smaller step indices in the EDM scheduler, a valid
(initial, final) pair requires  initial_threshold > final_threshold  (so that
step_initial < step_final and the "middle" slice is non-empty). The script
therefore builds a TRIANGULAR grid over the (initial, final) ranges: it takes the
cartesian product of the two ranges and keeps only the pairs satisfying that
constraint. The late-threshold grid is scanned INDEPENDENTLY.

All resulting correlations are written to a single CSV file inside an output
folder created next to this script.
"""

import argparse
import csv
import pathlib

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import diffsci.models


# KL direction labels, matching the notebook order:
#   index 0 -> sde_entropies      -> H(~p||p)
#   index 1 -> inv_sde_entropies  -> H(p||~p)
KL_NAMES = ["H(~p|p)", "H(p|~p)"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def iter_saved_items(saved_list):
    """Yield (key, value) from each dict in saved_list, preserving sorted keys.

    Mirrors the helper used in the notebook so that the ordering of data points
    is identical.
    """
    for saved_dict in saved_list:
        for key, value in sorted(saved_dict.items()):
            yield key, value


def load_saved(paths):
    """Load all_entropies.pt / all_errors.pt from each path.

    Returns (all_entropies_saved, all_errors_saved) as lists of dicts, one per
    input path.
    """
    all_entropies_saved = []
    all_errors_saved = []
    for p in paths:
        p = pathlib.Path(p)
        ent_path = p / "all_entropies.pt"
        err_path = p / "all_errors.pt"
        if not ent_path.exists():
            raise FileNotFoundError(f"Missing {ent_path}")
        if not err_path.exists():
            raise FileNotFoundError(f"Missing {err_path}")
        all_entropies_saved.append(torch.load(ent_path))
        all_errors_saved.append(torch.load(err_path))
    return all_entropies_saved, all_errors_saved


# ─────────────────────────────────────────────────────────────────────────────
# Scalar metrics (independent of thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def compute_improvements_deteriorations(all_entropies_saved):
    """Replicate notebook cell #68.

    Returns:
        epochs        : list of epoch keys (in iteration order)
        improvements  : np.ndarray [n, 2]   (KL index along axis 1)
        deteriorations: np.ndarray [n, 2]   (log-transformed, as in the notebook)
    """
    improvements = []
    deteriorations = []
    epochs = []
    for epoch, g_ent in iter_saved_items(all_entropies_saved):
        gamma_values, sde_entropies, inv_sde_entropies = g_ent
        improvement = []
        deterioration = []
        for ent in [sde_entropies, inv_sde_entropies]:
            ent = [float(e) for e in ent]
            min_ent = min(ent)
            improvement.append((ent[0] - min_ent) / min_ent)
            deterioration.append((ent[-1] - min_ent) / min_ent)
        improvements.append(improvement)
        deteriorations.append(deterioration)
        epochs.append(epoch)

    improvements = np.array(improvements, dtype=float)
    deteriorations = np.array(deteriorations, dtype=float)
    deteriorations = np.log(deteriorations)
    return epochs, improvements, deteriorations


# ─────────────────────────────────────────────────────────────────────────────
# Error ratios (depend on thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def compute_error_arrays(all_errors_saved):
    """Return a list of np.ndarray error_values, one per data point (cell #69)."""
    error_arrays = []
    for _epoch, (error_values, _dsm_losses) in iter_saved_items(all_errors_saved):
        error_arrays.append(np.asarray(error_values, dtype=float))
    return error_arrays


def early_mid_ratios(error_arrays, step_initial, step_final):
    """E_{early/mid} = sum(err[:step_initial]) / sum(err[step_initial:step_final])."""
    ratios = []
    for err in error_arrays:
        early = np.sum(err[:step_initial])
        middle = np.sum(err[step_initial:step_final])
        ratios.append(early / middle)
    return np.asarray(ratios, dtype=float)


def late_ratios(error_arrays, step_late):
    """E_{late/tot} = sum(err[step_late:]) / sum(err)."""
    ratios = []
    for err in error_arrays:
        late = np.sum(err[step_late:])
        total = np.sum(err)
        ratios.append(late / total)
    return np.asarray(ratios, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Correlations
# ─────────────────────────────────────────────────────────────────────────────

def safe_corr(x, y):
    """Pearson r/p and Spearman rho/p, returning NaNs on degenerate input."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    try:
        pr, pp = pearsonr(x, y)
    except Exception:
        pr, pp = np.nan, np.nan
    try:
        sr, sp = spearmanr(x, y)
    except Exception:
        sr, sp = np.nan, np.nan
    return pr, pp, sr, sp


# ─────────────────────────────────────────────────────────────────────────────
# Threshold grids
# ─────────────────────────────────────────────────────────────────────────────

def build_range(spec):
    """spec = (start, stop, num) -> np.linspace(start, stop, int(num))."""
    start, stop, num = spec
    return np.linspace(float(start), float(stop), int(num))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    scheduler = diffsci.models.EDMScheduler()
    nsteps = args.nsteps

    def step_of(t):
        return int(scheduler.step_from_time(torch.tensor(float(t)), nsteps).item())

    # ── Load data ───────────────────────────────────────────────────────────
    all_entropies_saved, all_errors_saved = load_saved(args.paths)
    epochs, improvements, deteriorations = compute_improvements_deteriorations(
        all_entropies_saved
    )
    error_arrays = compute_error_arrays(all_errors_saved)

    n_ent = len(epochs)
    n_err = len(error_arrays)
    if n_ent != n_err:
        raise ValueError(
            f"Mismatch between #entropy points ({n_ent}) and #error points "
            f"({n_err}); each path's all_entropies.pt and all_errors.pt must "
            f"have matching epochs."
        )
    print(f"Loaded {n_ent} data point(s) from {len(args.paths)} path(s).")

    # ── Epoch mask (cell #73: mask = arange(n) > epoch_threshold) ────────────
    mask = np.arange(n_ent) > args.epoch_threshold
    improvements_f = improvements[mask]
    deteriorations_f = deteriorations[mask]
    error_arrays_f = [e for i, e in enumerate(error_arrays) if mask[i]]
    n = int(mask.sum())
    print(f"After epoch mask (index > {args.epoch_threshold}): n = {n}")

    # ── Build threshold grids ───────────────────────────────────────────────
    initial_grid = build_range(args.initial_threshold_range)
    final_grid = build_range(args.final_threshold_range)
    late_grid = build_range(args.late_threshold_range)

    print(f"initial_threshold grid: {initial_grid}")
    print(f"final_threshold   grid: {final_grid}")
    print(f"late_threshold    grid: {late_grid}")

    rows = []

    # ── Triangular (initial, final) grid -> improvement correlations ─────────
    # Valid combination requires initial_threshold > final_threshold, which
    # guarantees step_initial < step_final (larger time -> smaller step index).
    n_early_combos = 0
    for it in initial_grid:
        for ft in final_grid:
            if not (it > ft):
                continue
            step_initial = step_of(it)
            step_final = step_of(ft)
            if step_initial >= step_final:
                # Degenerate after discretization: empty middle slice.
                continue
            x = early_mid_ratios(error_arrays_f, step_initial, step_final)
            n_early_combos += 1
            for j, kl_name in enumerate(KL_NAMES):
                pr, pp, sr, sp = safe_corr(x, improvements_f[:, j])
                rows.append({
                    "type": "early_mid_vs_improvement",
                    "initial_threshold": it,
                    "final_threshold": ft,
                    "late_threshold": np.nan,
                    "step_initial": step_initial,
                    "step_final": step_final,
                    "step_late": -1,
                    "kl_index": j,
                    "kl_name": kl_name,
                    "n": n,
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_r": sr,
                    "spearman_p": sp,
                })

    # ── Independent late-threshold grid -> deterioration correlations ────────
    n_late_combos = 0
    for lt in late_grid:
        step_late = step_of(lt)
        x = late_ratios(error_arrays_f, step_late)
        n_late_combos += 1
        for j, kl_name in enumerate(KL_NAMES):
            pr, pp, sr, sp = safe_corr(x, deteriorations_f[:, j])
            rows.append({
                "type": "late_vs_deterioration",
                "initial_threshold": np.nan,
                "final_threshold": np.nan,
                "late_threshold": lt,
                "step_initial": -1,
                "step_final": -1,
                "step_late": step_late,
                "kl_index": j,
                "kl_name": kl_name,
                "n": n,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sr,
                "spearman_p": sp,
            })

    print(f"Triangular (initial, final) combinations evaluated: {n_early_combos}")
    print(f"Late-threshold values evaluated:                   {n_late_combos}")
    print(f"Total correlation rows (x2 KL directions):         {len(rows)}")

    # ── Save single CSV file ────────────────────────────────────────────────
    output_dir = pathlib.Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / args.output_file

    fieldnames = [
        "type",
        "initial_threshold", "final_threshold", "late_threshold",
        "step_initial", "step_final", "step_late",
        "kl_index", "kl_name", "n",
        "pearson_r", "pearson_p", "spearman_r", "spearman_p",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved all correlations to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Scan time-threshold grids and compute Pearson/Spearman correlations "
            "between improvement/deterioration scalars and error ratios."
        )
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["../stats/output_default3/", "../stats/output_default4/", "../stats/output_default5/"],
        # default=["../stats/output_default6/", "../stats/output_default7/", "../stats/output_default8/"],
        help=(
            "One or more directories, each containing all_entropies.pt and "
            "all_errors.pt (e.g. /home/ubuntu/repos/DiffSci/scripts/testing/output_default3/)."
        ),
    )
    parser.add_argument(
        "--initial_threshold_range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "NUM"),
        default=[0.5, 0.025, 20],
        help="linspace(START, STOP, NUM) of initial_threshold values.",
    )
    parser.add_argument(
        "--final_threshold_range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "NUM"),
        default=[0.2, 0.01, 20],
        help="linspace(START, STOP, NUM) of final_threshold values.",
    )
    parser.add_argument(
        "--late_threshold_range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "NUM"),
        default=[0.5, 0.025, 20],
        help="linspace(START, STOP, NUM) of late_threshold values (scanned independently).",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=500,
        help="Number of diffusion steps used to map time thresholds to step indices.",
    )
    parser.add_argument(
        "--epoch_threshold",
        type=int,
        default=-1,
        help="Keep only data points with iteration index > epoch_threshold (cell #73).",
    )
    parser.add_argument(
        "--output_dir",
        default="../stats/correlation_thresholds_output",
        help="Output folder (created next to this script).",
    )
    parser.add_argument(
        "--output_file",
        default="correlations.csv",
        # default="correlations_heldout.csv",
        help="Name of the single CSV file collecting all correlations.",
    )
    args = parser.parse_args()
    main(args)

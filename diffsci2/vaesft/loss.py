"""SFT loss terms.

L_total = w_bce * BCE_with_logits(logits, x_real)
        + w_reg * mean_t [ omega_t * Huber( (R(x_soft)_t - y_true_t) / y_std_t , beta ) ]

The BCE anchor uses the same `logits = T * z(x_hat)` that feeds the regressor
input through a sigmoid, so the calibration of the soft binary input and the
pixel anchor are tied to the same temperature.

The regressor-head Huber is **strictly per-target z-scored before Huber**:
the four targets differ by orders of magnitude (e.g. surface_area_density
~1e-2 vs euler_number_density ~5e-5). Without dividing by the regressor's
`y_std` first, the large-scale targets dominate and the small-scale ones
contribute nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HuberCfg:
    beta: float = 1.0
    # Per-target weights for the 4 PoreRegressor outputs, ordered:
    #   (surface_area_density, mean_pore_size, mean_curvature, euler_number_density)
    # Weights are normalized to sum to len(targets) so the default
    # (1, 1, 1, 1) is exactly an unweighted mean.
    target_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)


def bce_pixel_anchor(logits: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    """Mean BCE-with-logits over voxels. `logits` is the same `T * z(x_hat)`
    used to build the regressor input via sigmoid.

    NOTE (blob diagnosis): the reduction is a plain mean over ALL voxels of
    the chunk. A boundary face is ~1/256 of a 256^3 chunk, so even a fully
    wrong face moves this anchor by only ~0.4%. Combined with the global-
    affine invariance of `logits` (see `regressor.normalize_to_logits`),
    this leaves the decoder's low-frequency output field effectively
    unconstrained near the chunk boundary -- the mechanism behind the s8
    binarization blob. This function is intentionally left as the plain
    global anchor; the field-flatness term is added separately.
    """
    return F.binary_cross_entropy_with_logits(logits, x_real)


def regressor_huber(
    y_pred: torch.Tensor,    # [B, T] regressor.predict_raw(x_soft)
    y_true: torch.Tensor,    # [B, T] cached morphology truth in raw units
    y_std: torch.Tensor,     # [T]    regressor.y_std buffer
    cfg: HuberCfg,
) -> torch.Tensor:
    """Per-target z-scored Huber, weighted mean across targets and batch.

    NOTE (blob diagnosis): `y_pred` is the regressor applied to the WHOLE
    chunk, so this term rewards only global morphology aggregates. Like the
    BCE anchor it is near-insensitive to a single boundary face and does
    not constrain the decoder's low-frequency field. See
    `regressor.normalize_to_logits`.
    """
    err_z = (y_pred - y_true) / y_std[None, :]
    huber = F.smooth_l1_loss(err_z, torch.zeros_like(err_z),
                              reduction="none", beta=cfg.beta)  # [B, T]
    w = torch.tensor(cfg.target_weights, dtype=huber.dtype, device=huber.device)
    w = w / w.sum() * w.numel()  # normalize so equal weights == unweighted mean
    return (huber * w[None, :]).mean()


def regressor_z_err_mean(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute z-scored error across batch and targets. Unit-free; useful
    as a single scalar to monitor with ModelCheckpoint."""
    return ((y_pred - y_true) / y_std[None, :]).abs().mean()


# ---------------------------------------------------------------------------
# Anti-drift terms (s8 PixelNorm binarization-blob fixes).
#
# Why these exist: both terms above (`bce_pixel_anchor`, `regressor_huber`)
# reduce `x_hat` to a GLOBAL aggregate. A boundary face is ~1/256 of a 256^3
# chunk, so the decoder's low-frequency output field is essentially
# unconstrained near the faces. SFT grows a smooth ramp (measured per-slice-
# mean spread 0.18 raw -> 0.37 sft); eval thresholds at the single global
# mean; the faces flip into a solid "blob". Full diagnosis in
# `diffsci2/vaesft/regressor.py::normalize_to_logits`.
#
# The two functions below are independent ways to pin that field. Both are
# additive loss terms carrying their own constant weight; they default OFF
# (weight 0.0 in `SFTConfig`) so existing runs are bit-for-bit unchanged.
# ---------------------------------------------------------------------------

# channel + D + H + W; mirrors `regressor.SPATIAL_DIMS` (kept local to avoid
# a circular import between loss.py and regressor.py).
_SPATIAL_DIMS = (1, 2, 3, 4)


def field_flatness_penalty(x_hat: torch.Tensor, kernel: int = 16) -> torch.Tensor:
    """Option B -- penalize the low-frequency spatial drift of `x_hat`.

    Mechanism, step by step:
      1. `avg_pool3d` over non-overlapping `kernel`^3 tiles extracts the local
         mean field -- exactly the low-frequency component the ramp lives in
         (high-frequency pore detail is averaged away inside each tile).
      2. `Var` of that pooled field across tiles is 0 iff the field is
         spatially flat, and grows monotonically as a ramp develops.
      3. Dividing by the per-chunk variance of `x_hat` makes the penalty
         (a) AFFINE-INVARIANT -- a pure global offset/scale cancels, so this
             term does NOT fight the harmless global drift, only the spatial
             ramp; and
         (b) dimensionless -- so the same weight transfers across runs whose
             decoder output happens to sit at a different scale.

    Self-contained: needs no reference decoder. Eval is unchanged -- once the
    field is flat, thresholding at the single global mean is correct again.

    `kernel` must divide the training chunk size (e.g. 16 or 8 for 64^3).
    """
    pooled = F.avg_pool3d(x_hat, kernel_size=kernel)            # [B,C,d,h,w]
    field_var = pooled.var(dim=_SPATIAL_DIMS, unbiased=False)   # [B] low-freq energy
    chunk_var = x_hat.var(dim=_SPATIAL_DIMS, unbiased=False)    # [B] total energy
    return (field_var / (chunk_var + 1e-8)).mean()


def raw_consistency_anchor(
    x_hat: torch.Tensor,
    x_hat_raw: torch.Tensor,
    lowpass_kernel: int = 0,
) -> torch.Tensor:
    """Option A -- pin the low-frequency *shape* of `x_hat` to the frozen raw
    (pre-SFT) decoder output.

    `x_hat_raw` is produced by a frozen deep-copy of the decoder taken before
    any SFT step (see `VAESFTModule.decoder_raw`). The pretrained decoder has
    only a small field (measured per-slice-mean spread 0.18 vs 0.37 after the
    standard SFT); anchoring toward its low-frequency field keeps SFT from
    growing the ramp.

    `lowpass_kernel == 0`  -> compare at full resolution (pins the whole
                              shape, including high-frequency detail).
    `lowpass_kernel  > 0`  -> compare only `avg_pool3d`-ed fields: pins just
                              the low-frequency component (the ramp) and
                              leaves high-frequency detail free for SFT to
                              keep improving. Recommended.

    The MSE is taken on MEAN-CENTERED fields -- the per-chunk global mean is
    removed from both sides first. This is essential: without it the MSE is
    dominated (~98%) by the harmless global-offset drift (mean 0.80 -> 1.58),
    and almost no gradient reaches the ramp that actually causes the blob.
    Centering makes the anchor offset-invariant, so all of its gradient
    targets the spatial ramp. Dividing by the raw field's own variance makes
    the term dimensionless (~1 when the SFT field deviates about as much as
    the raw field itself), so `w_raw_anchor` is interpretable across runs.
    """
    a, b = x_hat, x_hat_raw
    if lowpass_kernel > 0:
        a = F.avg_pool3d(a, kernel_size=lowpass_kernel)
        b = F.avg_pool3d(b, kernel_size=lowpass_kernel)
    # mean-center: drop the harmless global offset, keep only the ramp shape.
    a = a - a.mean(dim=_SPATIAL_DIMS, keepdim=True)
    b = b - b.mean(dim=_SPATIAL_DIMS, keepdim=True)
    denom = b.var(dim=_SPATIAL_DIMS, unbiased=False).mean() + 1e-8
    return F.mse_loss(a, b) / denom

"""Frozen PoreRegressor3D + the regressor-input transform.

The SFT loss applies the frozen regressor to a soft "near-binary" surrogate
of the decoder output. The transform is:

    z(x_hat) = (x_hat - mean_chunk) / (std_chunk + eps)        # per-chunk z-score
    logits   = clamp(T * z(x_hat), [-LOGIT_CLAMP, LOGIT_CLAMP])
    x_soft   = sigmoid(logits)                                  # in (0, 1), near-binary at high T

At T = 5 (default), sigmoid saturates within |z| >~ 1, so the regressor sees a
field that lies in [sigmoid(-5), sigmoid(+5)] = [~7e-3, ~1 - 7e-3] -- close to
the binary {0, 1} on which it was trained. The same `logits` are also the
input to the auxiliary BCE-with-logits pixel anchor against the real chunk.

`FrozenRegressor` is a thin nn.Module wrapper around `PoreRegressor3D` plus
the (y_mean, y_std) buffers. It is what the SFT LightningModule registers as
a submodule, so the regressor parameters live on the right device and stay
out of the gradient sync (all parameters have requires_grad=False).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# `_paths` puts the poreregressor scripts dir on sys.path so these imports
# resolve. The regressor itself stays as an external sibling package for v1
# (see PLAN §6.3).
from . import _paths  # noqa: F401
from poreregressor.dataset import compute_target_stats
from poreregressor.lightning_module import PoreRegressorModule


SPATIAL_DIMS = (1, 2, 3, 4)
LOGIT_CLAMP = 30.0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class FrozenRegressor(nn.Module):
    """Thin frozen wrapper around `PoreRegressor3D` with raw-unit prediction.

    Holds the model and the (y_mean, y_std) buffers as a single nn.Module so
    Lightning moves them to the right device with the parent module. All
    parameters have requires_grad=False, so DDP ignores this submodule in the
    gradient sync.
    """

    def __init__(self, lit: PoreRegressorModule):
        super().__init__()
        self.model = lit.model
        self.register_buffer("y_mean", lit.y_mean.detach().clone())
        self.register_buffer("y_std", lit.y_std.detach().clone())
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized predictions, shape [B, NUM_TARGETS]."""
        return self.model(x)

    def predict_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Predictions in raw target units, shape [B, NUM_TARGETS]. Differentiable."""
        y_norm = self.model(x)
        return y_norm * self.y_std[None, :] + self.y_mean[None, :]


def load_frozen_regressor(
    ckpt_path: str | None = None,
    train_targets_path: str | None = None,
    map_location: str | torch.device = "cpu",
) -> FrozenRegressor:
    """Load the frozen PoreRegressor3D from a Lightning checkpoint.

    If paths are not given, defaults to the run02 R²=0.98 ckpt and its
    matching cached train targets under
    `notebooks/exploratory/dfnai/scripts/poreregressor/`.
    """
    if ckpt_path is None:
        ckpt_path = _paths.DEFAULT_REGRESSOR_CKPT
    if train_targets_path is None:
        train_targets_path = _paths.DEFAULT_REGRESSOR_TRAIN_TARGETS

    targets = np.load(train_targets_path)
    stats = compute_target_stats(targets)
    lit = PoreRegressorModule.load_from_checkpoint(
        ckpt_path,
        target_stats=stats,
        map_location=map_location,
    )
    return FrozenRegressor(lit)


# ---------------------------------------------------------------------------
# Input transform
# ---------------------------------------------------------------------------

def normalize_to_logits(
    x_hat: torch.Tensor,
    temperature: float,
    eps_std: float = 1e-6,
    logit_clamp: float = LOGIT_CLAMP,
) -> torch.Tensor:
    """Per-chunk z-score, scale by temperature, clamp to a safe range.

    `mean`/`std` reduce over SPATIAL_DIMS = (channel, D, H, W), i.e. they
    collapse the WHOLE chunk to one scalar per batch element. Two
    consequences matter for the SFT objective -- both diagnosed while
    chasing the s8 PixelNorm binarization "blob":

    (1) GLOBAL-AFFINE INVARIANCE.  z(a*x_hat + b) == z(x_hat) for any
        a > 0, b. Every downstream loss term is built on `logits` (the
        BCE anchor) or on `sigmoid(logits)` (the regressor reward), so the
        SFT loss cannot see the absolute scale/offset of the decoder
        output at all. The decoder is free to drift its output level --
        observed per-chunk mean 0.80 (raw s8) -> 1.58 (sft s8). This part
        is HARMLESS: eval re-binarizes at the per-chunk mean,
        `(x_hat > x_hat.mean())`, which is itself affine-invariant.

    (2) LOW-FREQUENCY FIELD SURVIVES (this is the blob).  The z-score
        subtracts only the single global scalar `mean`. A spatially
        VARYING low-frequency field -- a smooth ramp across the chunk --
        is NOT removed; it passes straight into `logits`. It is therefore
        not invisible to the losses, but it is only weighted as a
        sub-region inside a *global mean* (see `loss.bce_pixel_anchor` and
        `loss.regressor_huber`): one boundary face is ~1/256 of a 256^3
        chunk, so its gradient contribution is negligible. SFT grows a
        low-frequency ramp essentially for free (measured per-slice-mean
        spread: 0.18 raw -> 0.37 sft). Because eval thresholds at the ONE
        global mean, the chunk faces -- at the extremes of that ramp --
        are mis-thresholded and binarize into a solid blob. This is NOT
        an architecture/resolution failure: a per-slice-local threshold
        collapses the worst-face error 3.99% -> 0.92%.

    Fix direction: add an SFT loss term that constrains the low-frequency
    field per-region, instead of letting both existing terms see it only
    through a global mean.
    """
    # Per-chunk (whole-volume) reduction -> one scalar mean/std per sample.
    mean = x_hat.mean(dim=SPATIAL_DIMS, keepdim=True)
    std = x_hat.std(dim=SPATIAL_DIMS, keepdim=True) + eps_std
    logits = temperature * (x_hat - mean) / std
    return logits.clamp(-logit_clamp, logit_clamp)


def near_binary(
    x_hat: torch.Tensor,
    temperature: float = 5.0,
    eps_std: float = 1e-6,
    logit_clamp: float = LOGIT_CLAMP,
) -> torch.Tensor:
    """Sigmoid of `temperature * z(x_hat)`. Differentiable near-binary surrogate."""
    logits = normalize_to_logits(x_hat, temperature, eps_std, logit_clamp)
    return torch.sigmoid(logits)


def deterministic_binary(
    x_hat: torch.Tensor,
    eps_std: float = 1e-6,
) -> torch.Tensor:
    """Hard sign-of-z threshold; matches eval-time binarization.

    This is the GLOBAL threshold: every voxel is compared against the single
    per-chunk mean. It is what the standard SFT pipeline trains against and
    what the s8 blob shows up under -- see `normalize_to_logits` for why.
    """
    mean = x_hat.mean(dim=SPATIAL_DIMS, keepdim=True)
    return (x_hat > mean).to(x_hat.dtype)


# ---------------------------------------------------------------------------
# Option C -- LOCAL z-score transform.
#
# `normalize_to_logits` above subtracts a single global scalar mean/std per
# chunk. As documented there, a smooth low-frequency ramp in `x_hat` survives
# that subtraction, and both SFT loss terms only ever see it diluted inside a
# global mean -> the s8 binarization blob.
#
# The local variant subtracts a *locally pooled* mean/std field instead. A
# ramp is, by construction, captured by the local mean and therefore removed
# from `logits` entirely: the BCE anchor and the regressor input both become
# blind to the low-frequency drift. The optimizer can then neither be helped
# nor hurt by the ramp, so it stops developing one; the remaining gradient
# pressure falls on genuine local structure.
#
# IMPORTANT -- this redefines "binarization". With a local z-score the
# decision surface is `x_hat > local_mean(x_hat)`, NOT the global mean. Eval
# MUST use `deterministic_binary_local` with the SAME kernel, or training and
# inference disagree on the threshold. `VAESFTModule` wires this switch
# automatically from `SFTConfig.zscore_mode`.
# ---------------------------------------------------------------------------


def _local_mean_std(
    x_hat: torch.Tensor,
    kernel: int,
    eps_std: float,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Locally-pooled mean and std fields, same shape as `x_hat`.

    Uses a stride-1 `avg_pool3d` with symmetric padding so the output keeps
    the input resolution. `count_include_pad=False` makes boundary windows
    average only real voxels (no zero-padding bias) -- this is what keeps the
    transform well-behaved exactly at the chunk faces, which is where the
    blob lives. Variance is E[x^2] - E[x]^2, clamped at 0 for fp safety.
    """
    pad = kernel // 2
    mean = F.avg_pool3d(x_hat, kernel, stride=1, padding=pad,
                        count_include_pad=False)
    mean_sq = F.avg_pool3d(x_hat * x_hat, kernel, stride=1, padding=pad,
                           count_include_pad=False)
    std = (mean_sq - mean * mean).clamp_min(0.0).sqrt() + eps_std
    return mean, std


def normalize_to_logits_local(
    x_hat: torch.Tensor,
    temperature: float,
    kernel: int,
    eps_std: float = 1e-6,
    logit_clamp: float = LOGIT_CLAMP,
) -> torch.Tensor:
    """Local-z-score counterpart of `normalize_to_logits` (Option C).

    `kernel` must be ODD so the stride-1 pooling with `padding=kernel//2`
    returns the input resolution exactly. A larger kernel removes only
    longer-wavelength drift (kernel -> infinity recovers the global z-score);
    a smaller kernel also flattens genuine medium-scale structure, so do not
    shrink it below the scale of real pore features.
    """
    assert kernel % 2 == 1, f"local z-score kernel must be odd, got {kernel}"
    mean, std = _local_mean_std(x_hat, kernel, eps_std)
    logits = temperature * (x_hat - mean) / std
    return logits.clamp(-logit_clamp, logit_clamp)


def deterministic_binary_local(
    x_hat: torch.Tensor,
    kernel: int,
) -> torch.Tensor:
    """Local-threshold counterpart of `deterministic_binary` (Option C).

    Thresholds each voxel against its local pooled mean instead of the single
    per-chunk mean. This MUST be used for eval/inference whenever the model
    was trained with `normalize_to_logits_local` at the same `kernel`, so the
    decision surface matches the one the SFT loss optimized.
    """
    assert kernel % 2 == 1, f"local threshold kernel must be odd, got {kernel}"
    pad = kernel // 2
    mean = F.avg_pool3d(x_hat, kernel, stride=1, padding=pad,
                        count_include_pad=False)
    return (x_hat > mean).to(x_hat.dtype)

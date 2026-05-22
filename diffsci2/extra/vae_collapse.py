"""Post-hoc latent-collapse detection and surgical z_dim reduction for VAEs.

Two utilities for VAE-family architectures (``VAENet`` and any
sub/sibling that mirrors its layout):

- :func:`detect_collapsed_channels` — encode a batch of inputs,
  return which mean-channels have a per-pixel σ below a threshold.
- :func:`stitch_z_dim_reduction` — given a trained VAE and a list of
  active channels, build a smaller VAE (``z_dim = len(active)``) and
  transplant weights. Only two 1×1 conv layers' weights are
  surgically modified (``quant_conv`` rows and ``post_quant_conv``
  columns); everything else copies verbatim.

These are pipeline tools — not part of any training loop. They live
under ``diffsci2.extra`` because they apply to *any* VAE checkpoint,
not just one specific architecture. Originally developed in
``notebooks/exploratory/dfnai/scripts/vaenewnorm/`` (see
``FINAL_REPORT.md`` § 6 for the motivating experiment).

Assumptions about the architecture:

- The encoder produces ``[B, 2*z_dim_old, *spatial]`` (mean ‖ logvar).
- The encoder has an attribute path ``encoder.quant_conv`` ending in
  a 1×1 ``nn.Conv?d`` whose weight has shape
  ``[2*z_dim_old, C_in, 1, 1, ...]`` and bias ``[2*z_dim_old]``.
- The decoder has ``decoder.post_quant_conv``: 1×1 ``nn.Conv?d``
  with weight ``[C_out, z_dim_old, 1, 1, ...]``.

If your VAE follows ``diffsci2.nets.VAENet`` you already have these.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

@dataclass
class CollapseReport:
    """Per-channel statistics of the encoder mean over a sample batch.

    Attributes
    ----------
    z_dim_full : int
        Number of mean channels in the encoder output.
    active : list[int]
        Channels whose per-pixel σ exceeded ``threshold``.
    collapsed : list[int]
        Channels whose σ was ≤ ``threshold``.
    threshold : float
        The σ cutoff used.
    per_channel_std, per_channel_mean : np.ndarray
        Statistics computed across pixels of all sampled inputs.
    """
    z_dim_full: int
    active: list[int]
    collapsed: list[int]
    threshold: float
    per_channel_std: np.ndarray
    per_channel_mean: np.ndarray

    @property
    def z_dim_effective(self) -> int:
        return len(self.active)


def detect_collapsed_channels(
    encoder: torch.nn.Module,
    inputs: Iterable[torch.Tensor],
    z_dim: int,
    threshold: float = 0.05,
    device: torch.device | str | None = None,
) -> CollapseReport:
    """Encode each ``x`` in ``inputs`` and statistics-fold the mean output.

    Parameters
    ----------
    encoder : nn.Module
        Anything whose forward returns ``[B, 2*z_dim, *spatial]`` (mean
        followed by logvar). The mean is the first ``z_dim`` channels.
    inputs : iterable of tensors
        Each shaped ``[1, C_in, *spatial]`` (or any batch size — values
        are flattened per channel before stats).
    z_dim : int
        How many channels of the encoder output to treat as the mean.
    threshold : float
        ``σ ≤ threshold`` flags a channel as collapsed.
    device : optional
        Move each input to this device before encoding. ``None`` means
        leave it on whatever device it's already on.

    Returns
    -------
    CollapseReport
        With ``active`` / ``collapsed`` lists ready to feed into
        :func:`stitch_z_dim_reduction`.
    """
    encoder.eval()
    flat_per_ch: list[list[np.ndarray]] = [[] for _ in range(z_dim)]
    with torch.no_grad():
        for x in inputs:
            if device is not None:
                x = x.to(device)
            out = encoder(x)
            mu = out[:, :z_dim]                     # [B, z_dim, *spatial]
            mu_np = mu.detach().cpu().numpy()
            for ch in range(z_dim):
                flat_per_ch[ch].append(mu_np[:, ch].ravel())

    per_ch_std = np.zeros(z_dim)
    per_ch_mean = np.zeros(z_dim)
    for ch in range(z_dim):
        vals = np.concatenate(flat_per_ch[ch]) if flat_per_ch[ch] else np.zeros(1)
        per_ch_std[ch] = float(vals.std())
        per_ch_mean[ch] = float(vals.mean())

    active = [ch for ch in range(z_dim) if per_ch_std[ch] > threshold]
    collapsed = [ch for ch in range(z_dim) if per_ch_std[ch] <= threshold]
    return CollapseReport(
        z_dim_full=z_dim,
        active=active,
        collapsed=collapsed,
        threshold=float(threshold),
        per_channel_std=per_ch_std,
        per_channel_mean=per_ch_mean,
    )


# --------------------------------------------------------------------------- #
# Surgery
# --------------------------------------------------------------------------- #

def stitch_z_dim_reduction(
    old_encdec: torch.nn.Module,
    active_channels: list[int],
    new_encdec_factory,
) -> torch.nn.Module:
    """Surgically reduce the latent dimensionality of a trained VAE.

    Parameters
    ----------
    old_encdec : nn.Module
        The trained VAE (any ``VAENet``-family object — must have
        ``encoder.quant_conv`` and ``decoder.post_quant_conv``).
    active_channels : list[int]
        Indices (in 0..z_dim_old) of the latent channels to keep.
    new_encdec_factory : callable
        Zero-arg callable returning a freshly-constructed VAE with
        ``z_dim = len(active_channels)`` and otherwise identical
        config. The caller is responsible for matching everything
        except z_dim — typically::

            def factory():
                cfg2 = copy.deepcopy(old_cfg)
                cfg2.z_dim = len(active_channels)
                return VAENet(cfg2)

    Returns
    -------
    new_encdec : nn.Module
        Loaded with the surgically-transplanted weights. ``z_dim`` is
        ``len(active_channels)``.

    Notes
    -----
    Two weights change shape:

    - ``encoder.quant_conv.weight``: rows
      ``active ∪ (active + z_dim_old)`` (μ rows for active channels,
      then logvar rows for the same active channels).
    - ``decoder.post_quant_conv.weight``: columns ``active``.

    All other parameters are copied verbatim. Bias on ``quant_conv``
    is sliced like the rows; bias on ``post_quant_conv`` is unchanged
    (output dim is z_channels, which is independent of z_dim).
    """
    new_encdec = new_encdec_factory()

    old_sd = old_encdec.state_dict()
    new_sd = new_encdec.state_dict()

    # Infer z_dim_old from the old encoder's quant_conv bias.
    quant_w_key, quant_bias_key = _find_quant_conv_keys(old_sd)
    z_dim_old = old_sd[quant_bias_key].shape[0] // 2
    z_dim_new = len(active_channels)
    if any(c < 0 or c >= z_dim_old for c in active_channels):
        raise ValueError(
            f"active_channels {active_channels} out of [0, {z_dim_old})"
        )

    keep_rows_quant = list(active_channels) + [c + z_dim_old for c in active_channels]

    post_w_key = _find_post_quant_conv_weight_key(old_sd)

    for k in new_sd:
        if k in (quant_w_key, quant_bias_key, post_w_key):
            continue
        if k not in old_sd:
            raise KeyError(f"new model has unexpected key: {k}")
        if new_sd[k].shape != old_sd[k].shape:
            raise ValueError(
                f"verbatim-copy shape mismatch on {k}: "
                f"old={tuple(old_sd[k].shape)} new={tuple(new_sd[k].shape)}"
            )
        new_sd[k] = old_sd[k].clone()

    new_sd[quant_w_key] = old_sd[quant_w_key][keep_rows_quant].clone()
    new_sd[quant_bias_key] = old_sd[quant_bias_key][keep_rows_quant].clone()
    new_sd[post_w_key] = old_sd[post_w_key][:, active_channels].clone()

    new_encdec.load_state_dict(new_sd, strict=True)
    return new_encdec


def _find_quant_conv_keys(sd: dict) -> tuple[str, str]:
    """Locate ``encoder.quant_conv`` weight/bias keys, tolerant of inner
    wrappers (e.g. ``PatchedConv`` which exposes ``.conv.weight``)."""
    bias_candidates = [k for k in sd
                       if 'quant_conv' in k and k.endswith('.bias')
                       and 'post_' not in k]
    if not bias_candidates:
        raise KeyError("no encoder.quant_conv bias found in state_dict; "
                       "this architecture isn't supported")
    if len(bias_candidates) > 1:
        raise KeyError(f"ambiguous quant_conv bias candidates: {bias_candidates}")
    bkey = bias_candidates[0]
    wkey = bkey[:-len('.bias')] + '.weight'
    if wkey not in sd:
        raise KeyError(f"expected {wkey!r} matching {bkey!r}; not found")
    return wkey, bkey


def _find_post_quant_conv_weight_key(sd: dict) -> str:
    candidates = [k for k in sd
                  if 'post_quant_conv' in k and k.endswith('.weight')]
    if not candidates:
        raise KeyError("no decoder.post_quant_conv.weight found in state_dict")
    if len(candidates) > 1:
        raise KeyError(f"ambiguous post_quant_conv.weight candidates: {candidates}")
    return candidates[0]

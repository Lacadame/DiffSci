"""Finite-receptive-field probe for trained encoder / decoder networks.

For an architecture with a *truly* local spatial receptive field (e.g.
the ``VAENetConfig(norm_type='pixel')`` path), the output at a given
pixel depends only on the input within a bounded window. This module
exposes a programmatic check of that property:

- :func:`encoder_central_equivalence` — encode a full input and a
  centre-with-halo sub-input; verify the central latent matches.
- :func:`decoder_central_equivalence` — same in the other direction.

Both return a small ``dict`` of summary statistics. On CPU with fp32
and a PixelNorm-only architecture, the differences should be 0.00e+00.
On GPU there is cuDNN 3D-conv non-determinism (~1e-2 slack) — use CPU
to claim "finite RF" with certainty.

Motivating context: the existing GroupNorm-based ``VAENet`` does
**not** have this property; its receptive field is global because
GroupNorm normalises across all spatial positions. The
``cached_norms.gather_norm_stats_full`` pipeline papers over this by
running a calibration pass on the full input before each tiled
forward. The MP/PixelNorm variant doesn't need that.

These functions are architecture-agnostic — they just call the
provided ``encoder`` / ``decoder`` callables.
"""
from __future__ import annotations

import torch


def _check_int_div(value: int, divisor: int, name: str) -> None:
    if value % divisor != 0:
        raise ValueError(
            f"{name} ({value}) must be a multiple of {divisor}"
        )


@torch.no_grad()
def encoder_central_equivalence(
    encoder,
    x_full: torch.Tensor,
    halo_pixels: int,
    stride: int,
) -> dict:
    """Compare ``encoder(x_full)`` against the encoding of a central
    sub-region plus a halo.

    Parameters
    ----------
    encoder : callable
        Network that maps ``[1, C, L, L, L]`` → ``[1, C', L_lat, L_lat, L_lat]``.
        Spatial extent must be cubic and divisible by ``stride``.
    x_full : torch.Tensor
        Input tensor ``[1, C, L, L, L]``.
    halo_pixels : int
        Per-side halo in *input* pixel units. Must be a multiple of
        ``stride``. Pick ≥ ½ × the empirical input-space RF.
    stride : int
        Spatial downsampling factor of the encoder (e.g. 4 for a
        config with ``ch_mult=[1, 2, 4]``).

    Returns
    -------
    dict
        Keys ``central_latent_shape``, ``sub_input_shape``,
        ``max_abs_diff``, ``p99_9_diff``, ``p50_diff``,
        ``full_central_l2``.
    """
    _check_int_div(halo_pixels, stride, "halo_pixels")
    if x_full.ndim < 4:
        raise ValueError(f"x_full must be [B, C, ...] with spatial dims, "
                         f"got shape {tuple(x_full.shape)}")
    L = x_full.shape[-1]

    z_full = encoder(x_full)
    L_lat = z_full.shape[-1]
    if L != L_lat * stride:
        raise ValueError(
            f"expected encoder stride {stride}, got L={L}, L_lat={L_lat}"
        )

    lat_lo, lat_hi = L_lat // 4, 3 * L_lat // 4
    in_lo_central, in_hi_central = lat_lo * stride, lat_hi * stride

    in_lo = max(0, in_lo_central - halo_pixels)
    in_hi = min(L, in_hi_central + halo_pixels)
    x_sub = x_full[..., in_lo:in_hi, in_lo:in_hi, in_lo:in_hi].contiguous()
    z_sub = encoder(x_sub)

    sub_lat_lo = (in_lo_central - in_lo) // stride
    sub_lat_hi = sub_lat_lo + (lat_hi - lat_lo)
    z_sub_central = z_sub[..., sub_lat_lo:sub_lat_hi,
                          sub_lat_lo:sub_lat_hi,
                          sub_lat_lo:sub_lat_hi]
    z_full_central = z_full[..., lat_lo:lat_hi,
                            lat_lo:lat_hi, lat_lo:lat_hi]

    diff = (z_sub_central - z_full_central).abs()
    return {
        'central_latent_shape': tuple(z_full_central.shape),
        'sub_input_shape': tuple(x_sub.shape),
        'max_abs_diff': diff.max().item(),
        'p99_9_diff': torch.quantile(diff.flatten(), 0.999).item(),
        'p50_diff': torch.quantile(diff.flatten(), 0.5).item(),
        'full_central_l2': z_full_central.pow(2).mean().sqrt().item(),
    }


@torch.no_grad()
def decoder_central_equivalence(
    decoder,
    z_full: torch.Tensor,
    halo_latent: int,
    stride: int,
) -> dict:
    """Mirror of :func:`encoder_central_equivalence` for a decoder.

    Parameters
    ----------
    decoder : callable
        Network that maps ``[1, C, L_lat, L_lat, L_lat]`` →
        ``[1, C', L, L, L]`` with ``L = L_lat * stride``.
    z_full : torch.Tensor
        Input latent ``[1, C, L_lat, L_lat, L_lat]``.
    halo_latent : int
        Per-side halo in *latent* pixel units. Pick ≥ ½ × the
        empirical latent-space RF.
    stride : int
        Spatial upsampling factor (e.g. 4).
    """
    L_lat = z_full.shape[-1]
    x_full = decoder(z_full)
    L = x_full.shape[-1]
    if L != L_lat * stride:
        raise ValueError(
            f"expected decoder stride {stride}, got L={L}, L_lat={L_lat}"
        )

    out_lo, out_hi = L // 4, 3 * L // 4
    lat_lo_central, lat_hi_central = out_lo // stride, out_hi // stride

    lat_lo = max(0, lat_lo_central - halo_latent)
    lat_hi = min(L_lat, lat_hi_central + halo_latent)
    z_sub = z_full[..., lat_lo:lat_hi, lat_lo:lat_hi, lat_lo:lat_hi].contiguous()
    x_sub = decoder(z_sub)

    sub_out_lo = (lat_lo_central - lat_lo) * stride
    sub_out_hi = sub_out_lo + (out_hi - out_lo)
    x_sub_central = x_sub[..., sub_out_lo:sub_out_hi,
                          sub_out_lo:sub_out_hi, sub_out_lo:sub_out_hi]
    x_full_central = x_full[..., out_lo:out_hi,
                            out_lo:out_hi, out_lo:out_hi]

    diff = (x_sub_central - x_full_central).abs()
    return {
        'central_output_shape': tuple(x_full_central.shape),
        'sub_latent_shape': tuple(z_sub.shape),
        'max_abs_diff': diff.max().item(),
        'p99_9_diff': torch.quantile(diff.flatten(), 0.999).item(),
        'p50_diff': torch.quantile(diff.flatten(), 0.5).item(),
        'full_central_l2': x_full_central.pow(2).mean().sqrt().item(),
    }

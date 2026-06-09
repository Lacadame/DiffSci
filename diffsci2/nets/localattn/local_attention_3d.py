"""3D local (neighborhood) attention via NATTEN's na3d.

Mirrors :mod:`local_attention_2d` with one extra spatial dim. Only the
NATTEN backend is implemented (mask-on-full is hopeless at 3D shapes).
Periodic mode via circular pad-and-crop halo on all three axes.

Optional radial PE uses ``_radial_dist_bins_3d``: for a K^3 window the
Euclidean distances generalise the 2D D4 to the cubic point group
**Oh** (48 elements = 24 rotations + reflections), which is the
discrete subgroup of SO(3) compatible with a cubic lattice and matches
the data-augmentation group used elsewhere in the project (CAGEO
Algorithm 1). For K=3, num_bins=4 (distances 0, 1, sqrt(2), sqrt(3)).

Shape convention: external tensors are ``[B, C, D, H, W]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _radial_dist_bins_3d(kernel_size: int) -> tuple[list[int], int]:
    """K=3 -> {0, 1, sqrt(2), sqrt(3)}, nbins=4 over K^3=27 offsets."""
    R = (kernel_size - 1) // 2
    dist_sq = []
    for dz in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                dist_sq.append(dz * dz + dy * dy + dx * dx)
    unique = sorted(set(dist_sq))
    return [unique.index(d) for d in dist_sq], len(unique)


class RadialBias3D(nn.Module):
    """Learnable per-head bias indexed by 3D Euclidean distance within
    the K^3 window. For K=3: 4 bins. Preserves the cubic point group
    Oh (48 elements) — matches the data-augmentation symmetry.
    """

    def __init__(self, num_heads: int, kernel_size: int):
        super().__init__()
        idx, nbins = _radial_dist_bins_3d(kernel_size)
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.num_bins = nbins
        self.register_buffer(
            'dist_index_flat',
            torch.tensor(idx, dtype=torch.long),
        )  # [K^3]
        self.bias_table = nn.Parameter(torch.zeros(num_heads, nbins))

    @property
    def per_offset(self) -> torch.Tensor:
        """[num_heads, K^3] bias for na3d's K^3-flat sparse scores."""
        return self.bias_table[:, self.dist_index_flat]


class LocalSelfAttention3D(nn.Module):
    """NATTEN-backed 3D local attention with optional radial PE.

    ``[B, C, D, H, W] -> [B, C, D, H, W]``.

    Memory profile (B=32, base_dim=64, K=3, heads=4, bf16, fwd+bwd+AdamW,
    32^3 latent, single 40 GB A100):

      noPE, fused na3d                     15.5 GB
      noPE, fused na3d + periodic halo     ~15.5 GB (linear scale)
      radial PE (split path)               ~15.5 GB (linear scale)
      radial PE + periodic                 ~15.5 GB (linear scale)

    32^3 latent at B=32 is comfortable on a 40 GB card. 3D port is
    memory-unblocked.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int = 3,
        periodic: bool = False,
        radial_pe: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.periodic = periodic
        self.radial_pe = radial_pe

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rbias = RadialBias3D(num_heads, kernel_size) if radial_pe else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from natten.functional import na3d, na3d_qk, na3d_av
        if self.periodic:
            r = (self.kernel_size - 1) // 2
            x = F.pad(x, (r, r, r, r, r, r), mode='circular')
        x_bdhwc = rearrange(x, "b c d h w -> b d h w c")
        qkv = self.qkv(x_bdhwc)
        q, k, v = rearrange(
            qkv, "b d h w (three nh hd) -> three b d h w nh hd",
            three=3, nh=self.num_heads, hd=self.head_dim,
        ).unbind(0)
        K = self.kernel_size
        if self.rbias is None:
            out = na3d(q, k, v, kernel_size=K)
        else:
            attn = na3d_qk(q, k, kernel_size=K, dilation=1)
            bias = self.rbias.per_offset.to(attn.dtype)  # [heads, K^3]
            attn = attn + bias[None, None, None, None, :, :]
            attn = attn.softmax(dim=-1)
            out = na3d_av(attn, v, kernel_size=K, dilation=1)
        out = rearrange(out, "b d h w nh hd -> b d h w (nh hd)")
        out = self.proj(out)
        out = rearrange(out, "b d h w c -> b c d h w")
        if self.periodic:
            r = (self.kernel_size - 1) // 2
            out = out[..., r:-r, r:-r, r:-r]
        return out


def modulate3d(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    """``x * (1 + scale) + shift`` broadcast over D, H, W.

    This is feature-wise linear modulation (FiLM; Perez et al. 2018,
    "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI),
    whose style-transfer ancestor is AdaIN (Huang & Belongie 2017) /
    StyleGAN (Karras et al. 2019).
    """
    return (
        x * (1.0 + scale[:, :, None, None, None])
        + shift[:, :, None, None, None]
    )


class AdaLN3D(nn.Module):
    """adaLN-Zero conditioning block — 3D analog of
    :class:`local_attention_2d.AdaLN`.

    Lineage:

    - **Modulation** (shift + scale regressed from a conditioning
      vector) is FiLM / AdaIN — see :func:`modulate3d`.
    - The full **adaLN-Zero** recipe — regress *three* quantities
      ``(shift, scale, gate)`` from the conditioning embedding and
      zero-initialise the projection so the whole block is the identity
      at step 0 — is from **DiT** (Peebles & Xie 2023, "Scalable
      Diffusion Models with Transformers", ICCV, Sec. 3 / Fig. 3). The
      ``gate`` (DiT's ``alpha``) scales the *residual-branch output*
      (see :class:`LocalAttentionBlock3D`), not the input.
    - The zero-initialised residual gate itself predates DiT: ReZero
      (Bachlechner et al. 2020) and the "zero-init the last scale"
      trick (Goyal et al. 2017). DiT fuses these with FiLM modulation.

    Note: ``elementwise_affine=False`` — the LayerNorm carries no learned
    gamma/beta because the affine is supplied by ``(shift, scale)``. The
    norm runs over the channel axis, i.e. per-voxel across channels.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(cond_dim, 3 * dim)
        # adaLN-Zero: zero-init weight AND bias so proj(cond) == 0 for any
        # cond at step 0 -> shift=scale=gate=0 -> the block is the identity
        # (the attention/MLP branch is gated off; the residual passes
        # through untouched). Training lifts the gate off zero.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_b = rearrange(x, "b c d h w -> b d h w c")
        x_b = self.norm(x_b)
        x = rearrange(x_b, "b d h w c -> b c d h w")
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)
        return modulate3d(x, shift, scale), gate


class MLP3D(nn.Module):
    """Pointwise MLP via Linear (== 1x1x1 conv). No nn.Conv3d."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.fc2(self.act(self.fc1(x)))
        return rearrange(x, "b d h w c -> b c d h w")


class LocalAttentionBlock3D(nn.Module):
    """3D transformer block: adaLN-Zero, local-attn, adaLN-Zero, MLP.

    DiT block (Peebles & Xie 2023), conv-free 3D port. Each sub-layer is
    ``x = x + gate * sublayer(modulate(norm(x), shift, scale))``. With the
    adaLN-Zero init (see :class:`AdaLN3D`) ``gate == 0`` at step 0, so the
    prior is literally "nothing happens" — the residual stream is the
    identity and the network learns each branch's contribution from there.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        kernel_size: int = 3,
        periodic: bool = False,
        mlp_ratio: float = 4.0,
        radial_pe: bool = False,
    ):
        super().__init__()
        self.norm_attn = AdaLN3D(dim, cond_dim)
        self.attn = LocalSelfAttention3D(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            periodic=periodic,
            radial_pe=radial_pe,
        )
        self.norm_mlp = AdaLN3D(dim, cond_dim)
        self.mlp = MLP3D(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # h is modulate(norm(x)): at init shift=scale=0 so h == norm(x)
        # (NOT zero) — the branch is silenced by the gate below, not by
        # the modulation. gate==0 at init => x unchanged.
        h, gate_a = self.norm_attn(x, cond)
        x = x + gate_a[:, :, None, None, None] * self.attn(h)
        h, gate_m = self.norm_mlp(x, cond)
        x = x + gate_m[:, :, None, None, None] * self.mlp(h)
        return x

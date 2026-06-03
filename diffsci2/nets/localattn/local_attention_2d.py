"""2D local (neighborhood) attention.

Two backends:

- ``backend='mask'``: dense [B, heads, N, N] attention with a Chebyshev
  (K x K) sparsity mask. Trivial at small latent shapes (e.g. 16x16 =
  256 tokens) but quadratic in N — OOM for ~128px and beyond.
- ``backend='natten'``: NATTEN's ``na2d`` / ``na2d_qk`` / ``na2d_av``
  fused kernels. K^2-sparse intermediates, linear in N.

Periodicity at inference:

A model trained aperiodic can be flipped to periodic at inference by
walking the module tree and toggling ``periodic`` on every
``LocalSelfAttention2D``. See :func:`set_periodic`.

Shape convention: external tensors are ``[B, C, H, W]`` to stay
consistent with the rest of DiffSci2.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -- Radial positional bias ----------------------------------------------------


def _radial_dist_bins(kernel_size: int) -> tuple[list[int], int]:
    """For a K x K window with offsets in row-major order
    ``(dy, dx) in [-R, R]^2`` (``dy`` outer, ``dx`` inner), return
    per-offset distance-bin index and the total number of distinct
    Euclidean distances.

    K=3 (R=1): distances [sqrt(2), 1, sqrt(2), 1, 0, 1, sqrt(2), 1, sqrt(2)]
    -> sorted unique [0, 1, sqrt(2)] -> indices [2, 1, 2, 1, 0, 1, 2, 1, 2],
    nbins=3.
    """
    R = (kernel_size - 1) // 2
    dist_sq = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            dist_sq.append(dy * dy + dx * dx)
    unique = sorted(set(dist_sq))
    return [unique.index(d) for d in dist_sq], len(unique)


class RadialBias(nn.Module):
    """Learnable per-head attention bias indexed only by Euclidean
    distance to the query within the local window. Preserves the D4 +
    translation symmetry of the operator. For K=3, num_bins=3
    (distances 0, 1, sqrt(2)).
    """

    def __init__(self, num_heads: int, kernel_size: int):
        super().__init__()
        idx, nbins = _radial_dist_bins(kernel_size)
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.num_bins = nbins
        self.register_buffer(
            'dist_index_flat',
            torch.tensor(idx, dtype=torch.long),
        )  # [K^2]
        # Zero-init: model starts as NoPE and learns to deviate.
        self.bias_table = nn.Parameter(torch.zeros(num_heads, nbins))

    @property
    def per_offset(self) -> torch.Tensor:
        """[num_heads, K^2] bias laid out for NATTEN's flat K^2 scores."""
        return self.bias_table[:, self.dist_index_flat]

    def dense_bias(
        self, H: int, W: int, periodic: bool, device: torch.device,
    ) -> torch.Tensor:
        """[num_heads, N, N] dense bias for the mask backend.

        For each (i, j) pair, looks up the distance bin from the
        (wrapped, if periodic) |dy|, |dx|. Out-of-window pairs are
        assigned bin 0; their entries are masked out by the Chebyshev
        mask anyway.
        """
        R = (self.kernel_size - 1) // 2
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        qy, qx = torch.meshgrid(ys, xs, indexing="ij")
        q_idx = torch.stack([qy.reshape(-1), qx.reshape(-1)], dim=-1)
        k_idx = q_idx
        dy = (q_idx[:, None, 0] - k_idx[None, :, 0]).abs()
        dx = (q_idx[:, None, 1] - k_idx[None, :, 1]).abs()
        if periodic:
            dy = torch.minimum(dy, H - dy)
            dx = torch.minimum(dx, W - dx)
        in_window = (dy <= R) & (dx <= R)
        offset_idx = (dy + R) * (2 * R + 1) + (dx + R)
        offset_idx = torch.where(
            in_window, offset_idx, torch.zeros_like(offset_idx),
        )
        bin_idx = self.dist_index_flat[offset_idx.clamp(0, (2 * R + 1) ** 2 - 1)]
        return self.bias_table[:, bin_idx]  # [heads, N, N]


def chebyshev_local_mask(
    H: int,
    W: int,
    kernel_size: int,
    periodic: bool,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """[N, N] bool mask. True = key is in the K x K neighborhood of query.

    Non-periodic: boundary queries lose keys outside the grid (simple
    variant — *not* NATTEN's inward-shifted window). Periodic: wraps
    with modular Chebyshev distance.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    R = (kernel_size - 1) // 2

    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    qy, qx = torch.meshgrid(ys, xs, indexing="ij")
    ky, kx = qy, qx
    q_idx = torch.stack([qy.reshape(-1), qx.reshape(-1)], dim=-1)
    k_idx = torch.stack([ky.reshape(-1), kx.reshape(-1)], dim=-1)

    dy = (q_idx[:, None, 0] - k_idx[None, :, 0]).abs()
    dx = (q_idx[:, None, 1] - k_idx[None, :, 1]).abs()
    if periodic:
        dy = torch.minimum(dy, H - dy)
        dx = torch.minimum(dx, W - dx)
    return (dy <= R) & (dx <= R)


class LocalSelfAttention2D(nn.Module):
    """Multi-head 2D local self-attention.

    Parameters
    ----------
    dim: int
        Input/output channels.
    num_heads: int
        Must divide ``dim``.
    kernel_size: int
        Odd integer K. Each query attends to a K x K neighborhood.
    periodic: bool
        If True, the neighborhood wraps modulo the grid. Mutable at
        runtime: a model trained aperiodic can be flipped to periodic
        inference with :func:`set_periodic`.
    backend: ``'mask'`` or ``'natten'``
        Implementation. ``'mask'`` is dense + Chebyshev mask (works at
        small N). ``'natten'`` requires the ``natten`` package and uses
        the fused ``na2d`` kernel (linear in N).
    radial_pe: bool
        If True, add a learnable D4-symmetric per-head bias indexed by
        Euclidean distance in the K-window. Default False (NoPE).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int = 3,
        periodic: bool = False,
        backend: str = 'mask',
        radial_pe: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert backend in ('mask', 'natten')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.periodic = periodic
        self.backend = backend
        self.radial_pe = radial_pe

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rbias = RadialBias(num_heads, kernel_size) if radial_pe else None

        self._mask_cache: dict[tuple[int, int, int, bool], torch.Tensor] = {}

    def _get_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, self.kernel_size, self.periodic)
        cached = self._mask_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        mask = chebyshev_local_mask(
            H, W, self.kernel_size, self.periodic, device=device,
        )
        self._mask_cache[key] = mask
        return mask

    def _forward_mask(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        qkv = self.qkv(x_flat)
        q, k, v = rearrange(
            qkv, "b n (three h d) -> three b h n d",
            three=3, h=self.num_heads, d=self.head_dim,
        ).unbind(0)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        if self.rbias is not None:
            attn = attn + self.rbias.dense_bias(
                H, W, self.periodic, x.device,
            ).unsqueeze(0)
        mask = self._get_mask(H, W, x.device)
        attn = attn.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return rearrange(self.proj(out), "b (h w) c -> b c h w", h=H, w=W)

    def _forward_natten(self, x: torch.Tensor) -> torch.Tensor:
        """NATTEN-backed local attention.

        Shape walkthrough (N = H*W, K = kernel_size):

          x_bhwc        [B, H, W, dim]           — channels-last view
          qkv           [B, H, W, 3*dim]         — single Linear
          q, k, v       [B, H, W, heads, head_d] — split

        Two paths depending on whether a radial-PE bias must be
        injected between QK and softmax:

        (A) ``self.rbias is None``: fused
              out = na2d(q, k, v, K)            [B, H, W, heads, head_d]
            Internally na2d tiles the K^2-sparse score block and never
            materializes any score tensor in user memory. Cheapest
            memory profile.

        (B) ``self.rbias is set``: split
              attn = na2d_qk(q, k, K)           [B, H, W, heads, K^2]
              attn = attn + radial_bias[None,None,None,:,:]
              attn = softmax(attn, dim=-1)
              out  = na2d_av(attn, v, K)        [B, H, W, heads, head_d]
            ``attn`` here is K^2-sparse, NOT a full [B, heads, N, N]
            matrix. For 128^2 pixel with B=32, heads=4, K=3:
              attn ~ 32 * 128^2 * 4 * 9 * 2B  ~  38 MB    (cheap)
            Compare mask-on-full at the same shape:
              QK^T ~ 32 * 4 * 16384^2 * 2B    ~  68.7 GB  (OOM)

        Empirical peak memory at the LA-UNet pixel level-0 shape
        (B=32, dim=64, heads=4, 128^2 spatial, K=3, fp16, fwd+bwd):
              natten fused        (no PE)         1317.8 MB
              natten split        (radial PE)      793.0 MB
              natten split + periodic              816.4 MB
              mask-on-full                         OOM

        Periodicity: when ``self.periodic`` we pad the input with a
        circular halo of width ``r = (K - 1) / 2``, run aperiodic
        NATTEN on the padded tensor, and crop the halo off the output.
        The interior of na2d remains aperiodic; the wrap-around is
        purely in the padding.
        """
        from natten.functional import na2d, na2d_qk, na2d_av
        if self.periodic:
            r = (self.kernel_size - 1) // 2
            x = F.pad(x, (r, r, r, r), mode='circular')
        B, C, H, W = x.shape
        x_bhwc = rearrange(x, "b c h w -> b h w c")
        qkv = self.qkv(x_bhwc)
        q, k, v = rearrange(
            qkv, "b h w (three nh d) -> three b h w nh d",
            three=3, nh=self.num_heads, d=self.head_dim,
        ).unbind(0)
        if self.rbias is None:
            out = na2d(q, k, v, kernel_size=self.kernel_size)
        else:
            attn = na2d_qk(q, k, kernel_size=self.kernel_size)
            bias = self.rbias.per_offset.to(attn.dtype)  # [heads, K^2]
            attn = attn + bias[None, None, None, :, :]
            attn = attn.softmax(dim=-1)
            out = na2d_av(attn, v, kernel_size=self.kernel_size)
        out = rearrange(out, "b h w nh d -> b h w (nh d)")
        out = self.proj(out)
        out = rearrange(out, "b h w c -> b c h w")
        if self.periodic:
            r = (self.kernel_size - 1) // 2
            out = out[..., r:-r, r:-r]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == 'natten':
            return self._forward_natten(x)
        return self._forward_mask(x)


# -- AdaLN modulation helpers (DiT-style) --------------------------------------


def modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    """``x * (1 + scale) + shift`` broadcast over the spatial axes.

    ``x``: [B, C, H, W]; ``shift``, ``scale``: [B, C].
    """
    return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class AdaLN(nn.Module):
    """Channel-axis LayerNorm + adaLN-Zero modulation.

    LayerNorm over the channel axis of a [B, C, H, W] tensor, followed
    by a (shift, scale, gate) triple computed from a 1D conditioning
    embedding. The projection is zero-initialised so the block is the
    identity at step zero (DiT recipe).
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_bhwc = rearrange(x, "b c h w -> b h w c")
        x_bhwc = self.norm(x_bhwc)
        x = rearrange(x_bhwc, "b h w c -> b c h w")
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)
        return modulate(x, shift, scale), gate  # gate: [B, C]


class MLP(nn.Module):
    """Pointwise MLP via Linear (== 1x1 conv). No nn.Conv2d."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b h w c")
        x = self.fc2(self.act(self.fc1(x)))
        return rearrange(x, "b h w c -> b c h w")


class LocalAttentionBlock2D(nn.Module):
    """Transformer block: adaLN-Zero, local-attn, adaLN-Zero, MLP.

    With adaLN-Zero init the gates are zero at step zero, so the block
    is the identity initially. This is the standard DiT recipe.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        kernel_size: int = 3,
        periodic: bool = False,
        mlp_ratio: float = 4.0,
        backend: str = 'mask',
        radial_pe: bool = False,
    ):
        super().__init__()
        self.norm_attn = AdaLN(dim, cond_dim)
        self.attn = LocalSelfAttention2D(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            periodic=periodic,
            backend=backend,
            radial_pe=radial_pe,
        )
        self.norm_mlp = AdaLN(dim, cond_dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h, gate_attn = self.norm_attn(x, cond)
        x = x + gate_attn[:, :, None, None] * self.attn(h)
        h, gate_mlp = self.norm_mlp(x, cond)
        x = x + gate_mlp[:, :, None, None] * self.mlp(h)
        return x


def set_periodic(model: nn.Module, periodic: bool) -> int:
    """Walk the module tree and set ``periodic`` on every
    ``LocalSelfAttention2D`` (and ``LocalSelfAttention3D`` if present).

    Returns the number of layers flipped.
    """
    # Local import to avoid a hard dependency cycle and to keep 3D
    # optional at import time.
    try:
        from .local_attention_3d import LocalSelfAttention3D
    except Exception:
        LocalSelfAttention3D = None  # type: ignore[assignment]

    n = 0
    for m in model.modules():
        if isinstance(m, LocalSelfAttention2D):
            m.periodic = periodic
            m._mask_cache.clear()
            n += 1
        elif LocalSelfAttention3D is not None and isinstance(m, LocalSelfAttention3D):
            m.periodic = periodic
            n += 1
    return n

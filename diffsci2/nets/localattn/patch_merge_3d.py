"""Conv-free 2x up/down sampling for 3D feature maps.

Mirror of :mod:`patch_merge`. 2x2x2 tiles instead of 2x2.

- :class:`PatchMerge3D`:   ``[B, C, D, H, W] -> [B, C', D/2, H/2, W/2]``
- :class:`PatchUnmerge3D`: ``[B, C, D, H, W] -> [B, C', 2D, 2H, 2W]``
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class PatchMerge3D(nn.Module):
    """Halve all three spatial dimensions via a Linear over a 2x2x2 tile.

    Default channel growth: ``C_out = 2 * C_in``.
    """

    def __init__(self, in_dim: int, out_dim: int | None = None):
        super().__init__()
        if out_dim is None:
            out_dim = 2 * in_dim
        self.proj = nn.Linear(8 * in_dim, out_dim, bias=True)
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, (
            f"PatchMerge3D needs even D, H, W; got {(D, H, W)}"
        )
        x = rearrange(
            x, "b c (d p1) (h p2) (w p3) -> b d h w (p1 p2 p3 c)",
            p1=2, p2=2, p3=2,
        )
        x = self.proj(x)
        return rearrange(x, "b d h w c -> b c d h w")


class PatchUnmerge3D(nn.Module):
    """Double all three spatial dimensions via a Linear.

    Default channel shrink: ``C_out = C_in // 2``.
    """

    def __init__(self, in_dim: int, out_dim: int | None = None):
        super().__init__()
        if out_dim is None:
            assert in_dim % 2 == 0
            out_dim = in_dim // 2
        self.proj = nn.Linear(in_dim, 8 * out_dim, bias=True)
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.proj(x)
        return rearrange(
            x, "b d h w (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)",
            p1=2, p2=2, p3=2,
        )

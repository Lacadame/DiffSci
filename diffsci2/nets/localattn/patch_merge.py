"""Conv-free 2x up/down sampling for 2D feature maps.

- :class:`PatchMerge2D`:   ``[B, C_in, H, W] -> [B, C_out, H/2, W/2]``
  via 2x2 token concat along the channel axis + Linear projection.
- :class:`PatchUnmerge2D`: ``[B, C_in, H, W] -> [B, C_out, 2H, 2W]``
  via Linear expansion to ``4 * C_out`` then rearrange the channels
  back into a 2x2 spatial tile.

No ``nn.Conv2d``. Standard Swin V2 / DiT recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class PatchMerge2D(nn.Module):
    """Halve spatial resolution via a learned linear projection over a
    2x2 tile.

    Default channel growth: ``C_out = 2 * C_in`` (matches a stride-2
    conv halving).
    """

    def __init__(self, in_dim: int, out_dim: int | None = None):
        super().__init__()
        if out_dim is None:
            out_dim = 2 * in_dim
        self.proj = nn.Linear(4 * in_dim, out_dim, bias=True)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, (
            f"PatchMerge2D needs even H, W; got {(H, W)}"
        )
        x = rearrange(
            x, "b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=2, p2=2,
        )  # [B, H/2, W/2, 4*C]
        x = self.proj(x)
        return rearrange(x, "b h w c -> b c h w")


class PatchUnmerge2D(nn.Module):
    """Double spatial resolution via a learned linear projection.

    Default channel shrink: ``C_out = C_in // 2`` (mirror of
    :class:`PatchMerge2D`).
    """

    def __init__(self, in_dim: int, out_dim: int | None = None):
        super().__init__()
        if out_dim is None:
            assert in_dim % 2 == 0, "default out_dim = in_dim // 2 needs even"
            out_dim = in_dim // 2
        self.proj = nn.Linear(in_dim, 4 * out_dim, bias=True)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b h w c")
        x = self.proj(x)
        return rearrange(
            x, "b h w (p1 p2 c) -> b c (h p1) (w p2)", p1=2, p2=2,
        )

"""LA-UNet 3D: hierarchical UNet with NATTEN-backed local attention.

Conv-free 3D mirror of :class:`LAUNet`. Down/up via :class:`PatchMerge3D`
/ :class:`PatchUnmerge3D`. adaLN-Zero time conditioning.
``[B, C, D, H, W]`` in/out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from .local_attention_3d import LocalAttentionBlock3D
from .patch_merge_3d import PatchMerge3D, PatchUnmerge3D


# -- Time embedding (3D-internal copies; symmetric with la_unet.py) -----------


class GaussianFourierProjection3D(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        assert embed_dim % 2 == 0
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t[..., None]
        proj = 2.0 * math.pi * t * self.W
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class TimeMLP3D(nn.Module):
    def __init__(
        self, cond_dim: int, fourier_dim: int = 256, scale: float = 30.0,
    ):
        super().__init__()
        self.fourier = GaussianFourierProjection3D(fourier_dim, scale=scale)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier(t))


def _zero_linear(in_dim: int, out_dim: int) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim, bias=True)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _ChannelLinear3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.linear(x)
        return rearrange(x, "b d h w c -> b c d h w")


# -- Config --------------------------------------------------------------------


@dataclass
class LAUNet3DConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 64
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_blocks_per_level: int = 2
    num_heads_base: int = 4
    kernel_size: int = 3
    mlp_ratio: float = 4.0
    periodic: bool = False
    cond_dim: int = 256
    fourier_dim: int = 256
    radial_pe: bool = False


# -- Network -------------------------------------------------------------------


class LAUNet3D(nn.Module):
    """Hierarchical conv-free 3D UNet driven by NATTEN local attention."""

    def __init__(self, config: LAUNet3DConfig):
        super().__init__()
        self.config = config
        assert config.base_channels % config.num_heads_base == 0

        head_dim_base = config.base_channels // config.num_heads_base
        ch = [config.base_channels * m for m in config.ch_mult]
        heads = [c // head_dim_base for c in ch]
        n_levels = len(ch)

        self.time_mlp = TimeMLP3D(config.cond_dim, config.fourier_dim)
        self.stem = _ChannelLinear3D(config.in_channels, ch[0])

        def make_block(li: int) -> LocalAttentionBlock3D:
            return LocalAttentionBlock3D(
                dim=ch[li],
                num_heads=heads[li],
                cond_dim=config.cond_dim,
                kernel_size=config.kernel_size,
                periodic=config.periodic,
                mlp_ratio=config.mlp_ratio,
                radial_pe=config.radial_pe,
            )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for li in range(n_levels):
            self.down_blocks.append(nn.ModuleList(
                [make_block(li) for _ in range(config.num_blocks_per_level)],
            ))
            if li < n_levels - 1:
                self.downsamples.append(PatchMerge3D(ch[li], ch[li + 1]))

        self.mid_blocks = nn.ModuleList(
            [make_block(n_levels - 1) for _ in range(2)],
        )

        self.upsamples = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for li in reversed(range(n_levels - 1)):
            self.upsamples.append(PatchUnmerge3D(ch[li + 1], ch[li]))
            self.up_proj.append(_ChannelLinear3D(2 * ch[li], ch[li]))
            self.up_blocks.append(nn.ModuleList(
                [make_block(li) for _ in range(config.num_blocks_per_level)],
            ))

        self.final_norm = nn.LayerNorm(ch[0], eps=1e-6)
        self.out_proj_linear = _zero_linear(ch[0], config.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del y
        cond = self.time_mlp(t)
        h = self.stem(x)
        skips = []
        for li, blocks in enumerate(self.down_blocks):
            for blk in blocks:
                h = blk(h, cond)
            skips.append(h)
            if li < len(self.downsamples):
                h = self.downsamples[li](h)
        for blk in self.mid_blocks:
            h = blk(h, cond)
        for up, proj, blocks, skip in zip(
            self.upsamples,
            self.up_proj,
            self.up_blocks,
            reversed(skips[:-1]),
        ):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = proj(h)
            for blk in blocks:
                h = blk(h, cond)
        h_b = rearrange(h, "b c d h w -> b d h w c")
        h_b = self.final_norm(h_b)
        out = self.out_proj_linear(h_b)
        return rearrange(out, "b d h w c -> b c d h w")

"""LA-UNet: a 2D hierarchical U-Net with local attention everywhere.

Conv-free: no ``nn.Conv2d`` anywhere in the module tree. Up/down via
:class:`PatchMerge2D` / :class:`PatchUnmerge2D`. Skip connections via
channel-axis concat + Linear projection. Time conditioning via Gaussian
Fourier projection + MLP -> adaLN-Zero gates inside every block.

Input/output: ``[B, in_channels, H, W] -> [B, out_channels, H, W]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from .local_attention_2d import LocalAttentionBlock2D
from .patch_merge import PatchMerge2D, PatchUnmerge2D


# -- Time embedding ------------------------------------------------------------


class GaussianFourierProjection(nn.Module):
    """Sinusoidal projection of a scalar ``t`` to ``R^{embed_dim}``."""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        assert embed_dim % 2 == 0
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [...] -> [..., embed_dim]
        t = t[..., None]
        proj = 2.0 * math.pi * t * self.W
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class TimeMLP(nn.Module):
    """``t -> Fourier -> MLP -> cond`` embedding."""

    def __init__(
        self, cond_dim: int, fourier_dim: int = 256, scale: float = 30.0,
    ):
        super().__init__()
        self.fourier = GaussianFourierProjection(fourier_dim, scale=scale)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier(t))


# -- Helpers -------------------------------------------------------------------


def _zero_linear(in_dim: int, out_dim: int) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim, bias=True)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _ChannelLinear2D(nn.Module):
    """1x1 spatial 'conv' implemented as a Linear over the channel axis.

    Operates on ``[B, C, H, W]`` tensors.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b h w c")
        x = self.linear(x)
        return rearrange(x, "b h w c -> b c h w")


# -- Config --------------------------------------------------------------------


@dataclass
class LAUNetConfig:
    in_channels: int = 4
    out_channels: int = 4
    base_channels: int = 64
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_blocks_per_level: int = 2
    num_heads_base: int = 4
    kernel_size: int = 3
    mlp_ratio: float = 4.0
    periodic: bool = False
    cond_dim: int = 256
    fourier_dim: int = 256
    backend: str = 'mask'  # 'mask' or 'natten'
    radial_pe: bool = False


# -- Network -------------------------------------------------------------------


class LAUNet(nn.Module):
    """Hierarchical conv-free UNet driven by local attention."""

    def __init__(self, config: LAUNetConfig):
        super().__init__()
        self.config = config

        # Heads scale with channel multiplier so head_dim stays fixed.
        head_dim_base = config.base_channels // config.num_heads_base
        assert config.base_channels % config.num_heads_base == 0

        ch = [config.base_channels * m for m in config.ch_mult]
        heads = [c // head_dim_base for c in ch]
        n_levels = len(ch)

        self.time_mlp = TimeMLP(
            cond_dim=config.cond_dim, fourier_dim=config.fourier_dim,
        )
        self.stem = _ChannelLinear2D(config.in_channels, ch[0])

        def make_block(li: int) -> LocalAttentionBlock2D:
            return LocalAttentionBlock2D(
                dim=ch[li],
                num_heads=heads[li],
                cond_dim=config.cond_dim,
                kernel_size=config.kernel_size,
                periodic=config.periodic,
                mlp_ratio=config.mlp_ratio,
                backend=config.backend,
                radial_pe=config.radial_pe,
            )

        # Down path.
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for li in range(n_levels):
            self.down_blocks.append(nn.ModuleList(
                [make_block(li) for _ in range(config.num_blocks_per_level)],
            ))
            if li < n_levels - 1:
                self.downsamples.append(PatchMerge2D(ch[li], ch[li + 1]))

        # Bottleneck (extra pair at the deepest level).
        self.mid_blocks = nn.ModuleList(
            [make_block(n_levels - 1) for _ in range(2)],
        )

        # Up path: unmerge, concat skip, project, blocks.
        self.upsamples = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for li in reversed(range(n_levels - 1)):
            self.upsamples.append(PatchUnmerge2D(ch[li + 1], ch[li]))
            self.up_proj.append(_ChannelLinear2D(2 * ch[li], ch[li]))
            self.up_blocks.append(nn.ModuleList(
                [make_block(li) for _ in range(config.num_blocks_per_level)],
            ))

        self.final_norm = nn.LayerNorm(
            ch[0], elementwise_affine=True, eps=1e-6,
        )
        self.out_proj_linear = _zero_linear(ch[0], config.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            raise ValueError("y is not supported for LAUNet yet")
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

        # Skips list has one entry per level; bottleneck-level skip was
        # consumed by mid_blocks above, so iterate in reverse over
        # skips[:-1].
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

        h_bhwc = rearrange(h, "b c h w -> b h w c")
        h_bhwc = self.final_norm(h_bhwc)
        out = self.out_proj_linear(h_bhwc)
        return rearrange(out, "b h w c -> b c h w")

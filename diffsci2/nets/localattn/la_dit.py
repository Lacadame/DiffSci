"""LA-DiT: a flat (single-resolution) DiT-style transformer with local
attention.

We are already in latent space, so no patchify step is needed — each
latent voxel is one token. Pure Linear, zero ``nn.Conv2d``, no spatial
downsampling. Time conditioning via adaLN-Zero inside every block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from .local_attention_2d import LocalAttentionBlock2D
from .la_unet import TimeMLP, _ChannelLinear2D, _zero_linear


@dataclass
class LADitConfig:
    in_channels: int = 4
    out_channels: int = 4
    embed_dim: int = 128
    depth: int = 8
    num_heads: int = 4
    kernel_size: int = 3
    mlp_ratio: float = 4.0
    periodic: bool = False
    cond_dim: int = 256
    fourier_dim: int = 256
    backend: str = 'mask'  # 'mask' or 'natten'
    radial_pe: bool = False


class LADit(nn.Module):
    """Flat DiT with local attention everywhere."""

    def __init__(self, config: LADitConfig):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.config = config

        self.time_mlp = TimeMLP(
            cond_dim=config.cond_dim, fourier_dim=config.fourier_dim,
        )
        self.stem = _ChannelLinear2D(config.in_channels, config.embed_dim)
        self.blocks = nn.ModuleList([
            LocalAttentionBlock2D(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                cond_dim=config.cond_dim,
                kernel_size=config.kernel_size,
                periodic=config.periodic,
                mlp_ratio=config.mlp_ratio,
                backend=config.backend,
                radial_pe=config.radial_pe,
            )
            for _ in range(config.depth)
        ])
        self.final_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.out_proj_linear = _zero_linear(
            config.embed_dim, config.out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del y
        cond = self.time_mlp(t)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h, cond)
        h_bhwc = rearrange(h, "b c h w -> b h w c")
        h_bhwc = self.final_norm(h_bhwc)
        out = self.out_proj_linear(h_bhwc)
        return rearrange(out, "b h w c -> b c h w")

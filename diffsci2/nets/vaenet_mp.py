"""VAENetMP — magnitude-preserving variant of `diffsci2.nets.vaenet.VAENet`.

Same encoder / mid / decoder skeleton, but:

- Every `nn.GroupNorm` is replaced by `PixelNorm` (per-pixel channel-RMS).
  Placement is SYMMETRIC: same positions in both encoder and decoder,
  because chunked inference applies to both networks.
- The remaining EDM2 CONFIG-G ingredients (MP-SiLU, MP-Sum, output Gain)
  are wired in via flags so we can bisect Approaches A → B → C:

      flag           A (start)   B            C (full EDM2-G)
      -------------  ----------  -----------  --------------
      pixel_norm        always   always       always
      mp_silu              off   on           on
      mp_sum               off   on           on
      mp_conv              off   off          on   (not supported here)
      forced_wn            off   off          on   (gated by mp_conv)
      output_gain          on    on           on   (init=1.0)

  Approach C requires `MPConv` (a dim-generic magnitude-preserving conv
  with forced weight-normalization). It is not currently ported into
  `diffsci2.nets`; instantiating a config with `mp_conv=True` raises
  `NotImplementedError`. The s8 production VAE uses approach_b.

- `quant_conv` and `post_quant_conv` (both 1x1) are kept as bare
  `nn.Conv?d` so the latent (mean, logvar) split is not re-scaled by the
  MP machinery. The KL term in the VAE loss shapes them.
- `Gain(init=1.0)` is placed right before the decoder output so the
  network has explicit control over the output magnitude. Init 1.0 (not
  0.0 like Karras' score nets) because we predict x, not a noise residual.
- No attention. We rely on the standard config having `has_mid_attn=False`
  and `attn_resolutions=[]`. If attention were added later it would
  break the finite-RF property and these classes would have to refuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normedlayers import PixelNorm, Gain, mp_silu, mp_sum


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class VAENetMPConfig:
    """Mirrors the parts of `VAENetConfig` we need, plus the MP flags."""
    # Geometry
    dimension: int = 2
    in_channels: int = 1
    out_channels: int = 1
    z_channels: int = 4
    z_dim: int = 4
    ch: int = 32
    ch_mult: List[int] = field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: int = 2
    resolution: int = 128
    resamp_with_conv: bool = True
    tanh_out: bool = False
    double_z: bool = True

    # MP toggles
    pixel_norm: bool = True          # PixelNorm replaces GroupNorm (always on)
    mp_silu: bool = False            # rescale SiLU output by 1/0.596
    mp_sum: bool = False             # weighted residual sum (t=0.3)
    mp_conv: bool = False            # MPConv (not supported here — approach_c)
    forced_wn: bool = False          # forced weight normalization step
    output_gain_init: float = 1.0    # learned scalar at decoder output
    mp_sum_t: float = 0.3            # residual blend factor (EDM2 default)

    # Loss-time bias / conv bias (kept on in our VAE since no conditioning
    # pathway provides per-channel offsets).
    input_bias: bool = True
    output_bias: bool = True
    conv_bias: bool = True           # bias on all hidden convs

    @property
    def num_resolutions(self) -> int:
        return len(self.ch_mult)

    @property
    def norm_type(self) -> str:
        """Same contract as `VAENetConfig.norm_type` so downstream code
        (e.g. the cached-norms guard in `diffsci2.extra.chunk_*`) can
        sniff which kind of norm a model uses without caring about its
        class. VAENetMP always uses PixelNorm when `pixel_norm=True`."""
        return "pixel" if self.pixel_norm else "group"

    # ----- Attention invariants pinned at False ----------------------- #
    # VAENetMP enforces "no attention" to keep the receptive field
    # finite; chunk_decode/chunk_encode infrastructure reads these
    # attributes to compute stage-wise RF, so we expose them as
    # properties matching the `VAENetConfig` contract.
    @property
    def has_mid_attn(self) -> bool:
        return False

    @property
    def attn_resolutions(self) -> tuple:
        return ()

    def assert_consistent(self) -> None:
        assert self.dimension in (1, 2, 3)
        if self.forced_wn:
            assert self.mp_conv, "forced_wn requires mp_conv=True"
        if self.mp_conv:
            raise NotImplementedError(
                "VAENetMP with mp_conv=True (approach_c) requires a "
                "dim-generic MPConv primitive that is not yet ported "
                "into diffsci2.nets.normedlayers. Use approach_a or "
                "approach_b."
            )

    @classmethod
    def approach_a(cls, **overrides) -> "VAENetMPConfig":
        """PixelNorm only — minimal change from vanilla."""
        return cls(mp_silu=False, mp_sum=False, mp_conv=False,
                   forced_wn=False, **overrides)

    @classmethod
    def approach_b(cls, **overrides) -> "VAENetMPConfig":
        """+ MP residuals and MP-SiLU. Magnitude control on the main path."""
        return cls(mp_silu=True, mp_sum=True, mp_conv=False,
                   forced_wn=False, **overrides)

    @classmethod
    def approach_c(cls, **overrides) -> "VAENetMPConfig":
        """+ MP-Conv with forced WN. Full EDM2 CONFIG-G analog.

        Not currently supported (requires the dim-generic MPConv
        primitive). `assert_consistent` will raise NotImplementedError.
        """
        return cls(mp_silu=True, mp_sum=True, mp_conv=True,
                   forced_wn=True, **overrides)


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #

def _make_norm(config: VAENetMPConfig, _channels: int) -> nn.Module:
    """Always returns PixelNorm at present. Channels arg kept for parity."""
    return PixelNorm() if config.pixel_norm else nn.Identity()


def _make_act(config: VAENetMPConfig) -> Callable[[torch.Tensor], torch.Tensor]:
    return mp_silu if config.mp_silu else F.silu


def _make_hidden_conv(config: VAENetMPConfig, in_ch: int, out_ch: int,
                      kernel: int = 3, stride: int = 1,
                      bias: bool | None = None) -> nn.Module:
    """3x3 (or 1x1) conv in the hidden path. Bare `nn.Conv?d` only;
    `config.mp_conv=True` is rejected by `assert_consistent`."""
    if bias is None:
        bias = config.conv_bias
    Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[config.dimension]
    if stride == 1:
        return Conv(in_ch, out_ch, kernel, stride=1, padding=kernel // 2, bias=bias)
    # stride==2: caller handles asymmetric pad (see Downsample).
    return Conv(in_ch, out_ch, kernel, stride=stride, padding=0, bias=bias)


def _make_io_conv(config: VAENetMPConfig, in_ch: int, out_ch: int,
                  bias: bool, kernel: int = 3) -> nn.Module:
    """Input / output 3x3 conv. Always biased per config; bare."""
    return _make_hidden_conv(config, in_ch, out_ch, kernel=kernel, stride=1,
                             bias=bias)


def _make_one_one_conv(config: VAENetMPConfig, in_ch: int, out_ch: int,
                       bias: bool = True) -> nn.Module:
    """1x1 conv. Used for quant_conv / post_quant_conv / shortcut projection.

    Kept as bare `nn.Conv?d` — these carry information-bottleneck
    semantics (quant_conv) or are pure linear adapters (shortcut). We do
    not MP-normalise them.
    """
    Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[config.dimension]
    return Conv(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class MPResnetBlock(nn.Module):
    """Pre-activation residual block, EDM2-style placement.

    Structure:

        x
        |- PixelNorm -> act -> Conv3x3
        |       |- PixelNorm -> act -> Conv3x3
        |                              |
        |- (1x1 shortcut if in!=out) --|
                                    sum/mp_sum
    """

    def __init__(self, config: VAENetMPConfig,
                 in_channels: int, out_channels: int):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = _make_norm(config, in_channels)
        self.conv1 = _make_hidden_conv(config, in_channels, out_channels,
                                       kernel=3, stride=1)
        self.norm2: nn.Module = _make_norm(config, out_channels)
        self.conv2 = _make_hidden_conv(config, out_channels, out_channels,
                                       kernel=3, stride=1)
        self.act = _make_act(config)

        if in_channels != out_channels:
            self.shortcut = _make_one_one_conv(config, in_channels, out_channels,
                                               bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        x_proj = self.shortcut(x)
        if self.config.mp_sum:
            return mp_sum(x_proj, h, t=self.config.mp_sum_t)
        return x_proj + h


class Downsample(nn.Module):
    """Strided 3x3 conv. Mirrors `vaenet.Downsample(with_conv=True)`."""

    def __init__(self, config: VAENetMPConfig, channels: int):
        super().__init__()
        self.config = config
        self.with_conv = config.resamp_with_conv
        if self.with_conv:
            self.conv = _make_hidden_conv(config, channels, channels,
                                          kernel=3, stride=2, bias=config.conv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.with_conv:
            return {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[
                self.config.dimension](x, kernel_size=2, stride=2)
        pad = (0, 1) * self.config.dimension
        x = F.pad(x, pad, mode='constant', value=0.0)
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest/area-interp upsample + (optional) 3x3 conv. Matches vaenet."""

    def __init__(self, config: VAENetMPConfig, channels: int):
        super().__init__()
        self.config = config
        self.with_conv = config.resamp_with_conv
        if self.with_conv:
            self.conv = _make_hidden_conv(config, channels, channels,
                                          kernel=3, stride=1, bias=config.conv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='area')
        if self.with_conv:
            x = self.conv(x)
        return x


# --------------------------------------------------------------------------- #
# Encoder / Decoder
# --------------------------------------------------------------------------- #

class VAEEncoderMP(nn.Module):
    """Matches `vaenet.VAEEncoder` skeleton; MP layers throughout."""

    def __init__(self, config: VAENetMPConfig):
        super().__init__()
        self.config = config

        self.conv_in = _make_io_conv(config, config.in_channels, config.ch,
                                     bias=config.input_bias)

        block_in = config.ch
        self.down = nn.ModuleList()
        for i_level in range(config.num_resolutions):
            block = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(config.num_res_blocks):
                block.append(MPResnetBlock(config, block_in, block_out))
                block_in = block_out
            level = nn.Module()
            level.block = block
            if i_level != config.num_resolutions - 1:
                level.downsample = Downsample(config, block_in)
            self.down.append(level)

        self.mid_block_1 = MPResnetBlock(config, block_in, block_in)
        self.mid_block_2 = MPResnetBlock(config, block_in, block_in)

        z_channels = config.z_channels * (2 if config.double_z else 1)
        self.norm_out = _make_norm(config, block_in)
        self.act = _make_act(config)
        self.conv_out = _make_io_conv(config, block_in, z_channels,
                                      bias=True)
        self.quant_conv = _make_one_one_conv(
            config, z_channels, 2 * config.z_dim, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level, level in enumerate(self.down):
            for block in level.block:
                h = block(h)
            if i_level != self.config.num_resolutions - 1:
                h = level.downsample(h)
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h

    def calculate_receptive_field(self) -> dict:
        """Same return shape as `VAEEncoder.calculate_receptive_field`
        (input-space RF, latent-space RF, downsampling factor, trace)."""
        config = self.config
        rf_per_block = 4  # two 3x3 convs per MPResnetBlock
        rf = 1
        trace = []
        rf += 2  # conv_in (3x3)
        trace.append(f"conv_in: RF = {rf}")
        current_stride = 1
        for i_level in range(config.num_resolutions):
            num_blocks = config.num_res_blocks
            rf += num_blocks * rf_per_block
            trace.append(f"down[{i_level}] ({num_blocks} blocks): RF = {rf}")
            if i_level != config.num_resolutions - 1:
                if config.resamp_with_conv:
                    rf += 2
                else:
                    rf += 1
                current_stride *= 2
                trace.append(f"down[{i_level}].downsample: RF = {rf}")
        rf += 2 * rf_per_block  # mid blocks
        trace.append(f"mid blocks: RF = {rf}")
        rf += 2  # conv_out (3x3)
        trace.append(f"conv_out: RF = {rf}")
        # quant_conv: 1x1, no RF change
        rf_at_latent = rf // current_stride
        return {
            'rf_input': rf,
            'rf_latent': rf_at_latent,
            'downsampling_factor': current_stride,
            'has_attention': False,
            'feasible_chunking': True,
            'trace': trace,
            'rf_per_block': rf_per_block,
            'mode': 'standard',
        }


class VAEDecoderMP(nn.Module):
    """Matches `vaenet.VAEDecoder` skeleton; MP layers throughout."""

    def __init__(self, config: VAENetMPConfig):
        super().__init__()
        self.config = config

        self.post_quant_conv = _make_one_one_conv(
            config, config.z_dim, config.z_channels, bias=True
        )

        block_in = config.ch * config.ch_mult[-1]
        self.conv_in = _make_io_conv(config, config.z_channels, block_in,
                                     bias=config.input_bias)

        self.mid_block_1 = MPResnetBlock(config, block_in, block_in)
        self.mid_block_2 = MPResnetBlock(config, block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(config.num_resolutions)):
            block = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(config.num_res_blocks + 1):
                block.append(MPResnetBlock(config, block_in, block_out))
                block_in = block_out
            level = nn.Module()
            level.block = block
            if i_level != 0:
                level.upsample = Upsample(config, block_in)
            self.up.insert(0, level)

        self.norm_out = _make_norm(config, block_in)
        self.act = _make_act(config)
        self.conv_out = _make_io_conv(config, block_in, config.out_channels,
                                      bias=config.output_bias)
        self.gain_out = Gain(init=config.output_gain_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        h = self.conv_in(z)
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        for i_level in reversed(range(len(self.up))):
            level = self.up[i_level]
            for block in level.block:
                h = block(h)
            if i_level != 0:
                h = level.upsample(h)
        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)
        h = self.gain_out(h)
        if self.config.tanh_out:
            h = torch.tanh(h)
        return h

    def calculate_receptive_field(self) -> dict:
        """Same return shape as `VAEDecoder.calculate_receptive_field`
        (latent-space RF, after-middle RF, output-space RF, trace)."""
        config = self.config
        rf_per_block = 4  # two 3x3 convs per MPResnetBlock
        rf = 1
        trace = []
        # post_quant_conv: 1x1, no RF change
        trace.append(f"post_quant_conv (1x1): RF = {rf}")
        rf += 2  # conv_in (3x3)
        trace.append(f"conv_in (3x3): RF = {rf}")
        rf += rf_per_block  # mid.block_1
        trace.append(f"mid.block_1: RF = {rf}")
        rf += rf_per_block  # mid.block_2
        trace.append(f"mid.block_2: RF = {rf}")
        rf_after_middle = rf
        num_levels = config.num_resolutions
        for i_level in reversed(range(num_levels)):
            num_blocks = config.num_res_blocks + 1
            rf += num_blocks * rf_per_block
            trace.append(f"up[{i_level}] ({num_blocks} blocks): RF = {rf}")
            if i_level != 0:
                trace.append(f"up[{i_level}].upsample (no RF change in latent coords)")
        rf += 2  # conv_out (3x3)
        trace.append(f"conv_out (3x3): RF = {rf}")
        # Gain is 1x1, no RF change
        recommended_overlap = int(rf * 1.5)
        if recommended_overlap <= 16:
            recommended_overlap = 16
        elif recommended_overlap <= 24:
            recommended_overlap = 24
        elif recommended_overlap <= 32:
            recommended_overlap = 32
        else:
            recommended_overlap = ((recommended_overlap + 15) // 16) * 16
        spatial_factor = 2 ** (num_levels - 1)
        return {
            'rf_latent': rf,
            'rf_after_middle': rf_after_middle,
            'rf_output': rf * spatial_factor,
            'min_overlap': rf,
            'recommended_overlap': recommended_overlap,
            'spatial_upsampling_factor': spatial_factor,
            'has_attention': False,
            'feasible_chunking': True,
            'trace': trace,
            'rf_per_block': rf_per_block,
            'mode': 'standard',
            'num_convolutions': len([t for t in trace if 'RF' in t and 'no RF' not in t]),
        }


# --------------------------------------------------------------------------- #
# Top-level wrapper
# --------------------------------------------------------------------------- #

class VAENetMP(nn.Module):
    """Drop-in replacement for `VAENet` (same `encoder`/`decoder` attrs).

    Used with `diffsci2.models.VAEModule` exactly like `VAENet`:
        VAEModule(config=VAEModuleConfig(), encdec=VAENetMP(config))
    """

    def __init__(self, config: VAENetMPConfig):
        super().__init__()
        config.assert_consistent()
        self.config = config
        self.encoder = VAEEncoderMP(config)
        self.decoder = VAEDecoderMP(config)

    def calculate_receptive_field(self) -> dict:
        """Mirror of `VAENet.calculate_receptive_field` for the MP variant.

        Same convolution stack (two 3x3 per ResBlock, 3x3 io convs,
        3x3 strided downsample). PixelNorm, mp_silu, mp_sum and Gain
        all have spatial RF = 1, so the only contributors to RF are
        the 3x3 convs.
        """
        cfg = self.config
        rf_per_block = 4  # two 3x3 convs per ResBlock

        # ---- encoder ---------------------------------------------------
        rf = 1
        trace = []
        rf += 2  # conv_in (3x3)
        trace.append(f"conv_in: rf={rf}")
        current_stride = 1
        for i_level in range(cfg.num_resolutions):
            rf += cfg.num_res_blocks * rf_per_block
            trace.append(f"down[{i_level}] ({cfg.num_res_blocks} blocks): rf={rf}")
            if i_level != cfg.num_resolutions - 1:
                if cfg.resamp_with_conv:
                    rf += 2  # 3x3 strided
                else:
                    rf += 1  # 2x2 avg pool
                current_stride *= 2
                trace.append(f"down[{i_level}].downsample: rf={rf}")
        rf += 2 * rf_per_block  # mid_block_1, mid_block_2
        trace.append(f"mid blocks: rf={rf}")
        rf += 2  # conv_out
        trace.append(f"conv_out: rf={rf}")
        enc_rf_input = rf
        enc_rf_latent = rf // current_stride

        # ---- decoder ---------------------------------------------------
        rf = 1
        rf += 2  # conv_in
        rf += 2 * rf_per_block  # mid
        for i_level in reversed(range(cfg.num_resolutions)):
            rf += (cfg.num_res_blocks + 1) * rf_per_block
        rf += 2  # conv_out
        dec_rf_latent = rf
        spatial_factor = 2 ** (cfg.num_resolutions - 1)
        dec_rf_output = dec_rf_latent * spatial_factor

        return {
            'encoder': {
                'rf_input': enc_rf_input,
                'rf_latent': enc_rf_latent,
                'downsampling_factor': current_stride,
                'trace': trace,
            },
            'decoder': {
                'rf_latent': dec_rf_latent,
                'rf_output': dec_rf_output,
                'spatial_upsampling_factor': spatial_factor,
            },
            'config': {
                'pixel_norm': cfg.pixel_norm,
                'mp_silu': cfg.mp_silu,
                'mp_sum': cfg.mp_sum,
                'mp_conv': cfg.mp_conv,
                'forced_wn': cfg.forced_wn,
                'output_gain_init': cfg.output_gain_init,
                'rf_per_block': rf_per_block,
            },
        }

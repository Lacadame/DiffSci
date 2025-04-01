from typing import List

import functools
import pathlib
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F


from .patched_conv import get_patch_conv


class VAENetConfig:
    """Configuration class for dimensionally-flexible VAE architecture."""

    def __init__(
        self,
        dimension: int = 3,                # Spatial dimensions (1, 2, or 3)
        in_channels: int = 1,
        out_channels: int = 1,
        z_channels: int = 4,               # Latent space channels before mean/logvar split
        z_dim: int = 4,                    # Final latent dimension after mean/logvar
        ch: int = 32,                      # Base channel count
        ch_mult: List[int] = [1, 2, 4],    # Channel multipliers for each level
        num_res_blocks: int = 2,           # Number of residual blocks per level
        attn_resolutions: List[int] = [],  # Resolutions where attention is applied
        dropout: float = 0.0,
        resolution: int = 64,              # Input resolution
        has_mid_attn: bool = True,         # Whether to use attention in the middle blocks
        resamp_with_conv: bool = True,     # Use conv for up/downsampling
        attn_type: str = "vanilla",        # Attention type: vanilla, linear, none
        tanh_out: bool = False,            # Apply tanh to output
        input_bias: bool = True,           # Use bias in input conv
        output_bias: bool = True,          # Use bias in output conv
        with_time_emb: bool = False,       # Use time embeddings
        double_z: bool = True,             # Double the output in encoder for mean and logvar
        num_groups: int = 32,              # Number of groups for GroupNorm
        patch_size: int = None,            # Patch size for patch-based convolutions
        memory_efficient_variant: bool = False  # Use memory efficient decoding
    ):
        assert dimension in [1, 2, 3], f"Dimension must be 1, 2, or 3, got {dimension}"

        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.z_dim = z_dim
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resolution = resolution
        self.has_mid_attn = has_mid_attn
        self.resamp_with_conv = resamp_with_conv
        self.attn_type = attn_type
        self.tanh_out = tanh_out
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.with_time_emb = with_time_emb
        self.double_z = double_z
        self.num_resolutions = len(self.ch_mult)
        self.num_groups = num_groups
        self.patch_size = patch_size
        self.memory_efficient_variant = memory_efficient_variant

    def export_description(self) -> dict:
        """Export configuration as a dictionary."""
        return {
            "dimension": self.dimension,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "z_channels": self.z_channels,
            "z_dim": self.z_dim,
            "ch": self.ch,
            "ch_mult": self.ch_mult,
            "num_res_blocks": self.num_res_blocks,
            "attn_resolutions": self.attn_resolutions,
            "dropout": self.dropout,
            "resolution": self.resolution,
            "has_mid_attn": self.has_mid_attn,
            "resamp_with_conv": self.resamp_with_conv,
            "attn_type": self.attn_type,
            "tanh_out": self.tanh_out,
            "input_bias": self.input_bias,
            "output_bias": self.output_bias,
            "with_time_emb": self.with_time_emb,
            "double_z": self.double_z,
            "num_groups": self.num_groups,
            "patch_size": self.patch_size,
            "memory_efficient_variant": self.memory_efficient_variant
        }
    
    @classmethod
    def from_description(cls, description: dict):
        return cls(**description)
    
    @classmethod
    def from_config_file(cls, config_file: pathlib.Path | str):
        with open(config_file, "r") as f:
            description = yaml.safe_load(f)
        return cls.from_description(description)


class DimensionHelper:
    """Helper class for dimension-specific operations."""

    @staticmethod
    def get_conv_cls(dimension):
        if dimension == 1:
            return nn.Conv1d
        elif dimension == 2:
            return nn.Conv2d
        elif dimension == 3:
            return nn.Conv3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    @staticmethod
    def get_patch_conv_cls(dimension):
        return PatchedConv.initialize_with_dimension(dimension)

    @staticmethod
    def get_convtranspose_cls(dimension):
        if dimension == 1:
            return nn.ConvTranspose1d
        elif dimension == 2:
            return nn.ConvTranspose2d
        elif dimension == 3:
            return nn.ConvTranspose3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    @staticmethod
    def get_shape_for_broadcast(dimension, batch_size, channels, *spatial_dims):
        """Return shape for broadcasting time embedding to spatial dimensions."""
        if dimension == 1:
            return [batch_size, channels, 1]
        elif dimension == 2:
            return [batch_size, channels, 1, 1]
        elif dimension == 3:
            return [batch_size, channels, 1, 1, 1]
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    @staticmethod
    def flatten_spatial_dims(x, dimension):
        """Flatten spatial dimensions for attention."""
        b, c = x.shape[0], x.shape[1]
        if dimension == 1:
            # [b, c, d] -> [b, c, d]
            return x.reshape(b, c, -1), x.shape[2]
        elif dimension == 2:
            # [b, c, h, w] -> [b, c, h*w]
            return x.reshape(b, c, -1), (x.shape[2], x.shape[3])
        elif dimension == 3:
            # [b, c, h, w, d] -> [b, c, h*w*d]
            return x.reshape(b, c, -1), (x.shape[2], x.shape[3], x.shape[4])
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    @staticmethod
    def unflatten_spatial_dims(x, dimension, spatial_size):
        """Unflatten spatial dimensions after attention."""
        b, c = x.shape[0], x.shape[1]
        if dimension == 1:
            # [b, c, d] -> [b, c, d]
            return x.reshape(b, c, spatial_size)
        elif dimension == 2:
            # [b, c, h*w] -> [b, c, h, w]
            h, w = spatial_size
            return x.reshape(b, c, h, w)
        elif dimension == 3:
            # [b, c, h*w*d] -> [b, c, h, w, d]
            h, w, d = spatial_size
            return x.reshape(b, c, h, w, d)
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")


class PatchedConv(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 patch_size: int | None = None,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int | None = None,
                 bias: bool = True,
                 dimension: int = 3):
        # TODO: Generalize for non-square patches
        super().__init__()
        assert kernel_size % 2 == 1, f"Kernel size must be odd, got {kernel_size}"
        assert stride == 1, f"Only implemented for stride == 1, got {stride}"
        assert padding is None or padding == kernel_size//2, \
            f"Padding must be kernel_size//2, got {padding}"
        self.padding = padding if padding is not None else kernel_size//2
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.conv = DimensionHelper.get_conv_cls(dimension)(in_channels,
                                                            out_channels,
                                                            kernel_size,
                                                            padding=0,
                                                            bias=bias)
        self.patch_conv = get_patch_conv(dimension)
        self.patch_size = patch_size

    @classmethod
    def initialize_with_dimension(cls, dimension: int):
        return functools.partial(cls, dimension=dimension)

    def forward(self,
                x: torch.Tensor,
                custom_patch_size: int = None) -> torch.Tensor:
        # x is assumed to be of shape [..., *spatial_shape]
        dimensions = x.shape[-self.dimension:]
        patch_size = custom_patch_size if custom_patch_size is not None else self.patch_size
        if patch_size is None:
            need_patch = False
        else:
            need_patch = any(d > patch_size for d in dimensions)
        if need_patch:
            return self.patch_conv(x,
                                   self.patch_size,
                                   conv_cls=self.conv,
                                   padding=self.padding)
        else:
            # We manually pad the input to match the output shape
            pd = [self.padding]*2*self.dimension
            x = torch.nn.functional.pad(x, pd)
            return self.conv(x)
        
    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias


# Utility functions and blocks
def get_norm(in_channels, num_groups=32):
    """Normalized layer with dimension flexibility."""
    return nn.GroupNorm(num_groups=num_groups,
                        num_channels=in_channels,
                        eps=1e-6,
                        affine=True)


def nonlinearity(x):
    # swish activation
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    """Residual block with dimension flexibility."""

    def __init__(self, *, dimension, in_channels, out_channels=None,
                 conv_shortcut=False, dropout, temb_channels=0, num_groups=32,
                 patch_size=None):
        super().__init__()
        self.dimension = dimension
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.patch_size = patch_size

        Conv = DimensionHelper.get_patch_conv_cls(dimension)

        self.norm1 = get_norm(in_channels, num_groups=num_groups)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                          patch_size=patch_size)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = get_norm(out_channels, num_groups=num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                          patch_size=patch_size)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv(in_channels, out_channels, kernel_size=3,
                                          stride=1, padding=1, patch_size=patch_size)
            else:
                self.nin_shortcut = Conv(in_channels, out_channels, kernel_size=1,
                                         stride=1, padding=0, patch_size=patch_size)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None and hasattr(self, 'temb_proj'):
            temb_h = self.temb_proj(nonlinearity(temb))
            broadcast_shape = DimensionHelper.get_shape_for_broadcast(
                self.dimension, h.shape[0], temb_h.shape[1])
            h = h + temb_h.view(*broadcast_shape)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """Self-attention block with dimension flexibility."""

    def __init__(self, dimension, in_channels, num_groups=32, patch_size=None):
        super().__init__()
        self.dimension = dimension
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.norm = get_norm(in_channels, num_groups=num_groups)

        Conv = DimensionHelper.get_patch_conv_cls(dimension)
        self.q = Conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, patch_size=patch_size)
        self.k = Conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, patch_size=patch_size)
        self.v = Conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, patch_size=patch_size)
        self.proj_out = Conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, patch_size=patch_size)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        flat_q, orig_shape = DimensionHelper.flatten_spatial_dims(q, self.dimension)
        flat_k, _ = DimensionHelper.flatten_spatial_dims(k, self.dimension)
        flat_v, _ = DimensionHelper.flatten_spatial_dims(v, self.dimension)

        b, c, hw = flat_q.shape
        flat_q = flat_q.permute(0, 2, 1)    # b, hw, c

        w_ = torch.bmm(flat_q, flat_k)      # b, hw, hw
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)   # b, hw, hw (first hw of k, second of q)
        h_ = torch.bmm(flat_v, w_)  # b, c, hw (hw of q)

        h_ = DimensionHelper.unflatten_spatial_dims(h_, self.dimension, orig_shape)
        h_ = self.proj_out(h_)

        return x + h_


class LinearAttention(nn.Module):
    """Linear attention mechanism with dimension flexibility."""

    def __init__(self, dimension, dim, heads=4, dim_head=32, patch_size=None):
        super().__init__()
        self.dimension = dimension
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        self.patch_size = patch_size

        Conv = DimensionHelper.get_patch_conv_cls(dimension)
        self.to_qkv = Conv(dim, hidden_dim * 3, 1, bias=False, patch_size=patch_size)
        self.to_out = Conv(hidden_dim, dim, 1, patch_size=patch_size)

    def forward(self, x):
        b, _ = x.shape[0], x.shape[1]
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Flatten spatial dimensions
        flat_q, orig_shape = DimensionHelper.flatten_spatial_dims(q, self.dimension)
        flat_k, _ = DimensionHelper.flatten_spatial_dims(k, self.dimension)
        flat_v, _ = DimensionHelper.flatten_spatial_dims(v, self.dimension)

        # Reshape for multi-head attention
        q = flat_q.reshape(b, self.heads, self.dim_head, -1)
        k = flat_k.reshape(b, self.heads, self.dim_head, -1)
        v = flat_v.reshape(b, self.heads, self.dim_head, -1)

        # Linear attention
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)

        # Reshape and project out
        out = out.reshape(b, self.heads * self.dim_head, -1)
        out = DimensionHelper.unflatten_spatial_dims(out, self.dimension, orig_shape)

        return self.to_out(out)


class LinAttnBlock(nn.Module):
    """Linear attention block wrapper for compatibility."""

    def __init__(self, dimension, in_channels, patch_size=None):
        super().__init__()
        self.patch_size = patch_size
        self.attn = LinearAttention(dimension, dim=in_channels, heads=1, dim_head=in_channels,
                                    patch_size=patch_size)

    def forward(self, x):
        return self.attn(x)


def make_attn(dimension, in_channels, attn_type="vanilla", num_groups=32, patch_size=None):
    """Factory function to create an attention block of specified type."""
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'

    if attn_type == "vanilla":
        return AttnBlock(dimension, in_channels, num_groups=num_groups, patch_size=patch_size)
    elif attn_type == "none":
        return nn.Identity()
    else:
        return LinAttnBlock(dimension, in_channels, patch_size=patch_size)


class Upsample(nn.Module):
    """Upsampling module with dimension flexibility."""

    def __init__(self, dimension, in_channels, with_conv, patch_size=None):
        super().__init__()
        self.dimension = dimension
        self.with_conv = with_conv
        self.patch_size = patch_size

        if self.with_conv:
            Conv = DimensionHelper.get_patch_conv_cls(dimension)
            self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, patch_size=patch_size)

    def forward(self, x):
        if self.dimension == 1:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        elif self.dimension == 2:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        elif self.dimension == 3:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.with_conv:
            x = self.conv(x)

        return x


class Downsample(nn.Module):
    """Downsampling module with dimension flexibility."""

    def __init__(self, dimension, in_channels, with_conv, patch_size=None):
        super().__init__()
        self.dimension = dimension
        self.with_conv = with_conv
        self.patch_size = patch_size

        if self.with_conv:
            Conv = DimensionHelper.get_conv_cls(dimension)
            stride = 2
            padding = 0
            self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x):
        if self.with_conv:
            if self.dimension == 1:
                pad = (0, 1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            elif self.dimension == 2:
                pad = (0, 1, 0, 1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            else:  # dimension == 3
                pad = (0, 1, 0, 1, 0, 1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            if self.dimension == 1:
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
            elif self.dimension == 2:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
            else:  # dimension == 3
                x = F.avg_pool3d(x, kernel_size=2, stride=2)

        return x


class VAEEncoder(nn.Module):
    """Multi-dimensional VAE encoder."""

    def __init__(self, config: VAENetConfig):
        super().__init__()
        self.config = config
        self.dimension = config.dimension
        self.patch_size = config.patch_size

        # Choose correct convolution class for the given dimension
        Conv = DimensionHelper.get_patch_conv_cls(config.dimension)

        # Define the embedding dimension for time if needed
        self.temb_ch = 0
        if config.with_time_emb:
            self.temb_ch = config.ch * 4
            self.time_embed = nn.Sequential(
                nn.Linear(config.ch, self.temb_ch),
                nn.SiLU(),
                nn.Linear(self.temb_ch, self.temb_ch),
            )

        # Input projection
        self.conv_in = Conv(
            config.in_channels,
            config.ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=config.input_bias,
            patch_size=config.patch_size
        )

        # Calculate the minimum resolution (bottleneck size)
        curr_res = config.resolution

        # Prepare channel dimensions for each resolution
        block_in = config.ch

        # Downsampling blocks
        self.down = nn.ModuleList()
        for i_level in range(config.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(config.num_res_blocks):
                block.append(ResnetBlock(
                    dimension=config.dimension,
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=config.dropout,
                    num_groups=config.num_groups,
                    patch_size=config.patch_size
                ))
                block_in = block_out

                if curr_res in config.attn_resolutions:
                    attn.append(make_attn(
                        config.dimension,
                        block_in,
                        attn_type=config.attn_type,
                        num_groups=config.num_groups,
                        patch_size=config.patch_size
                    ))

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != config.num_resolutions - 1:
                down.downsample = Downsample(config.dimension, block_in, config.resamp_with_conv,
                                             patch_size=config.patch_size)
                curr_res = curr_res // 2

            self.down.append(down)

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            dimension=config.dimension,
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            num_groups=config.num_groups,
            patch_size=config.patch_size
        )

        if config.has_mid_attn:
            self.mid.attn_1 = make_attn(
                config.dimension,
                block_in,
                attn_type=config.attn_type,
                num_groups=config.num_groups,
                patch_size=config.patch_size
            )

        self.mid.block_2 = ResnetBlock(
            dimension=config.dimension,
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            num_groups=config.num_groups,
            patch_size=config.patch_size
        )

        # Output projection
        z_channels = config.z_channels
        if config.double_z:
            z_channels = 2 * z_channels

        self.norm_out = get_norm(block_in, num_groups=config.num_groups)
        self.conv_out = Conv(
            block_in,
            z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            patch_size=config.patch_size
        )

        # Additional projection to z_dim if needed
        self.quant_conv = Conv(z_channels, 2 * config.z_dim, kernel_size=1, patch_size=config.patch_size)

    def forward(self, x, time=None):
        # print(f"Input x shape: {x.shape}, 10th item: {x.flatten()[9] if x.numel() > 9 else None}")
        
        # Time embedding
        temb = None
        if self.config.with_time_emb and time is not None:
            temb = self.time_embed(time)
            # print(f"Time embedding shape: {temb.shape}, 10th item: {temb.flatten()[9] if temb.numel() > 9 else None}")

        # Initial convolution
        h = self.conv_in(x)
        # print(f"After conv_in shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")

        # Downsampling
        hs = [h]
        for i_level in range(self.config.num_resolutions):
            # print(f"Downsampling level: {i_level}")
            for i_block in range(self.config.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(f"  After down[{i_level}].block[{i_block}] shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
                
                if len(self.down[i_level].attn) > i_block:
                    h = self.down[i_level].attn[i_block](h)
                    # print(f"  After down[{i_level}].attn[{i_block}] shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
                hs.append(h)

            if i_level != self.config.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                # print(f"  After down[{i_level}].downsample shape: {hs[-1].shape}, 10th item: {hs[-1].flatten()[9] if hs[-1].numel() > 9 else None}")

        # Middle
        h = hs[-1]
        # print(f"Middle input shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        h = self.mid.block_1(h, temb)
        # print(f"After mid.block_1 shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        if hasattr(self.mid, 'attn_1'):
            h = self.mid.attn_1(h)
            # print(f"After mid.attn_1 shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        h = self.mid.block_2(h, temb)
        # print(f"After mid.block_2 shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")

        # Normalize and project
        h = self.norm_out(h)
        # print(f"After norm_out shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        h = nonlinearity(h)
        # print(f"After nonlinearity shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        h = self.conv_out(h)
        # print(f"After conv_out shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")

        # Final projection to z_dim (for mean and variance)
        h = self.quant_conv(h)
        # print(f"After quant_conv shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")

        return h


class VAEDecoder(nn.Module):
    """Multi-dimensional VAE decoder."""

    def __init__(self, config: VAENetConfig):
        super().__init__()
        self.config = config
        self.dimension = config.dimension
        self.patch_size = config.patch_size

        # Choose correct convolution class for the given dimension
        Conv = DimensionHelper.get_patch_conv_cls(config.dimension)

        # Define the embedding dimension for time if needed
        self.temb_ch = 0
        if config.with_time_emb:
            self.temb_ch = config.ch * 4
            self.time_embed = nn.Sequential(
                nn.Linear(config.ch, self.temb_ch),
                nn.SiLU(),
                nn.Linear(self.temb_ch, self.temb_ch),
            )

        # Initial projection from z_dim to z_channels
        self.post_quant_conv = Conv(config.z_dim, config.z_channels, kernel_size=1, patch_size=config.patch_size)

        # Calculate the minimum resolution (bottleneck size)
        self.num_resolutions = len(config.ch_mult)
        self.min_res = config.resolution // (2 ** (self.num_resolutions - 1))
        curr_res = self.min_res

        # Input projection
        block_in = config.ch * config.ch_mult[-1]
        self.conv_in = Conv(
            config.z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=config.input_bias,
            patch_size=config.patch_size
        )

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            dimension=config.dimension,
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            num_groups=config.num_groups,
            patch_size=config.patch_size
        )

        if config.has_mid_attn:
            self.mid.attn_1 = make_attn(
                config.dimension,
                block_in,
                attn_type=config.attn_type,
                num_groups=config.num_groups,
                patch_size=config.patch_size
            )

        self.mid.block_2 = ResnetBlock(
            dimension=config.dimension,
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            num_groups=config.num_groups,
            patch_size=config.patch_size
        )

        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            if self.config.memory_efficient_variant:
                if i_level == 0:
                    block_out = config.ch * config.ch_mult[i_level]
                else:
                    block_out = config.ch * config.ch_mult[i_level-1]
            else:
                block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(config.num_res_blocks + 1):
                block.append(ResnetBlock(
                    dimension=config.dimension,
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=config.dropout,
                    num_groups=config.num_groups,
                    patch_size=config.patch_size
                ))
                block_in = block_out

                if curr_res in config.attn_resolutions:
                    attn.append(make_attn(
                        config.dimension,
                        block_in,
                        attn_type=config.attn_type,
                        num_groups=config.num_groups,
                        patch_size=config.patch_size
                    ))

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(config.dimension, block_in, config.resamp_with_conv,
                                       patch_size=config.patch_size)
                curr_res = curr_res * 2

            self.up.insert(0, up)  # Insert at the beginning for correct order

        # Output normalization and convolution
        self.norm_out = get_norm(block_in, num_groups=config.num_groups)
        self.conv_out = Conv(
            block_in,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=config.output_bias,
            patch_size=config.patch_size
        )

    def forward(self, z, time=None):
        # print(f"Input z shape: {z.shape}, 10th item: {z.flatten()[9] if z.numel() > 9 else None}")
        
        # Time embedding
        temb = None
        if self.config.with_time_emb and time is not None:
            temb = self.time_embed(time)
            # print(f"Time embedding shape: {temb.shape}, 10th item: {temb.flatten()[9] if temb.numel() > 9 else None}")

        # Project from z_dim to z_channels
        z = self.post_quant_conv(z)
        # print(f"After post_quant_conv shape: {z.shape}, 10th item: {z.flatten()[9] if z.numel() > 9 else None}")

        # Initial convolution
        h = self.conv_in(z)
        # print(f"After conv_in shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")

        # Middle block
        h = self.mid.block_1(h, temb)
        # print(f"After mid.block_1 shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        if hasattr(self.mid, 'attn_1'):
            h = self.mid.attn_1(h)
            # print(f"After mid.attn_1 shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")

        h = self.mid.block_2(h, temb)
        # print(f"After mid.block_2 shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")

        # Upsampling
        for i_level in reversed(range(len(self.up))):
            # print(f"Upsampling level: {i_level}")
            for i_block in range(len(self.up[i_level].block)):
                h = self.up[i_level].block[i_block](h, temb)
                # print(f"  After up[{i_level}].block[{i_block}] shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
                
                if len(self.up[i_level].attn) > i_block:
                    h = self.up[i_level].attn[i_block](h)
                    # print(f"  After up[{i_level}].attn[{i_block}] shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
            
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                # print(f"  After up[{i_level}].upsample shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        # Output normalization and convolution
        h = self.norm_out(h)
        # print(f"After norm_out shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        h = nonlinearity(h)
        # print(f"After nonlinearity shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        h = self.conv_out(h)
        # print(f"After conv_out shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")

        if self.config.tanh_out:
            h = torch.tanh(h)
            # print(f"After tanh shape: {h.shape}, 10th item: {h.flatten()[9] if h.numel() > 9 else None}")
        
        return h


class VAENet(nn.Module):
    """Dimensionally-flexible VAE architecture."""

    def __init__(self, config: VAENetConfig):
        super().__init__()
        self.config = config
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)

    def encode(self, x, time=None, sample=True):
        """Encode input to latent parameters (mean and logvar)."""
        z = self.encoder(x, time)
        if sample:
            mean, logvar = torch.chunk(z, 2, dim=1)
            normsample = torch.randn_like(mean)
            std = torch.exp(0.5 * logvar)
            z = mean + std * normsample
        return z

    def decode(self, z, time=None):
        """Decode latent representation to reconstructed input."""
        return self.decoder(z, time)

    def forward(self, x, time=None):
        """Full forward pass: encode, sample, decode."""
        moments = self.encode(x, time)
        # This would typically be used with DiagonalGaussianDistribution from vaemodule.py
        return moments, self.decode(moments[:, :self.config.z_dim], time)

    def export_description(self) -> dict:
        """Export model configuration as a dictionary."""
        return {
            "config": self.config.export_description(),
        }

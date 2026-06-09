# TODO: Put DimensionHelper in a unified separate file
from typing import Any, Optional

import warnings
import pathlib
import yaml
import math

import torch
import einops

from .commonlayers import GaussianFourierProjection


class ConVitConfig(object):
    def __init__(self,
                 in_channels: int = 1,
                 embed_dim: int = 64,
                 num_pos_dims: int = 2,
                 out_channels: Optional[int] = None,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ffn_expansion_factor: int = 4,
                 attn_compression_factor: int = 2,
                 rope_freq: float = 1.0,
                 with_conv_on_upsample: bool = False,
                 with_conv_on_downsample: bool = False,
                 kernel_size_conv: int = 1,
                 kernel_size_in_out: int = 1,
                 kernel_size_depthwise: int = 3,
                 has_time_embedding: bool = False,
                 has_conditional_embedding: bool = False,
                 fourier_projection_scale: float = 30.0,
                 relative_positioning: bool = False,
                 linear_attention: bool = False,
                 input_batch_norm: bool = False,
                 condition_dropout: float = 0.1):
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_pos_dims = num_pos_dims
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.attn_compression_factor = attn_compression_factor
        self.rope_freq = rope_freq
        self.with_conv_on_upsample = with_conv_on_upsample
        self.with_conv_on_downsample = with_conv_on_downsample
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_in_out = kernel_size_in_out
        self.kernel_size_depthwise = kernel_size_depthwise
        self.has_time_embedding = has_time_embedding
        self.has_conditional_embedding = has_conditional_embedding
        self.fourier_projection_scale = fourier_projection_scale
        self.relative_positioning = relative_positioning
        self.linear_attention = linear_attention
        self.input_batch_norm = input_batch_norm
        self.condition_dropout = condition_dropout

    def export_description(self) -> dict[str, Any]:
        args = dict(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            num_pos_dims=self.num_pos_dims,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_expansion_factor=self.ffn_expansion_factor,
            attn_compression_factor=self.attn_compression_factor,
            rope_freq=self.rope_freq,
            with_conv_on_upsample=self.with_conv_on_upsample,
            with_conv_on_downsample=self.with_conv_on_downsample,
            kernel_size_conv=self.kernel_size_conv,
            kernel_size_depthwise=self.kernel_size_depthwise,
            kernel_size_in_out=self.kernel_size_in_out,
            has_time_embedding=self.has_time_embedding,
            has_conditional_embedding=self.has_conditional_embedding,
            fourier_projection_scale=self.fourier_projection_scale,
            relative_positioning=self.relative_positioning,
            linear_attention=self.linear_attention,
            input_batch_norm=self.input_batch_norm,
            condition_dropout=self.condition_dropout
        )
        return args

    @property
    def has_embedding(self):
        return self.has_time_embedding or self.has_conditional_embedding

    @classmethod
    def from_description(cls, description: dict):
        return cls(**description)

    @classmethod
    def from_config_file(cls, config_file: pathlib.Path | str):
        with open(config_file, "r") as f:
            description = yaml.safe_load(f)
        return cls.from_description(description)


class ConditionDropout(torch.nn.Module):
    """A dropout layer that zeroes out entire samples in a batch with probability p.

    Unlike standard dropout which zeroes individual elements, this layer zeroes
    all elements for randomly selected samples in the batch.

    Args:
        p (float): probability of zeroing out a sample. Default: 0.5
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        # Generate mask of shape (N, 1, ..., 1) with same number of dims as input
        mask_shape = [x.shape[0]] + [1] * (x.dim() - 1)
        mask = torch.bernoulli(torch.ones(mask_shape, device=x.device) * (1 - self.p))

        return x * mask


class DimensionHelper:
    """Helper class for dimension-specific operations."""

    @staticmethod
    def get_batch_norm_cls(dimension):
        if dimension == 1:
            return torch.nn.BatchNorm1d
        elif dimension == 2:
            return torch.nn.BatchNorm2d
        elif dimension == 3:
            return torch.nn.BatchNorm3d

    @staticmethod
    def get_conv_cls(dimension):
        if dimension == 1:
            return torch.nn.Conv1d
        elif dimension == 2:
            return torch.nn.Conv2d
        elif dimension == 3:
            return torch.nn.Conv3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    @staticmethod
    def get_convtranspose_cls(dimension):
        if dimension == 1:
            return torch.nn.ConvTranspose1d
        elif dimension == 2:
            return torch.nn.ConvTranspose2d
        elif dimension == 3:
            return torch.nn.ConvTranspose3d
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

    @staticmethod
    def interpolate_fn(x, dimension, scale_factor):
        if dimension == 1:
            return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='linear', align_corners=False)
        elif dimension == 2:
            return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        elif dimension == 3:
            return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='trilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    @staticmethod
    def avgpool_fn(x, dimension, kernel_size, stride, padding):
        if dimension == 1:
            return torch.nn.functional.avg_pool1d(x, kernel_size, stride, padding)
        elif dimension == 2:
            return torch.nn.functional.avg_pool2d(x, kernel_size, stride, padding)
        elif dimension == 3:
            return torch.nn.functional.avg_pool3d(x, kernel_size, stride, padding)


class ChannelRMSNorm(torch.nn.Module):
    def __init__(self, channel_dim: int, element_wise_affine: bool = True):
        super().__init__()
        self.element_wise_affine = element_wise_affine
        self.channel_dim = channel_dim
        if self.element_wise_affine:
            self.weight = torch.nn.Parameter(torch.ones(channel_dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps
        norm = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
        x = x / norm
        if self.element_wise_affine:
            num_pos_dims = len(x.shape[2:])
            x = x * self.weight.view(1, -1, *([1] * num_pos_dims))
        return x


class Upsample(torch.nn.Module):
    def __init__(self, num_pos_dims, channels_in, channels_out=None, expansion_factor=2, with_conv=False):
        super().__init__()
        self.num_pos_dims = num_pos_dims
        self.channels_in = channels_in
        if channels_out is None:
            channels_out = channels_in
        self.channels_out = channels_out
        self.expansion_factor = expansion_factor
        stride = expansion_factor
        assert stride % 2 == 0
        kernel_size = 2 * expansion_factor
        padding = stride // 2
        if with_conv:
            self.conv = DimensionHelper.get_convtranspose_cls(self.num_pos_dims)(
                channels_in, channels_out, kernel_size, stride, padding)
        else:
            self.conv = lambda x: DimensionHelper.interpolate_fn(
                x, self.num_pos_dims, expansion_factor)

    def forward(self, x):
        assert x.shape[1] == self.channels_in
        assert len(x.shape) == self.num_pos_dims + 2

        x = self.conv(x)
        return x


class Downsample(torch.nn.Module):
    def __init__(self, num_pos_dims,
                 channels_in,
                 channels_out=None,
                 compression_factor=2,
                 with_conv=False):
        super().__init__()
        self.num_pos_dims = num_pos_dims
        self.channels_in = channels_in
        if channels_out is None:
            channels_out = channels_in
        self.compression_factor = compression_factor
        stride = compression_factor
        assert stride % 2 == 0
        kernel_size = 2 * compression_factor
        padding = stride // 2
        if with_conv:
            self.conv = DimensionHelper.get_conv_cls(self.num_pos_dims)(
                channels_in, channels_out, kernel_size, stride, padding)
        else:
            self.conv = lambda x: DimensionHelper.avgpool_fn(
                x, self.num_pos_dims, compression_factor, compression_factor, 0)

    def forward(self, x):
        assert x.shape[1] == self.channels_in
        assert len(x.shape) == self.num_pos_dims + 2

        x = self.conv(x)
        return x


class ConvSwiGLU(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_pos_dims: int,
                 expansion_factor: int = 4,
                 kernel_size: int = 1,
                 final_rms: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_in = DimensionHelper.get_conv_cls(num_pos_dims)(
            embed_dim, embed_dim * expansion_factor, kernel_size, padding='same')
        self.linear_gate = DimensionHelper.get_conv_cls(num_pos_dims)(
            embed_dim, embed_dim * expansion_factor, kernel_size, padding='same')
        self.swish = torch.nn.SiLU()
        self.linear_out = DimensionHelper.get_conv_cls(num_pos_dims)(
            embed_dim * expansion_factor, embed_dim, kernel_size, padding='same')
        if final_rms:
            self.rms = ChannelRMSNorm(channel_dim=embed_dim)
        else:
            self.rms = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_out(self.swish(self.linear_in(x)) * self.linear_gate(x))
        x = self.rms(x)
        return x


class SwiGLU(torch.nn.Module):
    def __init__(self, embed_dim: int, final_rms: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_in = torch.nn.Linear(embed_dim, embed_dim * 4)
        self.linear_gate = torch.nn.Linear(embed_dim, embed_dim * 4)
        self.swish = torch.nn.SiLU()
        self.linear_out = torch.nn.Linear(embed_dim * 4, embed_dim)
        if final_rms:
            self.rms = torch.nn.RMSNorm(embed_dim)
        else:
            self.rms = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_out(self.swish(self.linear_in(x)) * self.linear_gate(x))
        x = self.rms(x)
        return x


class LearnedRoPE(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_pos_dims: int = 1,
                 base_freq: float = 1.0,
                 relative_positioning: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0
        self.half_dim = embed_dim // 2
        self.base_freq = base_freq
        self.num_pos_dims = num_pos_dims
        self.relative_positioning = relative_positioning

        self.angles = torch.nn.Parameter(
            torch.randn(self.num_pos_dims, self.half_dim) * self.base_freq
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position_dimensions = x.shape[1:-1]
        normalizers = (
            torch.ones(len(position_dimensions)) if not self.relative_positioning else
            torch.tensor(position_dimensions)
        ).to(x)
        positions = torch.stack(
            torch.meshgrid(
                [torch.arange(d).to(x) / n
                 for d, n in zip(position_dimensions, normalizers)]),
            dim=-1
        )
        angles = (positions.unsqueeze(-1) * self.angles).sum(dim=-2)  # [*pos_dims, d//2]
        x = einops.rearrange(x, '... (d c) -> ... d c', c=2)  # [b, l, d//2, 2]
        x = torch.stack([
            x[..., 0] * torch.cos(angles) - x[..., 1] * torch.sin(angles),
            x[..., 0] * torch.sin(angles) + x[..., 1] * torch.cos(angles),
        ], dim=-1)
        x = einops.rearrange(x, '... d c -> ... (d c)', c=2)
        return x

    @classmethod
    def outer_product(cls, *vectors):
        """
        Compute the outer product of a list of vectors of shape [..., d_i],
        returning a tensor of shape [..., d_1, d_2, ..., d_n].
        """
        raise NotImplementedError("Not implemented")
        if not vectors:
            raise ValueError("At least one vector is required")

        result = vectors[0].clone()
        for i, vec in enumerate(vectors[1:], 1):
            result = result.unsqueeze(-1) * vec.unsqueeze(-2)
        return result


class MultiheadAttention(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dim_per_head: int | None = None,
                 num_pos_dims: int = 1,
                 rope_freq: float = 1.0,
                 relative_positioning: bool = False,
                 linear_attention: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if dim_per_head is None:
            dim_per_head = embed_dim // num_heads
        self.dim_per_head = dim_per_head
        self.linear_attention = linear_attention
        assert self.embed_dim % self.num_heads == 0
        assert self.dim_per_head % 2 == 0

        self.q_proj_tensor = torch.nn.Parameter(
            torch.empty((embed_dim, dim_per_head, num_heads))
        )
        self.k_proj_tensor = torch.nn.Parameter(
            torch.empty((embed_dim, dim_per_head, num_heads))
        )
        self.v_proj_tensor = torch.nn.Parameter(
            torch.empty((embed_dim, dim_per_head, num_heads))
        )
        self.out_proj_tensor = torch.nn.Parameter(
            torch.empty((embed_dim, dim_per_head, num_heads))
        )
        self.reset_parameters()
        self.rope_layer = LearnedRoPE(
            embed_dim=dim_per_head,
            num_pos_dims=num_pos_dims,
            base_freq=rope_freq,
            relative_positioning=relative_positioning
        )
        self.register_buffer("scale", torch.sqrt(torch.tensor(dim_per_head, dtype=torch.float32)))

    def reset_parameters(self):
        qk_fan_in, qk_fan_out = self.embed_dim, self.dim_per_head
        vo_fan_in, vo_fan_out = self.embed_dim, self.dim_per_head
        qk_bound = 6 / math.sqrt(qk_fan_in + qk_fan_out)
        vo_bound = 6 / math.sqrt(vo_fan_in + vo_fan_out)

        torch.nn.init.uniform_(self.q_proj_tensor, -qk_bound, qk_bound)
        torch.nn.init.uniform_(self.k_proj_tensor, -qk_bound, qk_bound)
        torch.nn.init.uniform_(self.v_proj_tensor, -vo_bound, vo_bound)
        torch.nn.init.uniform_(self.out_proj_tensor, -vo_bound, vo_bound)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        if y is None:
            y = x

        num_channel_indexes = len(x.shape[1: -1])
        channel_m_symbols = ' '.join(
            [f'm{i}' for i in range(num_channel_indexes)]
        )
        channel_n_symbols = ' '.join(
            [f'n{i}' for i in range(num_channel_indexes)]
        )
        # Store the original spatial dimensions for later reshaping
        spatial_dims = x.shape[1:-1] if num_channel_indexes > 0 else ()

        h = self.num_heads

        query_signature = f'b {channel_m_symbols} d, d dv h -> b {channel_m_symbols} dv h'
        query = einops.einsum(x, self.q_proj_tensor, query_signature)
        key_signature = f'b {channel_m_symbols} d, d dv h -> b {channel_m_symbols} dv h'
        key = einops.einsum(y, self.k_proj_tensor, key_signature)
        value_signature = f'b {channel_m_symbols} d, d dv h -> b {channel_m_symbols} dv h'
        value = einops.einsum(y, self.v_proj_tensor, value_signature)

        if self.linear_attention:
            kv_feature_map = lambda x: torch.nn.functional.elu(x) + 1  # noqa: E731
            # kv_feature_map = lambda x: torch.nn.functional.softmax(x / self.scale, dim=-2)  # noqa: E731
            query = kv_feature_map(query) / self.scale
            key = kv_feature_map(key)
            ksum_signature = f'b {channel_m_symbols} dk h -> b dk h'
            ksum = einops.einsum(key, ksum_signature)
            value_norm_signature = f'b {channel_m_symbols} dk h, b dk h -> b {channel_m_symbols} h'
            value_norm = einops.einsum(query, ksum, value_norm_signature) + torch.finfo(value.dtype).eps

        query_signature = f'b {channel_m_symbols} dv h -> (b h) {channel_m_symbols} dv'
        query = einops.rearrange(query, query_signature)
        key_signature = f'b {channel_m_symbols} dv h -> (b h) {channel_m_symbols} dv'
        key = einops.rearrange(key, key_signature)

        query = self.rope_layer(query)
        key = self.rope_layer(key)

        query_signature = f'(b h) {channel_m_symbols} dv -> b {channel_m_symbols} dv h'
        query = einops.rearrange(query, query_signature, h=h)
        key_signature = f'(b h) {channel_m_symbols} dv -> b {channel_m_symbols} dv h'
        key = einops.rearrange(key, key_signature, h=h)

        if self.linear_attention:

            kv_signature = f'b {channel_m_symbols} dk h, b {channel_m_symbols} dv h -> b dk dv h'
            kv = einops.einsum(key, value, kv_signature)
            value_signature = f'b {channel_m_symbols} dk h, b dk dv h -> b {channel_m_symbols} dv h'
            value = einops.einsum(query, kv, value_signature)
            value = value / value_norm.unsqueeze(-2)
        else:

            dim_dict = {f'm{i}': spatial_dims[i] for i in range(num_channel_indexes)}
            query = einops.rearrange(query, f'b {channel_m_symbols} dv h -> b h ({channel_m_symbols}) dv')
            key = einops.rearrange(key, f'b {channel_m_symbols} dv h -> b h ({channel_m_symbols}) dv')
            value = einops.rearrange(value, f'b {channel_m_symbols} dv h -> b h ({channel_m_symbols}) dv')

            # Use scaled_dot_product_attention
            value = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                is_causal=False,
                scale=1.0 / self.scale  # The function applies scale internally
            )

            value = einops.rearrange(
                value,
                f'b h ({channel_m_symbols}) dv -> b {channel_m_symbols} dv h',
                **dim_dict)

        value_signature = f'b {channel_m_symbols} dv h, d dv h -> b {channel_m_symbols} d'
        value = einops.einsum(value, self.out_proj_tensor, value_signature)

        return value


class ConVitBlock(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_pos_dims: int,
                 ffn_expansion_factor: int = 4,
                 attn_compression_factor: int = 2,
                 num_heads: int = 8,
                 rope_freq: float = 1.0,
                 with_conv_on_upsample: bool = False,
                 with_conv_on_downsample: bool = False,
                 kernel_size_conv: int = 3,
                 kernel_size_depthwise: int = 3,
                 has_embedding: bool = False,
                 relative_positioning: bool = False,
                 linear_attention: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_pos_dims = num_pos_dims
        self.ffn_expansion_factor = ffn_expansion_factor
        self.attn_compression_factor = attn_compression_factor
        self.num_heads = num_heads
        self.rope_freq = rope_freq
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_depthwise = kernel_size_depthwise
        self.linear_attention = linear_attention

        self.norm_1 = ChannelRMSNorm(channel_dim=embed_dim)
        self.norm_2 = ChannelRMSNorm(channel_dim=embed_dim)
        self.attention = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_pos_dims=num_pos_dims,
            rope_freq=rope_freq,
            relative_positioning=relative_positioning,
            linear_attention=linear_attention
        )
        self.upsample = Upsample(
            num_pos_dims=num_pos_dims,
            channels_in=embed_dim,
            channels_out=embed_dim,
            expansion_factor=attn_compression_factor,
            with_conv=with_conv_on_upsample)
        self.downsample = Downsample(
            num_pos_dims=num_pos_dims,
            channels_in=embed_dim,
            channels_out=embed_dim,
            compression_factor=attn_compression_factor,
            with_conv=with_conv_on_downsample)
        self.ffn = ConvSwiGLU(
            embed_dim=embed_dim,
            num_pos_dims=num_pos_dims,
            expansion_factor=ffn_expansion_factor,
            kernel_size=kernel_size_conv)

        # New efficient conv path for fine details
        self.depthwise_conv = DimensionHelper.get_conv_cls(num_pos_dims)(
            embed_dim, embed_dim, kernel_size=kernel_size_depthwise, groups=embed_dim, padding='same'
        )
        self.pointwise_conv = DimensionHelper.get_conv_cls(num_pos_dims)(
            embed_dim, embed_dim, kernel_size=1
        )
        self.conv_activation = torch.nn.SiLU()

        self.fusion_weight = torch.nn.Parameter(torch.tensor(0.0))

        self.has_embedding = has_embedding
        if has_embedding:
            self.embedding_projection = SwiGLU(embed_dim=embed_dim, final_rms=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        channel_m_symbols = ' '.join(
            [f'm{i}' for i in range(self.num_pos_dims)]
        )
        if y is not None:
            if not self.has_embedding:
                raise ValueError("Conditional embedding is not supported when self.has_embedding=False")
            y = self.embedding_projection(y)
            y = y.reshape(y.shape[0], -1, *([1] * self.num_pos_dims))
        else:
            y = 0.0

        x0 = x.clone()
        x = self.norm_1(x) + y
        x = self.downsample(x)
        x = einops.rearrange(x, f'b d {channel_m_symbols} -> b {channel_m_symbols} d')
        x = self.attention(x)
        x = einops.rearrange(x, f'b {channel_m_symbols} d -> b d {channel_m_symbols}')
        x = self.upsample(x)

        # Convolution pathway
        x_conv = self.pointwise_conv(self.conv_activation(self.depthwise_conv(x)))

        # Fusion
        x = (1 - torch.sigmoid(self.fusion_weight)) * x + torch.sigmoid(self.fusion_weight) * x_conv

        x = x + x0
        x0 = x.clone()
        x = self.norm_2(x) + y
        x = self.ffn(x)
        x = x + x0
        return x


class ConVit(torch.nn.Module):
    def __init__(self,
                 config: ConVitConfig,
                 conditional_embedding: None | torch.nn.Module = None):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim
        out_channels = config.out_channels if config.out_channels is not None else config.in_channels
        self.out_channels = out_channels
        self.num_pos_dims = config.num_pos_dims
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.ffn_expansion_factor = config.ffn_expansion_factor
        self.attn_compression_factor = config.attn_compression_factor
        self.rope_freq = config.rope_freq
        self.with_conv_on_upsample = config.with_conv_on_upsample
        self.with_conv_on_downsample = config.with_conv_on_downsample
        self.kernel_size_conv = config.kernel_size_conv
        self.kernel_size_depthwise = config.kernel_size_depthwise
        self.kernel_size_in_out = config.kernel_size_in_out
        self.has_embedding = config.has_embedding
        self.has_conditional_embedding = config.has_conditional_embedding
        self.fourier_projection_scale = config.fourier_projection_scale
        self.relative_positioning = config.relative_positioning
        self.linear_attention = config.linear_attention
        self.input_batch_norm = config.input_batch_norm
        self.condition_dropout = config.condition_dropout

        if self.input_batch_norm:
            self.normin = DimensionHelper.get_batch_norm_cls(self.num_pos_dims)(self.in_channels)
        else:
            self.normin = torch.nn.Identity()

        self.convin = DimensionHelper.get_conv_cls(self.num_pos_dims)(
            self.in_channels,
            self.embed_dim,
            self.kernel_size_in_out,
            padding='same'
        )
        self.convout = DimensionHelper.get_conv_cls(self.num_pos_dims)(
            self.embed_dim,
            self.out_channels,
            self.kernel_size_in_out,
            padding='same'
        )
        self.normout = ChannelRMSNorm(channel_dim=self.embed_dim)

        self.blocks = torch.nn.ModuleList([
            ConVitBlock(
                embed_dim=self.embed_dim,
                num_pos_dims=self.num_pos_dims,
                ffn_expansion_factor=self.ffn_expansion_factor,
                attn_compression_factor=self.attn_compression_factor,
                num_heads=self.num_heads,
                rope_freq=self.rope_freq,
                with_conv_on_upsample=self.with_conv_on_upsample,
                with_conv_on_downsample=self.with_conv_on_downsample,
                kernel_size_conv=self.kernel_size_conv,
                kernel_size_depthwise=self.kernel_size_depthwise,
                has_embedding=self.has_embedding,
                relative_positioning=self.relative_positioning,
                linear_attention=self.linear_attention)
            for _ in range(self.num_layers)
        ])
        if self.condition_dropout > 0.0:
            self.condition_dropout_module = ConditionDropout(self.condition_dropout)
        else:
            self.condition_dropout_module = torch.nn.Identity()
        if config.has_time_embedding:
            self.time_embedding = GaussianFourierProjection(
                embed_dim=self.embed_dim, scale=self.fourier_projection_scale)
        if config.has_conditional_embedding:
            assert isinstance(conditional_embedding, torch.nn.Module)
            self.conditional_embedding = conditional_embedding
        else:
            if isinstance(conditional_embedding, None):
                warnings.warn("Conditional embedding is not supported when self.has_conditional_embedding=False")
                self.conditional_embedding = torch.nn.Identity()
            else:
                self.conditional_embedding = conditional_embedding

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None, y: torch.Tensor | None = None) -> torch.Tensor:
        te = self.time_embedding(t) if t is not None else 0.0
        ye = self.conditional_embedding(y) if y is not None else 0.0
        if self.condition_dropout > 0.0 and y is not None:
            ye = self.condition_dropout_module(ye)
        y = te + ye
        if not isinstance(y, torch.Tensor):
            y = None
        x = self.normin(x)
        x = self.convin(x)
        for block in self.blocks:
            x = block(x, y)
        x = self.normout(x)
        x = self.convout(x)
        return x

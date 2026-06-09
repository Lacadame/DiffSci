"""
Generalized embedder classes for conditioning diffusion models.

Base Classes:
    - ScalarEmbedder: Embeds a single scalar value from a dict key
    - VectorEmbedder: Embeds a vector from a dict key
    - FunctionEmbedder: Embeds a function (arg, values pairs)
    - SequenceTransformer: Wraps any sequence embedder with a transformer
    - CompositeEmbedder: Combines multiple embedders by summing
"""
from typing import Literal

import torch
import einops

from . import commonlayers


class PositionalEncoding1d(torch.nn.Module):
    """Sinusoidal positional encoding for 1D sequences."""

    def __init__(self, dembed: int, denominator: float = 10000.0):
        super().__init__()
        self.dembed = dembed
        self.denominator = denominator
        indexes = torch.arange(start=0, end=dembed, step=2)
        div_term = denominator ** (indexes / dembed)
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        # x : [..., seq_len]
        sin = torch.sin(x.unsqueeze(-1) / self.div_term)
        cos = torch.cos(x.unsqueeze(-1) / self.div_term)
        return torch.stack([sin, cos], dim=-1).flatten(start_dim=-2)


class ScalarEmbedder(torch.nn.Module):
    """
    Embeds a scalar value from a dictionary.

    Args:
        dembed: Embedding dimension
        key: Dictionary key to extract the scalar from
        scale: Scale for Gaussian Fourier projection
        mlp_expansion: MLP hidden dimension multiplier

    Example:
        >>> embedder = ScalarEmbedder(256, key='porosity')
        >>> data = {'porosity': torch.tensor([[0.3], [0.5]])}
        >>> embedding = embedder(data)  # [2, 256]
    """

    def __init__(
        self,
        dembed: int,
        key: str,
        scale: float = 30.0,
        mlp_expansion: int = 4
    ):
        super().__init__()
        self.dembed = dembed
        self.key = key
        self.scale = scale

        self.gaussian_proj = commonlayers.GaussianFourierProjection(dembed, scale)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dembed, mlp_expansion * dembed),
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_expansion * dembed, mlp_expansion * dembed),
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_expansion * dembed, dembed)
        )

    def forward(self, data: dict) -> torch.Tensor:
        x = data[self.key]
        if x.ndim >= 1 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        ndim = x.ndim
        if ndim == 1:
            return self.net(self.gaussian_proj(x))
        elif ndim > 1:
            # [batch, *spatial] -> flatten, embed, reshape
            shape_strings = ' '.join([f's{dim}' for dim in range(1, ndim)])
            kwargs = {f's{dim}': x.shape[dim] for dim in range(1, ndim)}
            x_flat = einops.rearrange(x, f'nbatch {shape_strings} -> (nbatch {shape_strings})', **kwargs)
            y_flat = self.net(self.gaussian_proj(x_flat))
            return einops.rearrange(y_flat, f'(nbatch {shape_strings}) dembed -> nbatch dembed {shape_strings}', **kwargs)
        else:
            raise ValueError(f"Invalid dimensions: {ndim}")


class VectorEmbedder(torch.nn.Module):
    """
    Embeds a vector from a dictionary.

    Args:
        dembed: Embedding dimension
        input_dim: Dimension of input vector
        key: Dictionary key to extract the vector from
        scale: Scale for Gaussian Fourier projection
        fourier: Whether to use Gaussian Fourier features

    Example:
        >>> embedder = VectorEmbedder(256, input_dim=4, key='momenta')
        >>> data = {'momenta': torch.randn(2, 4)}
        >>> embedding = embedder(data)  # [2, 256]
    """

    def __init__(
        self,
        dembed: int,
        input_dim: int,
        key: str,
        scale: float = 30.0,
        mlp_expansion: int = 4,
        fourier: bool = True
    ):
        super().__init__()
        self.dembed = dembed
        self.input_dim = input_dim
        self.key = key
        self.fourier = fourier

        if fourier:
            self.gaussian_proj = commonlayers.GaussianFourierProjectionVector(input_dim, dembed, scale)
            first_dim = dembed
        else:
            first_dim = input_dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(first_dim, mlp_expansion * dembed),
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_expansion * dembed, mlp_expansion * dembed),
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_expansion * dembed, dembed)
        )

    def forward(self, data: dict) -> torch.Tensor:
        x = data[self.key]
        if self.fourier:
            return self.net(self.gaussian_proj(x))
        return self.net(x)


class FunctionEmbedder(torch.nn.Module):
    """
    Embeds a function represented as (argument, values) pairs.

    Uses positional encoding for arguments and Gaussian Fourier projection
    for values, then combines them additively.

    Args:
        dembed: Embedding dimension
        arg_key: Dictionary key for function arguments (x-axis)
        values_key: Dictionary key for function values (y-axis)
        reduction: How to reduce sequence dimension ('mean' or None)
        scale: Scale for Gaussian Fourier projection
        values_transform: Transform for values ('none' or 'neglog')

    Example:
        >>> embedder = FunctionEmbedder(256, arg_key='distance', values_key='correlation')
        >>> data = {'distance': torch.linspace(0, 1, 10).unsqueeze(0),
        ...         'correlation': torch.rand(1, 10)}
        >>> embedding = embedder(data)  # [1, 10, 256] or [1, 256] if reduction='mean'
    """

    def __init__(
        self,
        dembed: int,
        arg_key: str,
        values_key: str,
        reduction: Literal['mean', None] = None,
        scale: float = 30.0,
        values_transform: Literal['none', 'neglog'] = 'none'
    ):
        super().__init__()
        self.dembed = dembed
        self.arg_key = arg_key
        self.values_key = values_key
        self.reduction = reduction
        self.values_transform = values_transform

        self.pos_encoder = PositionalEncoding1d(dembed)
        self.gaussian_proj = commonlayers.GaussianFourierProjection(dembed, scale)

    def forward(self, data: dict) -> torch.Tensor:
        arg = data[self.arg_key]
        values = data[self.values_key]

        if self.values_transform == 'neglog':
            values = -torch.log(values + 1e-6)

        x = self.pos_encoder(arg) + self.gaussian_proj(values)

        if self.reduction == 'mean':
            x = x.mean(dim=-2)
        return x


class SequenceTransformer(torch.nn.Module):
    """
    Wraps a sequence embedder with a transformer encoder.

    Args:
        embedder: A sequence embedder (e.g., FunctionEmbedder with reduction=None)
        nhead: Number of attention heads
        ffn_expansion: Feedforward network expansion factor
        num_layers: Number of transformer layers
        reduction: How to reduce sequence after transformer ('mean' or 'first')
    """

    def __init__(
        self,
        embedder: torch.nn.Module,
        nhead: int = 4,
        ffn_expansion: int = 4,
        num_layers: int = 2,
        reduction: Literal['mean', 'first'] = 'mean'
    ):
        super().__init__()
        self.embedder = embedder
        self.reduction = reduction

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=embedder.dembed,
                nhead=nhead,
                dim_feedforward=embedder.dembed * ffn_expansion,
                batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, data: dict) -> torch.Tensor:
        x = self.embedder(data)
        x = self.encoder(x)
        if self.reduction == 'mean':
            return x.mean(dim=-2)
        elif self.reduction == 'first':
            return x[..., 0, :]
        return x


class CompositeEmbedder(torch.nn.Module):
    """
    Combines multiple embedders by summing their outputs.

    Example:
        >>> composite = CompositeEmbedder([
        ...     ScalarEmbedder(256, key='porosity'),
        ...     VectorEmbedder(256, 4, key='momenta')
        ... ])
    """

    def __init__(self, embedders: list[torch.nn.Module]):
        super().__init__()
        self.embedders = torch.nn.ModuleList(embedders)

    def forward(self, data: dict) -> torch.Tensor:
        return sum(embedder(data) for embedder in self.embedders)
